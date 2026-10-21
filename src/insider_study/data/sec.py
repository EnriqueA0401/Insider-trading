"""SEC EDGAR Insider Transactions Data Sets (Forms 3/4/5).

Source: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
Quarterly zips (by filing quarter) are available from 2006Q1. Each zip holds
SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, FOOTNOTES, ... as TSV.

The parser keeps every open-market trade (codes P and S) at line level and stores
only counts for other codes, which are excluded from the signal by design.
"""

from __future__ import annotations

import csv
import io
import re
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from insider_study.domain import Col
from insider_study.logging_setup import get_logger
from insider_study.reproducibility import DataManifest

log = get_logger(__name__)

BASE_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets"
FIRST_AVAILABLE_YEAR = 2006
OPEN_MARKET_CODES = ("P", "S")

# Rule 10b5-1 references in footnotes/remarks (pre-2023 there is no checkbox).
_RE_10B5_1 = re.compile(
    r"10\s*b\s*-?\s*5\s*-?\s*1|rule\s+10b5|pre-?arranged\s+(?:stock\s+)?trading\s+plan",
    re.IGNORECASE,
)
_FN_ID = re.compile(r"F\d+")


def quarter_keys(years: Sequence[int]) -> list[str]:
    return [f"{y}q{q}" for y in years for q in (1, 2, 3, 4)]


@dataclass
class SECForm345Ingestor:
    """Downloads quarterly zips politely (SEC fair-access: UA + <=10 req/s)."""

    raw_dir: Path
    user_agent: str
    max_requests_per_second: float = 8.0
    manifest_path: Path | None = None

    def _url(self, key: str) -> str:
        return f"{BASE_URL}/{key}_form345.zip"

    def fetch(self, years: Sequence[int]) -> list[Path]:
        if min(years) < FIRST_AVAILABLE_YEAR:
            raise ValueError(f"SEC structured data starts in {FIRST_AVAILABLE_YEAR}")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        manifest = DataManifest(self.manifest_path) if self.manifest_path else None
        session = requests.Session()
        session.headers.update({"User-Agent": self.user_agent, "Accept-Encoding": "gzip"})
        paths: list[Path] = []
        for key in quarter_keys(years):
            dest = self.raw_dir / f"{key}_form345.zip"
            if not dest.exists():
                self._download(session, self._url(key), dest)
            paths.append(dest)
            if manifest is not None:
                manifest.record(dest)
        if manifest is not None:
            manifest.save()
        return paths

    def _download(self, session: requests.Session, url: str, dest: Path) -> None:
        for attempt in range(5):
            time.sleep(1.0 / self.max_requests_per_second)
            resp = session.get(url, timeout=120)
            if resp.status_code == 200:
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(resp.content)
                tmp.rename(dest)
                log.info("sec_download", url=url, bytes=len(resp.content))
                return
            if resp.status_code == 404:
                raise FileNotFoundError(url)
            log.warning("sec_retry", url=url, status=resp.status_code, attempt=attempt)
            time.sleep(2**attempt)
        raise RuntimeError(f"failed to download {url}")


def _read_tsv(zf: zipfile.ZipFile, name: str, usecols: Sequence[str]) -> pd.DataFrame:
    with zf.open(name) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
        frame = pd.read_csv(
            text,
            sep="\t",
            dtype=str,
            usecols=lambda c: c in set(usecols),
            quoting=csv.QUOTE_NONE,
            keep_default_na=False,
            on_bad_lines="skip",
        )
    # Older quarters lack some columns (e.g. AFF10B5ONE before 2023): fill as empty.
    return frame.reindex(columns=list(usecols), fill_value="")


def _date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d-%b-%Y", errors="coerce").dt.date


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")


_FN_COLS = (
    "SECURITY_TITLE_FN",
    "TRANS_DATE_FN",
    "TRANS_SHARES_FN",
    "TRANS_PRICEPERSHARE_FN",
    "TRANS_ACQUIRED_DISP_CD_FN",
    "SHRS_OWND_FOLWNG_TRANS_FN",
    "DIRECT_INDIRECT_OWNERSHIP_FN",
    "NATURE_OF_OWNERSHIP_FN",
    "TRANS_TIMELINESS_FN",
)


def _aggregate_owners(owners: pd.DataFrame) -> pd.DataFrame:
    """One row per accession. Joint filings: prefer an officer/director as the insider."""
    rel = owners["RPTOWNER_RELATIONSHIP"].str.lower()
    owners = owners.assign(
        _is_dir=rel.str.contains("director"),
        _is_off=rel.str.contains("officer"),
        _is_ten=rel.str.contains("tenpercent"),
        _rank=(~rel.str.contains("officer")).astype(int) * 2
        + (~rel.str.contains("director")).astype(int),
    ).sort_values(["ACCESSION_NUMBER", "_rank"])
    agg = owners.groupby("ACCESSION_NUMBER", sort=False).agg(
        insider_cik=("RPTOWNERCIK", "first"),
        insider_name=("RPTOWNERNAME", "first"),
        relationship=("RPTOWNER_RELATIONSHIP", "first"),
        raw_title=("RPTOWNER_TITLE", lambda s: " | ".join(t for t in s if t)),
        is_director=("_is_dir", "any"),
        is_officer=("_is_off", "any"),
        is_ten_pct_owner=("_is_ten", "any"),
        n_owners=("RPTOWNERCIK", "nunique"),
    )
    return agg.reset_index()


def parse_quarter(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (open-market lines with P/S codes, counts of all codes by form type)."""
    with zipfile.ZipFile(zip_path) as zf:
        sub = _read_tsv(
            zf,
            "SUBMISSION.tsv",
            [
                "ACCESSION_NUMBER",
                "FILING_DATE",
                "PERIOD_OF_REPORT",
                "DATE_OF_ORIG_SUB",
                "DOCUMENT_TYPE",
                "ISSUERCIK",
                "ISSUERNAME",
                "ISSUERTRADINGSYMBOL",
                "REMARKS",
                "AFF10B5ONE",
            ],
        )
        owners = _read_tsv(
            zf,
            "REPORTINGOWNER.tsv",
            [
                "ACCESSION_NUMBER",
                "RPTOWNERCIK",
                "RPTOWNERNAME",
                "RPTOWNER_RELATIONSHIP",
                "RPTOWNER_TITLE",
            ],
        )
        trans = _read_tsv(
            zf,
            "NONDERIV_TRANS.tsv",
            [
                "ACCESSION_NUMBER",
                "NONDERIV_TRANS_SK",
                "SECURITY_TITLE",
                "TRANS_DATE",
                "TRANS_CODE",
                "TRANS_TIMELINESS",
                "TRANS_SHARES",
                "TRANS_PRICEPERSHARE",
                "TRANS_ACQUIRED_DISP_CD",
                "SHRS_OWND_FOLWNG_TRANS",
                "DIRECT_INDIRECT_OWNERSHIP",
                *_FN_COLS,
            ],
        )
        footnotes = _read_tsv(
            zf, "FOOTNOTES.tsv", ["ACCESSION_NUMBER", "FOOTNOTE_ID", "FOOTNOTE_TXT"]
        )

    trans = trans.merge(
        sub[["ACCESSION_NUMBER", "DOCUMENT_TYPE"]], on="ACCESSION_NUMBER", how="left"
    )
    counts = trans.groupby(["DOCUMENT_TYPE", "TRANS_CODE"]).size().rename("n_lines").reset_index()
    counts["quarter"] = zip_path.name.split("_")[0]

    om = trans.loc[trans["TRANS_CODE"].isin(OPEN_MARKET_CODES)].copy()

    # 10b5-1 flag from referenced footnotes.
    fn_flag = footnotes.assign(
        is_plan=footnotes["FOOTNOTE_TXT"].str.contains(_RE_10B5_1, na=False)
    ).loc[lambda d: d["is_plan"], ["ACCESSION_NUMBER", "FOOTNOTE_ID"]]
    plan_fns = set(zip(fn_flag["ACCESSION_NUMBER"], fn_flag["FOOTNOTE_ID"], strict=True))
    refs = om[list(_FN_COLS)].agg(" ".join, axis=1).str.findall(_FN_ID)
    om["fn_10b5_1"] = [
        any((acc, f) in plan_fns for f in ids)
        for acc, ids in zip(om["ACCESSION_NUMBER"], refs, strict=True)
    ]

    df = om.merge(sub.drop(columns="DOCUMENT_TYPE"), on="ACCESSION_NUMBER", how="left").merge(
        _aggregate_owners(owners), on="ACCESSION_NUMBER", how="left"
    )
    checkbox = df["AFF10B5ONE"].str.lower().isin(["1", "true"])
    remarks_plan = df["REMARKS"].str.contains(_RE_10B5_1, na=False)

    out = pd.DataFrame(
        {
            Col.ACCESSION: df["ACCESSION_NUMBER"],
            "line_id": df["NONDERIV_TRANS_SK"],
            Col.FORM_TYPE: df["DOCUMENT_TYPE"],
            Col.FILING_DATE: _date(df["FILING_DATE"]),
            "date_of_orig_sub": _date(df["DATE_OF_ORIG_SUB"]),
            Col.TRANSACTION_DATE: _date(df["TRANS_DATE"]),
            Col.ISSUER_CIK: df["ISSUERCIK"].str.lstrip("0"),
            "issuer_name": df["ISSUERNAME"],
            Col.TICKER: df["ISSUERTRADINGSYMBOL"].str.upper().str.strip(),
            Col.INSIDER_CIK: df["insider_cik"].str.lstrip("0"),
            Col.INSIDER_NAME: df["insider_name"],
            "relationship": df["relationship"],
            Col.RAW_TITLE: df["raw_title"],
            Col.IS_DIRECTOR: df["is_director"].fillna(False).astype(bool),
            Col.IS_OFFICER: df["is_officer"].fillna(False).astype(bool),
            Col.IS_TEN_PCT_OWNER: df["is_ten_pct_owner"].fillna(False).astype(bool),
            "n_owners": df["n_owners"].fillna(0).astype(int),
            "security_title": df["SECURITY_TITLE"],
            Col.TRANS_CODE: df["TRANS_CODE"],
            "timeliness": df["TRANS_TIMELINESS"],
            Col.ACQ_DISP: df["TRANS_ACQUIRED_DISP_CD"],
            Col.SHARES: _num(df["TRANS_SHARES"]),
            Col.PRICE: _num(df["TRANS_PRICEPERSHARE"]),
            Col.SHARES_OWNED_AFTER: _num(df["SHRS_OWND_FOLWNG_TRANS"]),
            "direct_indirect": df["DIRECT_INDIRECT_OWNERSHIP"],
            "is_10b5_1_checkbox": checkbox,
            Col.IS_10B5_1: checkbox | remarks_plan | df["fn_10b5_1"],
            Col.IS_AMENDMENT: df["DOCUMENT_TYPE"].str.endswith("/A"),
        }
    )
    return out, counts


def parse_all(zip_paths: Sequence[Path], out_dir: Path) -> tuple[Path, Path]:
    """Parse every quarter into two parquet files (open-market lines, code counts)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, counts = [], []
    for path in zip_paths:
        lines, c = parse_quarter(path)
        frames.append(lines)
        counts.append(c)
        log.info("sec_parsed", quarter=path.name, open_market_lines=len(lines))
    lines_path = out_dir / "open_market_lines.parquet"
    counts_path = out_dir / "code_counts.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(lines_path, index=False)
    pd.concat(counts, ignore_index=True).to_parquet(counts_path, index=False)
    return lines_path, counts_path
