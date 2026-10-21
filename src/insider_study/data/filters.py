"""Transaction filtering, amendment resolution and de-duplication.

Order of operations (each exclusion is counted and reported):
  1. resolve amendments (4/A matched to its original are dropped; signal stays at
     the original filing date with the originally reported values),
  2. drop cross-filed duplicates (same trade reported by several related owners),
  3. split into eligible purchases / 10b5-1 purchases / sales / excluded.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from insider_study.config import FilterConfig
from insider_study.domain import Col
from insider_study.interfaces import FilterResult

_NON_COMMON = re.compile(
    r"prefer|pfd|warrant|\bnote|debenture|\bunit|\bright|option|bond|convertible|"
    r"depositary|\badr\b|\bads\b|limited\s+partner|\blp\s+unit|trust\s+pref",
    re.IGNORECASE,
)
_BAD_TICKERS = {"", "NONE", "N/A", "NA", "NULL", "-", "--", "0", "."}


def resolve_amendments(lines: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop 4/A lines whose original Form 4 is in the dataset.

    Returns (lines, n_amendment_lines_dropped, n_orphan_amendment_lines_kept).
    Orphan amendments (original missing, e.g. paper filing) become filings dated at
    their own (later) filing date, which is conservative.
    """
    is_amend = lines[Col.FORM_TYPE].eq("4/A")
    originals = lines.loc[
        lines[Col.FORM_TYPE].eq("4"), [Col.ISSUER_CIK, Col.INSIDER_CIK, Col.FILING_DATE]
    ]
    keys = set(originals.itertuples(index=False, name=None))
    amend = lines.loc[is_amend]
    matched = pd.Series(
        [
            (i, o, d) in keys
            for i, o, d in zip(
                amend[Col.ISSUER_CIK],
                amend[Col.INSIDER_CIK],
                amend["date_of_orig_sub"],
                strict=True,
            )
        ],
        index=amend.index,
        dtype=bool,
    )
    drop_idx = matched[matched].index
    return lines.drop(index=drop_idx), len(drop_idx), int((~matched).sum())


def drop_cross_filed_duplicates(lines: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Same issuer/trade date/shares/price reported under several owner CIKs.

    Typical case: a fund, its general partner and its manager each file for one
    trade, sometimes on different days. Keep one line, preferring an officer, then a
    director (a later filing date for the kept line is conservative, never look-ahead).
    """
    key = [Col.ISSUER_CIK, Col.TRANSACTION_DATE, Col.SHARES, Col.PRICE, Col.TRANS_CODE]
    rank = (~lines[Col.IS_OFFICER]).astype(int) * 2 + (~lines[Col.IS_DIRECTOR]).astype(int)
    # Officer first, then director; ties broken by the earliest filing.
    ordered = lines.assign(_rank=rank).sort_values([*key, "_rank", Col.FILING_DATE])
    # Only collapse across *different* insiders; repeated identical lines from one
    # insider in one filing are legitimate partial fills and are aggregated later.
    diff_insider = ordered.groupby(key, dropna=False)[Col.INSIDER_CIK].transform("nunique").gt(1)
    first_insider = ordered.groupby(key, dropna=False)[Col.INSIDER_CIK].transform("first")
    drop = diff_insider & ordered[Col.INSIDER_CIK].ne(first_insider)
    return ordered.loc[~drop].drop(columns="_rank"), int(drop.sum())


@dataclass
class SECTransactionFilter:
    cfg: FilterConfig

    def apply(self, lines: pd.DataFrame) -> FilterResult:
        counts: Counter[str] = Counter()
        df, n_amend, n_orphan = resolve_amendments(lines)
        counts["amendment_matched_dropped"] = n_amend
        counts["amendment_orphan_kept"] = n_orphan
        df, n_dup = drop_cross_filed_duplicates(df)
        counts["cross_filed_duplicate"] = n_dup

        lag = (
            pd.to_datetime(df[Col.FILING_DATE]) - pd.to_datetime(df[Col.TRANSACTION_DATE])
        ).dt.days
        df = df.assign(filing_lag_days=lag)

        sales = df.loc[df[Col.TRANS_CODE].eq("S") & df[Col.ACQ_DISP].eq("D")]
        df = df.loc[df[Col.TRANS_CODE].isin(self.cfg.eligible_trans_codes)]

        rules: list[tuple[str, pd.Series]] = [
            ("form_type_not_eligible", ~df[Col.FORM_TYPE].isin(self.cfg.eligible_form_types)),
            ("not_acquisition", ~df[Col.ACQ_DISP].eq("A")),
            ("missing_dates", df[Col.FILING_DATE].isna() | df[Col.TRANSACTION_DATE].isna()),
            ("non_positive_price_or_shares", ~(df[Col.PRICE].gt(0) & df[Col.SHARES].gt(0))),
            ("non_common_security", df["security_title"].str.contains(_NON_COMMON, na=False)),
            ("negative_filing_lag", df["filing_lag_days"].lt(0)),
            ("missing_ticker", df[Col.TICKER].fillna("").isin(_BAD_TICKERS)),
        ]
        if self.cfg.max_filing_lag_days is not None:
            rules.append(
                ("filing_lag_too_long", df["filing_lag_days"].gt(self.cfg.max_filing_lag_days))
            )

        excluded = pd.Series(False, index=df.index)
        for name, mask in rules:
            new = mask.fillna(True) & ~excluded
            counts[name] = int(new.sum())
            excluded |= new
        kept = df.loc[~excluded]

        plan_mask = (
            kept[Col.IS_10B5_1] if self.cfg.exclude_10b5_1 else pd.Series(False, index=kept.index)
        )
        counts["plan_10b5_1_segregated"] = int(plan_mask.sum())
        return FilterResult(
            eligible=kept.loc[~plan_mask].copy(),
            sales=sales.copy(),
            plan_10b5_1=kept.loc[plan_mask].copy(),
            exclusion_counts=dict(counts),
        )
