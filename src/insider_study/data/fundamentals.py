"""Point-in-time shares outstanding and book equity from SEC XBRL frames (2009+).

The frames API returns, per calendar quarter, one value per company. A value is
treated as public only `lag_days` after its period end (conservative):
- dei:EntityCommonStockSharesOutstanding is a cover-page figure dated near the
  filing itself -> 30 days.
- us-gaap balance-sheet items (shares, equity) -> 90 days.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from insider_study.logging_setup import get_logger

log = get_logger(__name__)

FRAMES_URL = "https://data.sec.gov/api/xbrl/frames"
CONCEPTS = {
    "shares_dei": ("dei", "EntityCommonStockSharesOutstanding", "shares", 30),
    "shares_gaap": ("us-gaap", "CommonStockSharesOutstanding", "shares", 90),
    "book_equity": ("us-gaap", "StockholdersEquity", "USD", 90),
}


@dataclass
class SECFramesProvider:
    cache_dir: Path
    user_agent: str
    first_year: int = 2009
    last_year: int = 2025

    def _fetch(self, taxonomy: str, tag: str, unit: str, period: str) -> list[dict[str, object]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{tag}_{period}.json"
        if not path.exists():
            url = f"{FRAMES_URL}/{taxonomy}/{tag}/{unit}/{period}.json"
            for attempt in range(4):
                time.sleep(0.15)
                resp = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=60)
                if resp.status_code == 200:
                    path.write_text(resp.text)
                    break
                if resp.status_code == 404:
                    path.write_text('{"data": []}')
                    break
                time.sleep(2**attempt)
            else:
                raise RuntimeError(f"failed {url}")
        data: list[dict[str, object]] = json.loads(path.read_text()).get("data", [])
        return data

    def concept(self, name: str) -> pd.DataFrame:
        taxonomy, tag, unit, lag = CONCEPTS[name]
        rows = []
        for y in range(self.first_year, self.last_year + 1):
            for q in (1, 2, 3, 4):
                for rec in self._fetch(taxonomy, tag, unit, f"CY{y}Q{q}I"):
                    rows.append((str(rec["cik"]), rec["end"], rec["val"]))
        df = pd.DataFrame(rows, columns=["issuer_cik", "period_end", "value"])
        df["period_end"] = pd.to_datetime(df["period_end"])
        df["known_at"] = df["period_end"] + pd.Timedelta(days=lag)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.loc[df["value"] > 0] if name.startswith("shares") else df.dropna(subset=["value"])
        log.info("frames_concept", name=name, rows=len(df))
        return df.drop_duplicates(["issuer_cik", "period_end"]).sort_values("known_at")

    def shares_outstanding(self) -> pd.DataFrame:
        """dei cover value preferred; us-gaap balance-sheet value as fallback."""
        dei = self.concept("shares_dei").assign(source="dei")
        gaap = self.concept("shares_gaap").assign(source="gaap")
        both = pd.concat([dei, gaap], ignore_index=True)
        both["_prio"] = both["source"].map({"dei": 0, "gaap": 1})
        both = both.sort_values(["issuer_cik", "period_end", "_prio"])
        out = both.drop_duplicates(["issuer_cik", "period_end"]).drop(columns="_prio")
        return out.sort_values("known_at").reset_index(drop=True)
