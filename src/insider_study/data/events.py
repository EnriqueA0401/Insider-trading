"""Aggregate eligible purchase lines into filing-level events.

Event = (insider, issuer, filing). Several lines in one filing (partial fills,
direct + indirect accounts) are summed. Every event is dated by its FILING date;
the entry session is strictly after it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insider_study.config import StudyConfig
from insider_study.domain import Col
from insider_study.holdout import assign_split
from insider_study.temporal import assert_no_lookahead, entry_dates

ROLE_FLAGS = (
    "is_cfo",
    "is_cfo_broad",
    "is_ceo",
    "is_officer_any",
    "is_director_only",
    "is_ten_pct_only",
    "is_dual_ceo_cfo",
)
DELTA_H_CAP = 10.0  # +1000%: new or near-new positions


def build_events(lines: pd.DataFrame, cfg: StudyConfig) -> pd.DataFrame:
    df = lines.assign(
        value_usd=lines[Col.SHARES] * lines[Col.PRICE],
        _direct_shares=np.where(lines["direct_indirect"].eq("D"), lines[Col.SHARES], 0.0),
        _direct_after=np.where(
            lines["direct_indirect"].eq("D"), lines[Col.SHARES_OWNED_AFTER], np.nan
        ),
    )
    grouped = df.groupby(Col.ACCESSION, sort=False)
    agg: dict[str, tuple[str, str]] = {
        Col.FILING_DATE: (Col.FILING_DATE, "first"),
        Col.FORM_TYPE: (Col.FORM_TYPE, "first"),
        Col.ISSUER_CIK: (Col.ISSUER_CIK, "first"),
        "issuer_name": ("issuer_name", "first"),
        Col.TICKER: (Col.TICKER, "first"),
        Col.INSIDER_CIK: (Col.INSIDER_CIK, "first"),
        Col.INSIDER_NAME: (Col.INSIDER_NAME, "first"),
        Col.RAW_TITLE: (Col.RAW_TITLE, "first"),
        Col.ROLE: (Col.ROLE, "first"),
        "first_trade_date": (Col.TRANSACTION_DATE, "min"),
        "last_trade_date": (Col.TRANSACTION_DATE, "max"),
        "filing_lag_days": ("filing_lag_days", "max"),
        Col.SHARES: (Col.SHARES, "sum"),
        "value_usd": ("value_usd", "sum"),
        "direct_shares": ("_direct_shares", "sum"),
        "direct_after": ("_direct_after", "max"),
        "n_lines": (Col.SHARES, "size"),
        "n_owners": ("n_owners", "max"),
    }
    agg |= {flag: (flag, "max") for flag in ROLE_FLAGS}
    ev = grouped.agg(**agg).reset_index()
    ev["vwap"] = ev["value_usd"] / ev[Col.SHARES]
    before = ev["direct_after"] - ev["direct_shares"]
    delta_h = np.where(before > 0, ev["direct_shares"] / before, DELTA_H_CAP)
    ev["delta_h"] = np.where(ev["direct_shares"] > 0, np.minimum(delta_h, DELTA_H_CAP), np.nan)

    ev[Col.ENTRY_DATE] = entry_dates(ev[Col.FILING_DATE], cfg.signal.entry_lag_sessions)
    assert_no_lookahead(ev)
    ev[Col.SPLIT] = assign_split(ev[Col.FILING_DATE], cfg)
    return ev.sort_values([Col.FILING_DATE, Col.ACCESSION]).reset_index(drop=True)
