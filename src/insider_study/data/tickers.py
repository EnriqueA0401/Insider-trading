"""Map events to a validated price series.

Candidates per event: (1) the ticker reported on the filing (point-in-time),
(2) the issuer's current ticker from SEC company_tickers.json (handles renames).
A candidate is accepted only if its as-traded close on the last trade date is
within `max_price_deviation` of the Form 4 price and it trades on the entry date.
This rejects re-used tickers that now belong to a different company.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from insider_study.data.prices import YFinancePriceProvider, to_yahoo
from insider_study.domain import Col

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def current_sec_tickers(cache: Path, user_agent: str) -> dict[str, str]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(SEC_TICKERS_URL, headers={"User-Agent": user_agent}, timeout=60)
        resp.raise_for_status()
        cache.write_text(resp.text)
    raw = json.loads(cache.read_text())
    out: dict[str, str] = {}
    for rec in raw.values():
        out.setdefault(str(rec["cik_str"]), str(rec["ticker"]))
    return out


def candidate_tickers(events: pd.DataFrame, current: dict[str, str]) -> pd.DataFrame:
    pit = events[Col.TICKER].fillna("").map(to_yahoo)
    cur = events[Col.ISSUER_CIK].map(current).fillna("").map(lambda t: to_yahoo(t) if t else "")
    return pd.DataFrame({"cand_pit": pit, "cand_current": cur}, index=events.index)


def validate_events(
    events: pd.DataFrame,
    cands: pd.DataFrame,
    provider: YFinancePriceProvider,
    max_dev: float,
) -> pd.DataFrame:
    """Return per-event: yf_ticker, price_status, price_dev."""
    status = pd.Series("no_series", index=events.index, dtype="object")
    chosen = pd.Series("", index=events.index, dtype="object")
    dev_out = pd.Series(np.nan, index=events.index)
    last_trade = pd.to_datetime(events["last_trade_date"])
    entry = pd.to_datetime(events[Col.ENTRY_DATE])
    vwap = events["vwap"]

    for col in ("cand_pit", "cand_current"):
        pending = status.ne("priced") & cands[col].ne("")
        for ticker, idx in cands.loc[pending].groupby(col).groups.items():
            frame = provider.load(str(ticker))
            if frame is None:
                continue
            closes = frame["raw_close"].dropna()
            sessions = frame.index
            # As-traded close on (or just before) the trade date, at most 5 days stale.
            pos = sessions.searchsorted(last_trade.loc[idx], side="right") - 1
            ok_pos = pos >= 0
            ref_dates = sessions[np.clip(pos, 0, len(sessions) - 1)]
            fresh = ok_pos & ((last_trade.loc[idx].to_numpy() - ref_dates.to_numpy()) <= np.timedelta64(5, "D"))
            ref = closes.reindex(ref_dates).to_numpy()
            dev = vwap.loc[idx].to_numpy() / ref - 1.0
            trades_at_entry = entry.loc[idx].isin(sessions).to_numpy()
            good = fresh & trades_at_entry & (np.abs(dev) <= max_dev)
            bad_price = fresh & trades_at_entry & ~good
            idx_arr = np.asarray(idx)
            status.loc[idx_arr[good]] = "priced"
            chosen.loc[idx_arr[good]] = ticker
            dev_out.loc[idx_arr[good]] = dev[good]
            mism = bad_price & status.loc[idx_arr].ne("priced").to_numpy()
            status.loc[idx_arr[mism]] = "price_mismatch"
            dev_out.loc[idx_arr[mism]] = dev[mism]
    return pd.DataFrame({"yf_ticker": chosen, "price_status": status, "price_dev": dev_out})
