"""Analysis-ready market data built from the cached daily price files.

Monthly panel (one row per rebalance session x ticker):
- ret_nodl: open-to-open return from this rebalance to the next (first NYSE session
  of each month). If the series stops before the next rebalance and before the data
  end, the stock is flagged `delisted` and valued at its last close; the delisting
  return is applied later (it is a sensitivity parameter).
- adv20 (USD), sigma60 (daily), half_spread (Corwin-Schultz, prior 21 sessions),
  raw_px: all measured strictly BEFORE the rebalance session.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from insider_study.data.prices import YFinancePriceProvider
from insider_study.temporal import _nyse

_K = 3.0 - 2.0 * np.sqrt(2.0)


def rebalance_sessions(start: date, end: date) -> pd.DatetimeIndex:
    """First NYSE session of every month in [start, end]."""
    sessions = _nyse().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    s = pd.Series(sessions, index=sessions)
    first = s.groupby(sessions.to_period("M")).min()
    return pd.DatetimeIndex(first.to_numpy())


def corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily two-day Corwin-Schultz (2012) spread estimate; negatives set to 0."""
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(-1)
    hi2 = pd.concat([high, high.shift(-1)], axis=1).max(axis=1)
    lo2 = pd.concat([low, low.shift(-1)], axis=1).min(axis=1)
    gamma = np.log(hi2 / lo2) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0.0)


def monthly_panel_for_ticker(
    frame: pd.DataFrame, rebal: pd.DatetimeIndex, data_end: pd.Timestamp
) -> pd.DataFrame:
    f = frame.loc[frame.index <= data_end]
    if len(f) < 25:
        return pd.DataFrame()
    sessions = f.index
    last_session = sessions[-1]
    daily_ret = f["adj_close"].pct_change()
    adv20 = f["dollar_volume"].rolling(20, min_periods=10).mean().shift(1)
    sigma60 = daily_ret.rolling(60, min_periods=20).std().shift(1)
    cs = corwin_schultz(f["high"], f["low"])
    # Spread of day t uses days t and t+1; shift by 2 so only past days enter.
    hs = (cs.rolling(21, min_periods=10).mean() / 2.0).shift(2)
    raw_px = f["raw_close"].shift(1)

    r = rebal[(rebal >= sessions[0]) & (rebal <= last_session)]
    r = r[r.isin(sessions)]
    if len(r) == 0:
        return pd.DataFrame()
    pos_next = rebal.searchsorted(r, side="right")
    has_next = pos_next < len(rebal)
    nxt = pd.DatetimeIndex(
        np.where(
            has_next, rebal[pos_next.clip(max=len(rebal) - 1)], data_end + pd.Timedelta(days=1)
        )
    )
    observable = has_next & (nxt <= data_end)
    next_traded = observable & nxt.isin(sessions)
    # Otherwise value at the last close before the next rebalance (or data end).
    idx_last = sessions.searchsorted(nxt, side="left") - 1
    px_open = f["adj_open"].reindex(r).to_numpy()
    px_end = np.where(
        next_traded,
        f["adj_open"].reindex(nxt).to_numpy(),
        f["adj_close"].to_numpy()[idx_last],
    )
    # Series ends inside the month, well before the data end -> delisting.
    delisted = observable & (last_session < nxt) & ((data_end - last_session).days > 7)
    out = pd.DataFrame(
        {
            "rebal": r,
            "ret_nodl": px_end / px_open - 1.0,
            "delisted": delisted,
            "adv20": adv20.reindex(r).to_numpy(),
            "sigma60": sigma60.reindex(r).to_numpy(),
            "half_spread": hs.reindex(r).to_numpy(),
            "raw_px": raw_px.reindex(r).to_numpy(),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)


def build_monthly_panel(
    provider: YFinancePriceProvider,
    tickers: Sequence[str],
    rebal: pd.DatetimeIndex,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    parts = []
    for t in tickers:
        frame = provider.load(t)
        if frame is None:
            continue
        p = monthly_panel_for_ticker(frame, rebal, data_end)
        if not p.empty:
            parts.append(p.assign(ticker=t))
    return pd.concat(parts, ignore_index=True)


def event_market_features(
    events: pd.DataFrame,
    provider: YFinancePriceProvider,
    shares: pd.DataFrame,
    equity: pd.DataFrame,
    data_end: pd.Timestamp,
) -> pd.DataFrame:
    """Per-event market cap, liquidity and momentum, all measured before entry."""
    rows = []
    shares_by = {k: g for k, g in shares.groupby("issuer_cik")}
    equity_by = {k: g for k, g in equity.groupby("issuer_cik")}
    for ticker, g in events.groupby("yf_ticker"):
        frame = provider.load(str(ticker))
        if frame is None:
            continue
        f = frame.loc[frame.index <= data_end]
        sessions = f.index
        ret = f["adj_close"].pct_change()
        filing = pd.to_datetime(g["filing_date"]).to_numpy()
        # Last session on or before the filing date (information available at entry).
        pos = sessions.searchsorted(filing, side="right") - 1
        for idx, p, fd, cik in zip(g.index, pos, filing, g["issuer_cik"], strict=True):
            if p < 1:
                continue
            raw_close = f["raw_close"].iloc[p]
            lo = max(0, p - 59)
            dv = f["dollar_volume"].iloc[lo : p + 1]
            r60 = ret.iloc[lo + 1 : p + 1]
            amihud = float((r60.abs() / dv.iloc[1:].replace(0, np.nan)).mean() * 1e6)
            mom = (
                f["adj_close"].iloc[p - 21] / f["adj_close"].iloc[p - 252] - 1.0
                if p >= 252
                else np.nan
            )
            sh = shares_by.get(cik)
            mcap = np.nan
            if sh is not None:
                known = sh.loc[
                    (sh["known_at"] <= fd) & (sh["known_at"] > fd - np.timedelta64(400, "D"))
                ]
                if len(known):
                    mcap = float(known["value"].iloc[-1]) * raw_close / 1e6
            beq = np.nan
            eq = equity_by.get(cik)
            if eq is not None:
                known = eq.loc[
                    (eq["known_at"] <= fd) & (eq["known_at"] > fd - np.timedelta64(500, "D"))
                ]
                if len(known):
                    beq = float(known["value"].iloc[-1]) / 1e6
            rows.append(
                {
                    "event_idx": idx,
                    "raw_close_at_filing": raw_close,
                    "adv20_at_filing": float(dv.iloc[-20:].mean()),
                    "amihud": amihud,
                    "mom_12_1": mom,
                    "mcap_musd": mcap,
                    "book_equity_musd": beq,
                }
            )
    return pd.DataFrame(rows).set_index("event_idx")
