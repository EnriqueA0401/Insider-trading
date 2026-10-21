"""Event features for the sub-hypotheses. All are point-in-time by construction.

- CMP routine/opportunistic (H3): insider classified once per calendar year Y using
  only open-market trades FILED before Jan 1 of Y.
- Conviction terciles (H4): breakpoints from events filed before the event's month.
- Clustering (H5): distinct insiders buying the same issuer within a trailing window
  of filing dates ending at the event's filing date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from insider_study.domain import Col, TradingPattern


def cmp_classification(
    history: pd.DataFrame, years: range, lookback_years: int = 3, buys_only: bool = False
) -> pd.DataFrame:
    """Return (insider_cik, year, pattern) for insiders classifiable in each year."""
    h = history if not buys_only else history.loc[history[Col.TRANS_CODE].eq("P")]
    # CMP needs the trade month; only trades filed before Jan 1 of Y are used.
    trade_col = Col.TRANSACTION_DATE  # leakage-ok: CMP trade month (filed-before filter below)
    trade = pd.to_datetime(h[trade_col])
    filed = pd.to_datetime(h[Col.FILING_DATE])
    base = pd.DataFrame(
        {
            "insider_cik": h[Col.INSIDER_CIK].to_numpy(),
            "ty": trade.dt.year,
            "tm": trade.dt.month,
            "filed": filed,
        }
    ).dropna()
    out = []
    for year in years:
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        window = range(year - lookback_years, year)
        b = base.loc[(base["filed"] < cutoff) & base["ty"].isin(window)]
        if b.empty:
            continue
        years_per = b.groupby("insider_cik")["ty"].nunique()
        eligible = years_per.index[years_per.eq(lookback_years)]
        e = b.loc[b["insider_cik"].isin(eligible)]
        month_years = e.groupby(["insider_cik", "tm"])["ty"].nunique()
        routine = set(month_years.index[month_years.eq(lookback_years)].get_level_values(0))
        out.append(
            pd.DataFrame(
                {
                    "insider_cik": eligible,
                    "year": year,
                    "pattern": [
                        TradingPattern.ROUTINE.value
                        if c in routine
                        else TradingPattern.OPPORTUNISTIC.value
                        for c in eligible
                    ],
                }
            )
        )
    return (
        pd.concat(out, ignore_index=True)
        if out
        else pd.DataFrame(columns=["insider_cik", "year", "pattern"])
    )


def attach_cmp(events: pd.DataFrame, cls: pd.DataFrame, col: str = "pattern") -> pd.Series:
    year = pd.to_datetime(events[Col.FILING_DATE]).dt.year
    key = pd.DataFrame({"insider_cik": events[Col.INSIDER_CIK].to_numpy(), "year": year.to_numpy()})
    merged = key.merge(cls, on=["insider_cik", "year"], how="left")
    return pd.Series(
        merged["pattern"].fillna(TradingPattern.UNCLASSIFIED.value).to_numpy(),
        index=events.index,
        name=col,
    )


def expanding_terciles(events: pd.DataFrame, value_col: str, min_history: int = 500) -> pd.Series:
    """Tercile (1, 2, 3) using breakpoints from events filed before the event's month."""
    d = pd.to_datetime(events[Col.FILING_DATE])
    month = d.dt.to_period("M")
    vals = events[value_col]
    out = pd.Series(np.nan, index=events.index)
    order = np.sort(month.unique())
    for m in order:
        prior = vals.loc[(month < m) & vals.notna()]
        if len(prior) < min_history:
            continue
        lo, hi = prior.quantile([1 / 3, 2 / 3])
        idx = month.eq(m) & vals.notna()
        v = vals.loc[idx]
        out.loc[idx] = np.where(v <= lo, 1, np.where(v <= hi, 2, 3))
    return out


def cluster_size(events: pd.DataFrame, window_days: int) -> pd.Series:
    """Distinct insiders with eligible buys at the issuer in [filing - window, filing]."""
    df = pd.DataFrame(
        {
            "issuer": events[Col.ISSUER_CIK].to_numpy(),
            "insider": events[Col.INSIDER_CIK].to_numpy(),
            "d": pd.to_datetime(events[Col.FILING_DATE]).to_numpy(),
        },
        index=events.index,
    ).sort_values(["issuer", "d"])
    result = pd.Series(1, index=events.index, dtype=int)
    win = np.timedelta64(window_days, "D")
    for _, g in df.groupby("issuer", sort=False):
        if len(g) == 1:
            continue
        dates = g["d"].to_numpy()
        insiders = g["insider"].to_numpy()
        left = np.searchsorted(dates, dates - win, side="left")
        right = np.searchsorted(dates, dates, side="right")  # include same-day filings
        counts = [len(set(insiders[lo:hi])) for lo, hi in zip(left, right, strict=True)]
        result.loc[g.index] = counts
    return result


def size_bucket(
    mcap_musd: pd.Series, filing_dates: pd.Series, breakpoints: pd.DataFrame
) -> pd.Series:
    """NYSE breakpoints of the month-end BEFORE the filing month (known at filing)."""
    prev_month_end = (pd.to_datetime(filing_dates).dt.to_period("M") - 1).dt.to_timestamp("M")
    bp = breakpoints.reindex(prev_month_end.to_numpy())
    p20, p50, p80 = bp["p20"].to_numpy(), bp["p50"].to_numpy(), bp["p80"].to_numpy()
    m = mcap_musd.to_numpy()
    out = np.select(
        [m < p20, m < p50, m < p80, m >= p80], ["MICRO", "SMALL", "MID", "LARGE"], default=""
    )
    out = np.where(np.isnan(m) | np.isnan(p20), "", out)
    return pd.Series(out, index=mcap_musd.index)
