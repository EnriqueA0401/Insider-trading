"""Event study: buy-and-hold, market-adjusted and factor-adjusted returns per event.

Entry at the adjusted OPEN of the entry session (first session after filing).
Horizon h months = 21*h sessions; exit at the adjusted close of the last session.
- bhr: raw buy-and-hold return (delisting return applied if the series stops early)
- mar: bhr minus IWM buy-and-hold over the same dates
- car: sum of daily abnormal returns vs FF5+UMD, betas from sessions [-252, -22]
  relative to entry (min 120 observations), i.e. estimated before the filing.
Lag variants (entry 2 or 5 sessions later, or at the first close) measure how fast
the signal decays — the filing-lag sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from insider_study.data.prices import YFinancePriceProvider
from insider_study.evaluation.stats import FACTORS

HORIZON_SESSIONS = {1: 21, 3: 63, 6: 126, 12: 252}
LAG_VARIANTS = {
    "open_t1": (0, "open"),
    "close_t1": (0, "close"),
    "open_t2": (1, "open"),
    "open_t5": (4, "open"),
}


def _bhr(
    adj_open: np.ndarray,
    adj_close: np.ndarray,
    e: int,
    n: int,
    delisted: bool,
    dl: float,
    entry_px: str,
) -> float:
    last = min(e + n - 1, len(adj_close) - 1)
    start = adj_open[e] if entry_px == "open" else adj_close[e]
    r = adj_close[last] / start - 1.0
    return (1 + r) * (1 + dl) - 1 if delisted else r


def event_returns(
    events: pd.DataFrame,
    provider: YFinancePriceProvider,
    factors_daily: pd.DataFrame,
    market: pd.DataFrame,
    data_end: pd.Timestamp,
    delisting_return: float,
) -> pd.DataFrame:
    fac = factors_daily[[*FACTORS, "rf"]]
    mkt_open = market["adj_open"]
    mkt_close = market["adj_close"]
    rows: list[dict[str, float | int]] = []
    for ticker, g in events.groupby("yf_ticker"):
        frame = provider.load(str(ticker))
        if frame is None:
            continue
        f = frame.loc[frame.index <= data_end]
        dates = f.index
        ao = f["adj_open"].to_numpy()
        ac = f["adj_close"].to_numpy()
        ret = np.r_[np.nan, ac[1:] / ac[:-1] - 1]
        fx = fac.reindex(dates)
        fmat = fx[list(FACTORS)].to_numpy()
        rf = fx["rf"].to_numpy()
        series_stopped = (data_end - dates[-1]).days > 7
        entry_pos = dates.searchsorted(pd.to_datetime(g["entry_date"]))
        for idx, e in zip(g.index, entry_pos, strict=True):
            if e >= len(dates) or dates[e] != pd.Timestamp(g.at[idx, "entry_date"]):
                continue
            row: dict[str, float | int] = {"event_idx": idx}
            lo, hi = e - 252, e - 21
            beta = None
            if lo >= 1:
                y = ret[lo:hi] - rf[lo:hi]
                X = fmat[lo:hi]
                ok = np.isfinite(y) & np.isfinite(X).all(1)
                if ok.sum() >= 120:
                    Xc = np.c_[np.ones(ok.sum()), X[ok]]
                    beta = np.linalg.lstsq(Xc, y[ok], rcond=None)[0][1:]
            day1 = ac[e] / ao[e] - 1.0
            for h, n in HORIZON_SESSIONS.items():
                complete = e + n - 1 < len(dates)
                delisted = (not complete) and series_stopped
                if not (complete or delisted):
                    continue
                last = min(e + n - 1, len(dates) - 1)
                row[f"bhr_{h}"] = _bhr(ao, ac, e, n, delisted, delisting_return, "open")
                d0, d1 = dates[e], dates[last]
                if d0 in mkt_open.index and d1 in mkt_close.index:
                    m = mkt_close.at[d1] / mkt_open.at[d0] - 1.0
                    row[f"mar_{h}"] = row[f"bhr_{h}"] - m
                if beta is not None:
                    r_win = np.r_[day1, ret[e + 1 : last + 1]]
                    ar = r_win - rf[e : last + 1] - fmat[e : last + 1] @ beta
                    car = float(np.nansum(ar))
                    row[f"car_{h}"] = car + (np.log1p(delisting_return) if delisted else 0.0)
                row[f"delisted_{h}"] = int(delisted)
            # Lag sensitivity at the primary 6-month horizon (market-adjusted).
            n6 = HORIZON_SESSIONS[6]
            for name, (shift, px) in LAG_VARIANTS.items():
                es = e + shift
                if es + n6 - 1 >= len(dates):
                    continue
                d0, d1 = dates[es], dates[es + n6 - 1]
                if d0 in mkt_open.index and d1 in mkt_close.index:
                    b = _bhr(ao, ac, es, n6, False, 0.0, px)
                    m0 = mkt_open.at[d0] if px == "open" else mkt_close.at[d0]
                    row[f"mar6_{name}"] = b - (mkt_close.at[d1] / m0 - 1.0)
            rows.append(row)
    return pd.DataFrame(rows).set_index("event_idx")


def winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
    lo, hi = s.quantile([q, 1 - q])
    return s.clip(lo, hi)


def panel_regression(
    df: pd.DataFrame, y: str, regressors: list[str], controls: list[str]
) -> pd.DataFrame:
    """OLS with month fixed effects and two-way clustered SE (issuer, filing month)."""
    cols = [y, *regressors, *controls, "issuer_cik", "filing_month"]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    d[y] = winsorize(d[y])
    fe = pd.get_dummies(d["filing_month"].astype(str), prefix="m", drop_first=True, dtype=float)
    X = pd.concat([d[regressors + controls].astype(float), fe], axis=1)
    X = sm.add_constant(X)
    groups = np.column_stack([pd.factorize(d["issuer_cik"])[0], pd.factorize(d["filing_month"])[0]])
    fit = sm.OLS(d[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    keep = ["const", *regressors, *controls]
    return pd.DataFrame(
        {
            "coef": fit.params[keep],
            "t": fit.tvalues[keep],
            "p_two_sided": fit.pvalues[keep],
            "n_obs": int(fit.nobs),
        }
    )
