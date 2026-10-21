"""Performance statistics and multiple-testing corrections.

References:
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
- Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
  Overfitting" (CSCV).
- Holm (1979) step-down correction.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

FACTORS = ("mkt_rf", "smb", "hml", "rmw", "cma", "umd")
EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class AlphaResult:
    alpha_monthly: float
    alpha_annual: float
    t_stat: float
    p_one_sided: float
    n_months: int
    betas: Mapping[str, float]
    r2: float
    residual_series: pd.Series  # alpha + residual (the factor-neutral return stream)


def alpha_regression(excess: pd.Series, factors: pd.DataFrame, nw_lags: int) -> AlphaResult:
    """Time-series regression on FF5+UMD with Newey-West standard errors."""
    df = pd.concat([excess.rename("y"), factors[list(FACTORS)]], axis=1, join="inner").dropna()
    X = sm.add_constant(df[list(FACTORS)])
    fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    a = float(fit.params["const"])
    t = float(fit.tvalues["const"])
    neutral = df["y"] - df[list(FACTORS)].mul(fit.params[list(FACTORS)]).sum(axis=1)
    return AlphaResult(
        alpha_monthly=a,
        alpha_annual=(1 + a) ** 12 - 1,
        t_stat=t,
        p_one_sided=float(1 - stats.t.cdf(t, df=max(fit.df_resid, 1))),
        n_months=len(df),
        betas={f: float(fit.params[f]) for f in FACTORS},
        r2=float(fit.rsquared),
        residual_series=neutral,
    )


def sharpe(r: pd.Series, periods: int = 12) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sortino(r: pd.Series, periods: int = 12) -> float:
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(periods)) if downside > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def cagr(r: pd.Series, periods: int = 12) -> float:
    wealth = float((1 + r).prod())
    years = len(r) / periods
    return wealth ** (1 / years) - 1 if years > 0 and wealth > 0 else float("nan")


def calmar(r: pd.Series, periods: int = 12) -> float:
    mdd = max_drawdown(r)
    return cagr(r, periods) / abs(mdd) if mdd < 0 else float("nan")


def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """E[max SR] under the null of zero skill across N independent trials."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(r: pd.Series, n_trials: int, var_sr_trials: float) -> tuple[float, float]:
    """Return (DSR probability, SR0 threshold), Sharpe in per-period units."""
    r = r.dropna()
    t = len(r)
    sr = float(r.mean() / r.std(ddof=1))
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(var_sr_trials, n_trials)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr0) * np.sqrt(t - 1) / denom
    return float(stats.norm.cdf(z)), sr0


def pbo_cscv(returns: pd.DataFrame, n_splits: int = 10) -> tuple[float, pd.Series]:
    """Probability of Backtest Overfitting via combinatorially symmetric CV.

    `returns`: T x N matrix (rows = periods, columns = configurations).
    Returns (PBO, logits).
    """
    m = returns.dropna(how="any").to_numpy()
    t, n = m.shape
    if n < 2:
        raise ValueError("PBO needs at least two configurations")
    blocks = np.array_split(np.arange(t), n_splits)
    logits = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        is_sr = m[is_idx].mean(0) / m[is_idx].std(0, ddof=1)
        oos_sr = m[oos_idx].mean(0) / m[oos_idx].std(0, ddof=1)
        best = int(np.nanargmax(is_sr))
        rank = stats.rankdata(oos_sr)[best] / (n + 1)
        logits.append(np.log(rank / (1 - rank)))
    lg = pd.Series(logits)
    return float((lg <= 0).mean()), lg


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone step-down)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def summary(net: pd.Series, gross: pd.Series, rf: pd.Series) -> dict[str, float]:
    ex = net - rf.reindex(net.index).fillna(0.0)
    return {
        "months": float(len(net)),
        "cagr_net": cagr(net),
        "cagr_gross": cagr(gross),
        "vol_annual": float(net.std(ddof=1) * np.sqrt(12)),
        "sharpe_net": sharpe(ex),
        "sortino_net": sortino(ex),
        "calmar_net": calmar(net),
        "max_drawdown": max_drawdown(net),
        "hit_rate_months": float((ex > 0).mean()),
    }


def newey_west_mean(x: Sequence[float] | pd.Series, lags: int) -> tuple[float, float]:
    """Mean and HAC t-stat of a series (used for spread portfolios)."""
    s = pd.Series(x).dropna()
    fit = sm.OLS(s.to_numpy(), np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.tvalues[0])
