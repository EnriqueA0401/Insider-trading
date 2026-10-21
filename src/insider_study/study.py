"""End-to-end study runner.

Two modes:
- "dev": everything strictly before the holdout (prices clipped at holdout_start - 1
  day, events filed before holdout_start). Used for train/validation, robustness,
  trial registration, DSR and PBO.
- "final": requires the one-time holdout unseal for the frozen config hash. Runs the
  same pre-registered configurations on the holdout window once.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

from insider_study.backtest.engine import MonthlyResult, PortfolioSpec, run_monthly
from insider_study.config import StudyConfig
from insider_study.data.factors import FrenchFactorProvider
from insider_study.data.market import build_monthly_panel, event_market_features, rebalance_sessions
from insider_study.data.prices import YFinancePriceProvider
from insider_study.data.tickers import candidate_tickers, current_sec_tickers, validate_events
from insider_study.domain import Split
from insider_study.evaluation import stats as st
from insider_study.evaluation.event_study import event_returns, panel_regression
from insider_study.holdout import HoldoutAccessError, HoldoutGuard
from insider_study.logging_setup import get_logger
from insider_study.signals.features import (
    attach_cmp,
    cluster_size,
    cmp_classification,
    expanding_terciles,
    size_bucket,
)
from insider_study.trials import TrialRecord, TrialRegistry

log = get_logger(__name__)
Mode = Literal["dev", "final"]
PRICE_HISTORY_END = pd.Timestamp("2026-10-16")

# Evaluation windows (calendar months of portfolio returns), Amendment 2(h).
WINDOWS = {
    Split.TRAIN: ("2006-07-31", "2014-12-31"),
    Split.VALIDATION: ("2015-01-31", "2019-12-31"),
    Split.HOLDOUT: ("2020-01-31", "2025-12-31"),
}


@dataclass
class Context:
    cfg: StudyConfig
    mode: Mode
    data_end: pd.Timestamp
    out_dir: Path
    provider: YFinancePriceProvider
    events: pd.DataFrame
    panel: pd.DataFrame
    rebal: pd.DatetimeIndex
    factors_m: pd.DataFrame
    factors_d: pd.DataFrame
    results: dict[str, Any] = field(default_factory=dict)
    series: dict[str, MonthlyResult] = field(default_factory=dict)


# --------------------------------------------------------------------------- prepare


def _size_proxy(ev: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    """Fill missing market caps with an OLS proxy calibrated on 2009-07..2010-12."""
    d = pd.to_datetime(ev["filing_date"])
    x = pd.DataFrame(
        {
            "ladv": np.log(ev["adv20_at_filing"].clip(lower=1.0)),
            "lpx": np.log(ev["raw_close_at_filing"].clip(lower=0.01)),
        }
    )
    y = np.log(ev["mcap_musd"])
    calib = (d >= "2009-07-01") & (d <= "2010-12-31") & y.notna() & np.isfinite(x).all(1)
    fit = sm.OLS(y[calib], sm.add_constant(x[calib])).fit()
    pred = np.exp(fit.predict(sm.add_constant(x, has_constant="add")))
    filled = ev["mcap_musd"].where(ev["mcap_musd"].notna(), pred)
    return filled, pred, {"r2": float(fit.rsquared), "n_calib": int(calib.sum())}


def prepare(cfg: StudyConfig, mode: Mode, force: bool = False) -> Context:
    guard = HoldoutGuard.from_config(cfg)
    if mode == "final" and not guard.is_unsealed:
        raise HoldoutAccessError("final mode requires the committed HOLDOUT_UNSEALED record")
    data_end = (
        pd.Timestamp(cfg.splits.holdout_start) - pd.Timedelta(days=1)
        if mode == "dev"
        else PRICE_HISTORY_END
    )
    out_dir = Path("results") / mode
    work = cfg.data.processed_dir / mode
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = YFinancePriceProvider(cfg.data.raw_dir / "yf")
    french = FrenchFactorProvider(cfg.data.raw_dir / "french")
    factors_m = french.get_factors(cfg.splits.train_start, data_end.date(), "monthly")
    factors_d = french.get_factors(
        cfg.splits.train_start.replace(year=2004), data_end.date(), "daily"
    )
    rebal = rebalance_sessions(cfg.splits.train_start, data_end.date())

    ev_path, panel_path = work / "events_enriched.parquet", work / "panel.parquet"
    if ev_path.exists() and panel_path.exists() and not force:
        events = pd.read_parquet(ev_path)
        panel = pd.read_parquet(panel_path)
        prep = json.loads((work / "prepare.json").read_text())
    else:
        events, panel, prep = _build(cfg, provider, data_end, french, rebal, work)
        events.to_parquet(ev_path)
        panel.to_parquet(panel_path, index=False)
        (work / "prepare.json").write_text(json.dumps(prep, indent=2, default=str))
    ctx = Context(
        cfg, mode, data_end, out_dir, provider, events, panel, rebal, factors_m, factors_d
    )
    ctx.results["prepare"] = prep
    return ctx


def _build(
    cfg: StudyConfig,
    provider: YFinancePriceProvider,
    data_end: pd.Timestamp,
    french: FrenchFactorProvider,
    rebal: pd.DatetimeIndex,
    work: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    proc = cfg.data.processed_dir
    events = pd.read_parquet(proc / "events.parquet")
    events = events.loc[pd.to_datetime(events["filing_date"]) <= data_end].reset_index(drop=True)
    plan = pd.read_parquet(proc / "events_10b5_1.parquet")
    plan = plan.loc[pd.to_datetime(plan["filing_date"]) <= data_end].reset_index(drop=True)
    events = pd.concat([events.assign(is_plan=False), plan.assign(is_plan=True)], ignore_index=True)

    current = current_sec_tickers(
        cfg.data.raw_dir / "sec_company_tickers.json", cfg.data.sec_user_agent
    )
    cands = candidate_tickers(events, current)
    val = validate_events(events, cands, provider, cfg.filters.max_price_deviation)
    events = events.join(val)

    hist = pd.read_parquet(proc / "open_market_history.parquet")
    hist = hist.loc[pd.to_datetime(hist["filing_date"]) <= data_end]
    years = range(cfg.splits.train_start.year, data_end.year + 1)
    lb = cfg.signal.routine_lookback_years
    events["pattern"] = attach_cmp(events, cmp_classification(hist, years, lb))
    events["pattern_2y"] = attach_cmp(events, cmp_classification(hist, years, 2))
    events["pattern_buys"] = attach_cmp(events, cmp_classification(hist, years, lb, buys_only=True))

    base = ~events["is_plan"]
    events["conv_t"] = np.nan
    events.loc[base, "conv_t"] = expanding_terciles(events.loc[base], "delta_h")
    events["value_t"] = np.nan
    events.loc[base, "value_t"] = expanding_terciles(events.loc[base], "value_usd")
    events["cluster30"] = 0
    events.loc[base, "cluster30"] = cluster_size(events.loc[base], cfg.signal.cluster_window_days)
    events["cluster60"] = 0
    events.loc[base, "cluster60"] = cluster_size(events.loc[base], 60)

    priced = events["price_status"].eq("priced")
    shares = pd.read_parquet(cfg.data.raw_dir / "frames_shares.parquet")
    equity = pd.read_parquet(cfg.data.raw_dir / "frames_equity.parquet")
    for frame in (shares, equity):
        frame["known_at"] = pd.to_datetime(frame["known_at"])
    mf = event_market_features(events.loc[priced], provider, shares, equity, data_end)
    events = events.join(mf)
    events["mcap_xbrl"] = events["mcap_musd"]
    events["mcap_musd"], proxy_pred, proxy_info = _size_proxy(events)
    bp = french.get_size_breakpoints(cfg.splits.train_start.replace(year=2005), data_end.date())
    events["size_bucket"] = size_bucket(events["mcap_musd"], events["filing_date"], bp)
    events["size_bucket_xbrl"] = size_bucket(events["mcap_xbrl"], events["filing_date"], bp)
    d = pd.to_datetime(events["filing_date"])
    test = (d >= "2011-01-01") & (d <= "2014-12-31") & events["mcap_xbrl"].notna()
    proxy_bucket = size_bucket(proxy_pred.loc[test], events.loc[test, "filing_date"], bp)
    proxy_info["bucket_accuracy_2011_2014"] = float(
        (proxy_bucket == events.loc[test, "size_bucket_xbrl"]).mean()
    )
    proxy_info["share_events_proxy_size"] = float(
        (events["mcap_xbrl"].isna() & events["mcap_musd"].notna()).sum() / max(int(priced.sum()), 1)
    )

    tickers = sorted(set(events.loc[priced, "yf_ticker"]) | {"IWM"})
    panel = build_monthly_panel(provider, tickers, rebal, data_end)

    cov = (
        events.loc[base]
        .assign(year=d.dt.year)
        .groupby("year")["price_status"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    prep: dict[str, Any] = {
        "n_events": int(base.sum()),
        "n_plan_events": int((~base).sum()),
        "price_status": events.loc[base, "price_status"].value_counts().to_dict(),
        "coverage_by_year": cov.round(4).to_dict(orient="index"),
        "size_proxy": proxy_info,
        "n_tickers_panel": len(tickers),
    }
    return events, panel, prep


# --------------------------------------------------------------------------- portfolios

Selector = Callable[[pd.DataFrame], pd.Series]


def _bucket_selector(cfo_only: bool, bucket: str) -> Selector:
    def select(e: pd.DataFrame) -> pd.Series:
        base = e["is_cfo"] if cfo_only else pd.Series(True, index=e.index)
        return base & e["size_bucket"].eq(bucket)

    return select


SELECTORS: dict[str, Selector] = {
    "ALL": lambda e: pd.Series(True, index=e.index),
    "CFO": lambda e: e["is_cfo"],
    "NONCFO": lambda e: ~e["is_cfo"],
    "CEO": lambda e: e["is_ceo"],
    "DIR": lambda e: e["is_director_only"],
    "OPP": lambda e: e["pattern"].eq("OPPORTUNISTIC"),
    "ROUT": lambda e: e["pattern"].eq("ROUTINE"),
    "CFO_OPP": lambda e: e["is_cfo"] & e["pattern"].eq("OPPORTUNISTIC"),
    "CFO_ROUT": lambda e: e["is_cfo"] & e["pattern"].eq("ROUTINE"),
    "CONV_HI": lambda e: e["conv_t"].eq(3),
    "CONV_LO": lambda e: e["conv_t"].eq(1),
    "CFO_CONV_HI": lambda e: e["is_cfo"] & e["conv_t"].eq(3),
    "CFO_CONV_LO": lambda e: e["is_cfo"] & e["conv_t"].eq(1),
    "CLUSTER": lambda e: e["cluster30"].ge(3),
    "ISOLATED": lambda e: e["cluster30"].eq(1),
    "CFO_MIDLARGE": lambda e: e["is_cfo"] & e["size_bucket"].isin(["MID", "LARGE"]),
    "ALL_MIDLARGE": lambda e: e["size_bucket"].isin(["MID", "LARGE"]),
    **{
        f"{who}_{b}": _bucket_selector(who == "CFO", b)
        for who in ("CFO", "ALL")
        for b in ("MICRO", "SMALL", "MID", "LARGE")
    },
    # Robustness definitions
    "CFO_BROAD": lambda e: e["is_cfo_broad"],
    "CFO_NODUAL": lambda e: e["is_cfo"] & ~e["is_dual_ceo_cfo"],
    "OPP_2Y": lambda e: e["pattern_2y"].eq("OPPORTUNISTIC"),
    "ROUT_2Y": lambda e: e["pattern_2y"].eq("ROUTINE"),
    "OPP_BUYS": lambda e: e["pattern_buys"].eq("OPPORTUNISTIC"),
    "ROUT_BUYS": lambda e: e["pattern_buys"].eq("ROUTINE"),
    "VAL_HI": lambda e: e["value_t"].eq(3),
    "VAL_LO": lambda e: e["value_t"].eq(1),
    "CLUSTER2": lambda e: e["cluster30"].ge(2),
    "CLUSTER60": lambda e: e["cluster60"].ge(3),
    "ISOLATED60": lambda e: e["cluster60"].eq(1),
}
PLAN_SELECTOR = "PLAN_10B5_1"


def _spec(ctx: Context, name: str, **kw: Any) -> PortfolioSpec:
    c = ctx.cfg
    base: dict[str, Any] = {
        "name": name,
        "horizon_months": c.signal.primary_horizon_months,
        "aum_usd": c.portfolio.aum_usd,
        "max_position_weight": c.portfolio.max_position_weight,
        "delisting_return": c.data.delisting_return_primary,
    }
    return PortfolioSpec(**(base | kw))


def run_portfolio(ctx: Context, key: str, selector: str, **kw: Any) -> MonthlyResult:
    ev = ctx.events
    if selector == PLAN_SELECTOR:
        mask = ev["is_plan"]
    else:
        mask = ~ev["is_plan"] & SELECTORS[selector](ev).fillna(False).astype(bool)
    sel = ev.loc[mask & ev["price_status"].eq("priced")]
    res = run_monthly(
        sel, ctx.panel, ctx.rebal, ctx.factors_m["rf"], _spec(ctx, key, **kw), ctx.cfg.costs
    )
    ctx.series[key] = res
    return res


def window(s: pd.Series, split: Split) -> pd.Series:
    lo, hi = WINDOWS[split]
    return s.loc[(s.index >= lo) & (s.index <= hi)]


def evaluate(
    ctx: Context, key: str, split: Split, horizon: int, start: str | None = None
) -> dict[str, Any]:
    res = ctx.series[key]
    rf = ctx.factors_m["rf"]
    net, gross = window(res.net, split), window(res.gross, split)
    if start is not None:
        net, gross = net.loc[start:], gross.loc[start:]
    if len(net) < 12:
        return {"months": len(net)}
    lags = horizon + 1
    a_net = st.alpha_regression(net - rf.reindex(net.index), ctx.factors_m, lags)
    a_gross = st.alpha_regression(gross - rf.reindex(gross.index), ctx.factors_m, lags)
    out: dict[str, Any] = st.summary(net, gross, rf)
    out |= {
        "alpha_net_annual": a_net.alpha_annual,
        "alpha_net_t": a_net.t_stat,
        "alpha_net_p": a_net.p_one_sided,
        "alpha_gross_annual": a_gross.alpha_annual,
        "alpha_gross_t": a_gross.t_stat,
        "betas": a_net.betas,
        "r2": a_net.r2,
        "avg_holdings": float(window(res.n_holdings, split).mean()),
        "avg_cash": float(window(res.cash_weight, split).mean()),
        "turnover_monthly": float(window(res.turnover, split).mean()),
        "cost_annual": float(window(res.cost, split).mean() * 12),
        "neutral_sharpe_monthly": float(a_net.residual_series.mean() / a_net.residual_series.std()),
    }
    return out


def spread_test(
    ctx: Context, a: str, b: str, split: Split, horizon: int, start: str | None = None
) -> dict[str, Any]:
    diff = window(ctx.series[a].net - ctx.series[b].net, split)
    if start is not None:
        diff = diff.loc[start:]
    res = st.alpha_regression(diff, ctx.factors_m, horizon + 1)
    mean, t_mean = st.newey_west_mean(diff, horizon + 1)
    return {
        "spread": f"{a} - {b}",
        "months": res.n_months,
        "mean_monthly": mean,
        "mean_t": t_mean,
        "alpha_annual": res.alpha_annual,
        "alpha_t": res.t_stat,
        "alpha_p": res.p_one_sided,
    }


def register(ctx: Context, registry: TrialRegistry, key: str, hypothesis: str, desc: str) -> None:
    if ctx.mode != "dev":
        return
    rf = ctx.factors_m["rf"]
    net = window(ctx.series[key].net, Split.VALIDATION)
    ex = net - rf.reindex(net.index)
    registry.append(
        TrialRecord(
            config_hash=ctx.cfg.config_hash(),
            hypothesis=hypothesis,
            split=Split.VALIDATION,
            sharpe=float(ex.mean() / ex.std()),
            n_periods=len(ex),
            description=f"{key}: {desc}",
        )
    )


# --------------------------------------------------------------------------- analyses

MAIN_PORTFOLIOS = [
    "ALL",
    "CFO",
    "NONCFO",
    "CEO",
    "DIR",
    "OPP",
    "ROUT",
    "CFO_OPP",
    "CFO_ROUT",
    "CONV_HI",
    "CONV_LO",
    "CFO_CONV_HI",
    "CFO_CONV_LO",
    "CLUSTER",
    "ISOLATED",
    "CFO_MIDLARGE",
    "ALL_MIDLARGE",
    "CFO_MICRO",
    "CFO_SMALL",
    "CFO_MID",
    "CFO_LARGE",
    "ALL_MICRO",
    "ALL_SMALL",
    "ALL_MID",
    "ALL_LARGE",
]
CMP_START = "2009-02-28"
CONFIRMATORY = {
    "H1": ("CFO", None),
    "H2a": ("CFO", "NONCFO"),
    "H2b_CEO": ("CFO", "CEO"),
    "H2b_DIR": ("CFO", "DIR"),
    "H3": ("OPP", "ROUT"),
    "H4": ("CONV_HI", "CONV_LO"),
    "H5": ("CLUSTER", "ISOLATED"),
    "H6": ("CFO_MIDLARGE", None),
}
SECONDARY = {
    "CFO_vs_ALL": ("CFO", "ALL"),
    "H3_CFO": ("CFO_OPP", "CFO_ROUT"),
    "H4_CFO": ("CFO_CONV_HI", "CFO_CONV_LO"),
}


def _start_for(*keys: str) -> str | None:
    return CMP_START if any("OPP" in k or "ROUT" in k for k in keys) else None


def run_main(ctx: Context, registry: TrialRegistry | None) -> None:
    h = ctx.cfg.signal.primary_horizon_months
    for key in MAIN_PORTFOLIOS:
        run_portfolio(ctx, key, key)
        if registry is not None:
            register(ctx, registry, key, "main", f"selector={key}, H={h}m, net")
    splits = [Split.TRAIN, Split.VALIDATION] if ctx.mode == "dev" else [Split.HOLDOUT]
    ctx.results["portfolios"] = {
        key: {s.value: evaluate(ctx, key, s, h, _start_for(key)) for s in splits}
        for key in MAIN_PORTFOLIOS
    }
    tests: dict[str, dict[str, Any]] = {}
    for split in splits:
        out: dict[str, Any] = {}
        for name, (a, b) in {**CONFIRMATORY, **SECONDARY}.items():
            start = _start_for(a, b or "")
            if b is None:
                r = evaluate(ctx, a, split, h, start)
                out[name] = {
                    "portfolio": a,
                    "alpha_annual": r["alpha_net_annual"],
                    "alpha_t": r["alpha_net_t"],
                    "alpha_p": r["alpha_net_p"],
                    "months": r["months"],
                }
            else:
                out[name] = spread_test(ctx, a, b, split, h, start)
        adj = st.holm({k: out[k]["alpha_p"] for k in CONFIRMATORY})
        for k, p in adj.items():
            out[k]["holm_p"] = p
        tests[split.value] = out
    ctx.results["tests"] = tests


def run_robustness(ctx: Context, registry: TrialRegistry | None) -> None:
    c = ctx.cfg
    h = c.signal.primary_horizon_months
    variants: list[tuple[str, str, dict[str, Any], str]] = []
    for sel in ("CFO", "ALL"):
        for hz in c.signal.horizons_months:
            if hz != h:
                variants.append((f"{sel}_H{hz}", sel, {"horizon_months": hz}, f"horizon {hz}m"))
        for m in c.costs.cost_multipliers:
            variants.append((f"{sel}_COSTx{m}", sel, {"cost_multiplier": m}, f"costs x{m}"))
        for dl in c.data.delisting_return_sensitivities:
            variants.append((f"{sel}_DL{dl}", sel, {"delisting_return": dl}, f"delisting {dl}"))
        variants.append((f"{sel}_SKIP1", sel, {"skip_months": 1}, "signal used one month later"))
    for sel in (
        "CFO_BROAD",
        "CFO_NODUAL",
        "OPP_2Y",
        "ROUT_2Y",
        "OPP_BUYS",
        "ROUT_BUYS",
        "VAL_HI",
        "VAL_LO",
        "CLUSTER2",
        "CLUSTER60",
        "ISOLATED60",
    ):
        variants.append((sel, sel, {}, "alternative definition"))
    variants.append((PLAN_SELECTOR, PLAN_SELECTOR, {}, "10b5-1 purchases (diagnostic)"))

    out: dict[str, Any] = {}
    splits = [Split.TRAIN, Split.VALIDATION] if ctx.mode == "dev" else [Split.HOLDOUT]
    for key, sel, kw, desc in variants:
        run_portfolio(ctx, key, sel, **kw)
        if registry is not None:
            register(ctx, registry, key, "robustness", desc)
        hz = kw.get("horizon_months", h)
        out[key] = {s.value: evaluate(ctx, key, s, hz, _start_for(key)) for s in splits}
    spreads = {
        "H3_2Y": ("OPP_2Y", "ROUT_2Y"),
        "H3_BUYS": ("OPP_BUYS", "ROUT_BUYS"),
        "H4_VALUE": ("VAL_HI", "VAL_LO"),
        "H5_GE2": ("CLUSTER2", "ISOLATED"),
        "H5_60D": ("CLUSTER60", "ISOLATED60"),
    }
    out["spreads"] = {
        s.value: {
            n: spread_test(ctx, a, b, s, h, _start_for(a, b)) for n, (a, b) in spreads.items()
        }
        for s in splits
    }
    # Sub-periods (no new trial: slices of the primary series).
    periods = {
        "2006-2009 (incl. GFC)": ("2006-07-31", "2009-12-31"),
        "GFC 2008-2009": ("2008-01-31", "2009-12-31"),
        "2010-2014": ("2010-01-31", "2014-12-31"),
        "2015-2019": ("2015-01-31", "2019-12-31"),
    }
    if ctx.mode == "final":
        periods = {
            "COVID 2020": ("2020-01-31", "2020-12-31"),
            "2021-2025": ("2021-01-31", "2025-12-31"),
        }
    sub: dict[str, Any] = {}
    rf = ctx.factors_m["rf"]
    for key in ("CFO", "ALL", "CFO_MIDLARGE"):
        net = ctx.series[key].net
        sub[key] = {}
        for label, (lo, hi) in periods.items():
            seg = net.loc[lo:hi]
            if len(seg) < 12:
                continue
            a = st.alpha_regression(seg - rf.reindex(seg.index), ctx.factors_m, h + 1)
            sub[key][label] = {
                "alpha_net_annual": a.alpha_annual,
                "alpha_t": a.t_stat,
                "months": a.n_months,
                "cagr_net": st.cagr(seg),
            }
    out["subperiods"] = sub
    ctx.results["robustness"] = out


def run_capacity(ctx: Context, registry: TrialRegistry | None) -> None:
    h = ctx.cfg.signal.primary_horizon_months
    split = Split.VALIDATION if ctx.mode == "dev" else Split.HOLDOUT
    out: dict[str, Any] = {}
    for sel in ("CFO", "ALL", "CFO_MIDLARGE", "ALL_MIDLARGE"):
        curve = {}
        for aum in ctx.cfg.portfolio.capacity_grid_usd:
            key = f"{sel}_AUM{int(aum)}"
            run_portfolio(ctx, key, sel, aum_usd=aum)
            if registry is not None:
                register(ctx, registry, key, "capacity", f"AUM {aum:,.0f}")
            r_tr = evaluate(ctx, key, Split.TRAIN, h) if ctx.mode == "dev" else {}
            r = evaluate(ctx, key, split, h)
            curve[str(int(aum))] = {
                "alpha_net_annual": r["alpha_net_annual"],
                "alpha_net_t": r["alpha_net_t"],
                "avg_cash": r["avg_cash"],
                "cost_annual": r["cost_annual"],
                "train_alpha_net_annual": r_tr.get("alpha_net_annual"),
            }
        positive = [int(a) for a, v in curve.items() if v["alpha_net_annual"] > 0]
        out[sel] = {"curve": curve, "max_aum_positive_alpha": max(positive) if positive else 0}
    ctx.results["capacity"] = out


def run_multiple_testing(ctx: Context, registry: TrialRegistry) -> None:
    """DSR for the H1 candidate and PBO over every registered configuration."""
    h = ctx.cfg.signal.primary_horizon_months
    rf = ctx.factors_m["rf"]
    n_trials = registry.count()
    sharpes = np.array(registry.sharpes(Split.VALIDATION))
    var_sr = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    out: dict[str, Any] = {"n_trials": n_trials, "var_sr_monthly": var_sr}
    for key in ("CFO", "ALL", "CFO_MIDLARGE"):
        net = window(ctx.series[key].net, Split.VALIDATION)
        ex = net - rf.reindex(net.index)
        neutral = st.alpha_regression(ex, ctx.factors_m, h + 1).residual_series
        dsr_n, sr0 = st.deflated_sharpe(neutral, n_trials, var_sr)
        dsr_raw, _ = st.deflated_sharpe(ex, n_trials, var_sr)
        out[key] = {
            "dsr_factor_neutral": dsr_n,
            "dsr_raw_excess": dsr_raw,
            "sr0_monthly": sr0,
            "sr_neutral_monthly": float(neutral.mean() / neutral.std()),
        }
    keys = [k for k in ctx.series if "_AUM" not in k]
    mat = pd.DataFrame(
        {
            k: (ctx.series[k].net - rf.reindex(ctx.series[k].net.index)).loc[
                "2006-07-31":"2019-12-31"
            ]
            for k in keys
        }
    ).loc["2009-02-28":]  # common window where every configuration (incl. CMP) exists
    pbo, logits = st.pbo_cscv(mat, ctx.cfg.validation.cpcv_groups)
    out["pbo"] = pbo
    out["pbo_n_configs"] = len(keys)
    out["pbo_months"] = len(mat)
    out["pbo_logit_median"] = float(logits.median())
    mat.to_parquet(ctx.out_dir / "trial_returns.parquet")
    ctx.results["multiple_testing"] = out


def run_unpriced_bounding(ctx: Context) -> None:
    """Survivorship bounding: unpriced holdings earn IWM + delta (Amendment 2e)."""
    h = ctx.cfg.signal.primary_horizon_months
    ev = ctx.events.loc[~ctx.events["is_plan"]]
    rf = ctx.factors_m["rf"]
    iwm = ctx.panel.loc[ctx.panel["ticker"].eq("IWM")].set_index("rebal")["ret_nodl"]
    iwm.index = pd.DatetimeIndex(iwm.index).to_period("M").to_timestamp("M")
    out: dict[str, Any] = {}
    split = Split.VALIDATION if ctx.mode == "dev" else Split.HOLDOUT
    for key in ("CFO", "ALL"):
        sel = ev.loc[SELECTORS[key](ev).fillna(False).astype(bool)]
        m = pd.to_datetime(sel["filing_date"]).dt.to_period("M")
        held: dict[pd.Timestamp, tuple[set[str], set[str]]] = {}
        for k in range(1, h + 1):
            month = (m + k).dt.to_timestamp("M")
            for mo, cik, status in zip(month, sel["issuer_cik"], sel["price_status"], strict=True):
                priced_set, unpriced_set = held.setdefault(mo, (set(), set()))
                (priced_set if status == "priced" else unpriced_set).add(cik)
        share = pd.Series(
            {mo: len(u - p) / max(len(u | p), 1) for mo, (p, u) in held.items()}
        ).sort_index()
        net = window(ctx.series[key].net, split)
        u = share.reindex(net.index).fillna(0.0)
        res: dict[str, Any] = {"avg_unpriced_share": float(u.mean())}
        for delta in (0.0, -0.01, -0.03):
            adj = (1 - u) * net + u * (iwm.reindex(net.index).fillna(0.0) + delta)
            a = st.alpha_regression(adj - rf.reindex(adj.index), ctx.factors_m, h + 1)
            res[f"delta_{delta}"] = {"alpha_net_annual": a.alpha_annual, "alpha_t": a.t_stat}
        # Breakeven delta: alpha is linear in delta -> solve from two points.
        a0 = res["delta_0.0"]["alpha_net_annual"]
        a1 = res["delta_-0.03"]["alpha_net_annual"]
        res["breakeven_delta_monthly"] = float(-0.03 * a0 / (a0 - a1)) if a0 != a1 else float("nan")
        out[key] = res
    ctx.results["unpriced_bounding"] = out


# --------------------------------------------------------------------------- event study

EVENT_GROUPS = (
    "ALL",
    "CFO",
    "NONCFO",
    "CEO",
    "DIR",
    "OPP",
    "ROUT",
    "CONV_HI",
    "CONV_LO",
    "CLUSTER",
    "ISOLATED",
    "CFO_MIDLARGE",
    "ALL_MIDLARGE",
)


def _monthly_mean_t(df: pd.DataFrame, col: str, lags: int) -> dict[str, float]:
    """Mean of monthly cross-sectional averages with NW t (robust to clustering)."""
    s = df.groupby("filing_month")[col].mean().dropna()
    if len(s) < 12:
        return {"mean": float("nan"), "t": float("nan"), "n_events": int(df[col].notna().sum())}
    mean, t = st.newey_west_mean(s, lags)
    return {"mean": mean, "t": t, "n_events": int(df[col].notna().sum()), "n_months": len(s)}


def run_event_study(ctx: Context) -> pd.DataFrame:
    path = ctx.cfg.data.processed_dir / ctx.mode / "event_returns.parquet"
    ev = ctx.events.loc[~ctx.events["is_plan"] & ctx.events["price_status"].eq("priced")]
    if path.exists():
        er = pd.read_parquet(path)
    else:
        iwm = ctx.provider.load("IWM")
        assert iwm is not None
        er = event_returns(
            ev,
            ctx.provider,
            ctx.factors_d,
            iwm,
            ctx.data_end,
            ctx.cfg.data.delisting_return_primary,
        )
        er.to_parquet(path)
    df = ev.join(er, how="inner")
    df["filing_month"] = pd.to_datetime(df["filing_date"]).dt.to_period("M").astype(str)
    df["split"] = df["split"].astype(str)
    splits = ["train", "validation"] if ctx.mode == "dev" else ["holdout"]
    out: dict[str, Any] = {}
    for sp in splits:
        d = df.loc[df["split"].eq(sp)]
        grp: dict[str, Any] = {}
        for g in EVENT_GROUPS:
            m = SELECTORS[g](d).fillna(False).astype(bool)
            gd = d.loc[m]
            grp[g] = {
                f"{kind}_{hz}": _monthly_mean_t(gd, f"{kind}_{hz}", hz + 1)
                for kind in ("bhr", "mar", "car")
                for hz in ctx.cfg.signal.horizons_months
                if f"{kind}_{hz}" in gd
            }
            grp[g] |= {
                f"lag_{n}": _monthly_mean_t(gd, f"mar6_{n}", 7)
                for n in ("open_t1", "close_t1", "open_t2", "open_t5")
                if f"mar6_{n}" in gd
            }
        out[sp] = grp

    # Panel regression: marginal CFO effect over a generic insider (H2a, ii).
    df["log_mcap"] = np.log(df["mcap_musd"].clip(lower=0.1))
    df["log_amihud"] = np.log(df["amihud"].clip(lower=1e-6))
    df["log_value"] = np.log(df["value_usd"].clip(lower=1.0))
    df["log_bm"] = np.log((df["book_equity_musd"] / df["mcap_xbrl"]).where(lambda x: x > 0))
    for flag in ("is_cfo", "is_ceo", "is_director_only"):
        df[flag] = df[flag].astype(float)
    regs = ["is_cfo", "is_ceo", "is_director_only"]
    ctrls = ["log_mcap", "mom_12_1", "log_amihud", "log_value", "filing_lag_days"]
    panels: dict[str, Any] = {}
    for sp in splits:
        d = df.loc[df["split"].eq(sp)]
        panels[sp] = panel_regression(d, "car_6", regs, ctrls).to_dict(orient="index")
        d09 = d.loc[pd.to_datetime(d["filing_date"]) >= "2009-07-01"]
        if len(d09) > 1000:
            panels[f"{sp}_with_bm"] = panel_regression(
                d09, "car_6", regs, [*ctrls, "log_bm"]
            ).to_dict(orient="index")
    out["panel_regression"] = panels

    # Decay: yearly mean 6m CAR for CFO and all insiders.
    df["year"] = pd.to_datetime(df["filing_date"]).dt.year
    decay = {}
    for g in ("ALL", "CFO"):
        m = SELECTORS[g](df).fillna(False).astype(bool)
        yearly = df.loc[m].groupby("year")["car_6"].mean()
        x = sm.add_constant(yearly.index.to_numpy(dtype=float))
        fit = sm.OLS(yearly.to_numpy(), x).fit()
        decay[g] = {
            "yearly_car6": yearly.round(4).to_dict(),
            "trend_per_year": float(fit.params[1]),
            "trend_t": float(fit.tvalues[1]),
        }
    out["decay"] = decay
    ctx.results["event_study"] = out
    return df


# --------------------------------------------------------------------------- ML check

ML_FEATURES = [
    "is_cfo",
    "is_ceo",
    "is_director_only",
    "is_ten_pct_only",
    "delta_h",
    "log_value",
    "cluster30",
    "filing_lag_days",
    "log_mcap",
    "mom_12_1",
    "log_amihud",
    "is_opp",
    "is_rout",
]


def run_ml_check(ctx: Context, df: pd.DataFrame) -> None:
    """Does a non-linear model beat a linear baseline under purged CPCV? (exploratory)"""
    d = df.loc[df["split"].isin(["train", "validation"])].copy()
    d["is_opp"] = d["pattern"].eq("OPPORTUNISTIC").astype(float)
    d["is_rout"] = d["pattern"].eq("ROUTINE").astype(float)
    d["is_ten_pct_only"] = d["is_ten_pct_only"].astype(float)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[*ML_FEATURES, "car_6"])
    d = d.sort_values("filing_date")
    y = d["car_6"].clip(*d["car_6"].quantile([0.01, 0.99]))
    x = d[ML_FEATURES].astype(float)
    dates = pd.to_datetime(d["filing_date"])
    k, k_test = ctx.cfg.validation.cpcv_groups, ctx.cfg.validation.cpcv_test_groups
    blocks = np.array_split(np.arange(len(d)), k)
    bounds = [(dates.iloc[b[0]], dates.iloc[b[-1]]) for b in blocks]
    embargo = pd.Timedelta(days=183)  # label window of the 6-month horizon
    ic: dict[str, list[float]] = {"linear": [], "gbm": []}
    rng_seed = ctx.cfg.seed
    for test_ids in combinations(range(k), k_test):
        test_idx = np.concatenate([blocks[i] for i in test_ids])
        train_mask = np.ones(len(d), dtype=bool)
        train_mask[test_idx] = False
        for i in test_ids:
            lo, hi = bounds[i]
            # Purge training events whose label window overlaps the test block,
            # plus an embargo after it.
            overlap = (dates + embargo >= lo) & (dates <= hi + embargo)
            train_mask &= ~overlap.to_numpy()
        lin = LinearRegression().fit(x.iloc[train_mask], y.iloc[train_mask])
        gbm = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=200,
            random_state=rng_seed,
        )
        gbm.fit(x.iloc[train_mask], y.iloc[train_mask])
        yt = y.iloc[test_idx]
        ic["linear"].append(float(spearmanr(lin.predict(x.iloc[test_idx]), yt).statistic))
        ic["gbm"].append(float(spearmanr(gbm.predict(x.iloc[test_idx]), yt).statistic))
    diff = np.array(ic["gbm"]) - np.array(ic["linear"])
    final = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        random_state=rng_seed,
    ).fit(x, y)
    sample = x.sample(min(20000, len(x)), random_state=rng_seed)
    imp = permutation_importance(
        final, sample, y.loc[sample.index], n_repeats=5, random_state=rng_seed, scoring="r2"
    )
    pred = pd.Series(final.predict(x), index=x.index)
    ctx.results["ml_check"] = {
        "n_events": len(d),
        "n_folds": len(ic["gbm"]),
        "ic_linear_mean": float(np.mean(ic["linear"])),
        "ic_gbm_mean": float(np.mean(ic["gbm"])),
        "ic_gbm_minus_linear_mean": float(diff.mean()),
        "ic_gbm_minus_linear_t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))),
        "share_folds_gbm_better": float((diff > 0).mean()),
        "permutation_importance": dict(
            zip(ML_FEATURES, map(float, imp.importances_mean), strict=True)
        ),
        "corr_pred_log_mcap": float(pred.corr(x["log_mcap"], method="spearman")),
        "corr_pred_log_amihud": float(pred.corr(x["log_amihud"], method="spearman")),
        "corr_pred_mom": float(pred.corr(x["mom_12_1"], method="spearman")),
    }


# --------------------------------------------------------------------------- entry points


def _dump(ctx: Context) -> Path:
    path = ctx.out_dir / "results.json"
    path.write_text(json.dumps(ctx.results, indent=2, default=_json_default))
    rows = {k: v.net for k, v in ctx.series.items() if "_AUM" not in k}
    pd.DataFrame(rows).to_parquet(ctx.out_dir / "monthly_net_returns.parquet")
    return path


def _json_default(o: object) -> object:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp, pd.Period)):
        return str(o)
    return str(o)


def run(cfg: StudyConfig, mode: Mode) -> Path:
    ctx = prepare(cfg, mode)
    registry = (
        TrialRegistry(cfg.tracking.trial_registry_path, cfg.validation.max_trials)
        if mode == "dev"
        else None
    )
    if registry is not None and registry.count() > 0:
        raise RuntimeError(
            "trial registry already has records: the dev study is registered once. "
            "Re-runs must not silently double-count trials."
        )
    ctx.results["config_hash"] = cfg.config_hash()
    ctx.results["mode"] = mode
    ctx.results["data_end"] = str(ctx.data_end.date())
    run_main(ctx, registry)
    run_robustness(ctx, registry)
    run_capacity(ctx, registry)
    run_unpriced_bounding(ctx)
    df = run_event_study(ctx)
    if mode == "dev":
        assert registry is not None
        run_ml_check(ctx, df)
        if registry is not None:
            registry.append(
                TrialRecord(
                    cfg.config_hash(), "ml", Split.VALIDATION, 0.0, 0, "ML CPCV check: linear"
                )
            )
            registry.append(
                TrialRecord(cfg.config_hash(), "ml", Split.VALIDATION, 0.0, 0, "ML CPCV check: gbm")
            )
        run_multiple_testing(ctx, registry)
    return _dump(ctx)
