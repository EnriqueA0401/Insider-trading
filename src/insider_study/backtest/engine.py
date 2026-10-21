"""Monthly long-only calendar-time backtester with transaction costs and capacity limits.

At the first session of month m the target portfolio holds every stock with at
least one selected event FILED in months m-H .. m-1 (so filing date < rebalance
session: no look-ahead), equal-weighted, subject to:
  - max position weight,
  - liquidity cap: position <= max_adv_participation * ADV20 (USD),
  - excess weight stays in cash earning the risk-free rate.
Costs on traded notional: half-spread + square-root impact + per-share commission.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from insider_study.config import CostConfig


@dataclass(frozen=True)
class PortfolioSpec:
    name: str
    horizon_months: int
    aum_usd: float
    max_position_weight: float
    delisting_return: float
    cost_multiplier: float = 1.0
    skip_months: int = 0  # robustness: extra delay before a signal is used
    min_price_usd: float = 0.0
    min_adv_usd: float = 0.0


@dataclass(frozen=True)
class MonthlyResult:
    gross: pd.Series  # indexed by month-end
    net: pd.Series
    cash_weight: pd.Series
    n_holdings: pd.Series
    turnover: pd.Series
    cost: pd.Series


def holdings_table(
    events: pd.DataFrame, rebal: pd.DatetimeIndex, spec: PortfolioSpec
) -> pd.DataFrame:
    """(rebal, ticker) pairs to hold, from events with yf_ticker and filing_date."""
    month_f = pd.to_datetime(events["filing_date"]).dt.to_period("M")
    rebal_month = pd.PeriodIndex(rebal.to_period("M"))
    pairs = []
    for k in range(1 + spec.skip_months, spec.horizon_months + 1 + spec.skip_months):
        pairs.append(pd.DataFrame({"month": month_f + k, "ticker": events["yf_ticker"].to_numpy()}))
    h = pd.concat(pairs, ignore_index=True).drop_duplicates()
    m2r = pd.Series(rebal, index=rebal_month)
    h = h.loc[h["month"].isin(rebal_month)]
    h["rebal"] = m2r.reindex(h["month"]).to_numpy()
    return h[["rebal", "ticker"]]


def run_monthly(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    rebal: pd.DatetimeIndex,
    rf_monthly: pd.Series,
    spec: PortfolioSpec,
    costs: CostConfig,
) -> MonthlyResult:
    hold = holdings_table(events, rebal, spec).merge(panel, on=["rebal", "ticker"], how="inner")
    hold = hold.loc[hold["ret_nodl"].notna()]
    # Investability screen, measured before the rebalance (no look-ahead).
    hold = hold.loc[(hold["raw_px"] >= spec.min_price_usd) & (hold["adv20"] >= spec.min_adv_usd)]
    dl = spec.delisting_return
    hold["ret"] = (1 + hold["ret_nodl"]) * np.where(hold["delisted"], 1 + dl, 1.0) - 1

    n = hold.groupby("rebal")["ticker"].transform("size")
    w_eq = np.minimum(1.0 / n, spec.max_position_weight)
    liq_cap = costs.max_adv_participation * hold["adv20"].fillna(0.0) / spec.aum_usd
    hold["w"] = np.minimum(w_eq, liq_cap)

    mult = spec.cost_multiplier
    lo, hi = costs.half_spread_floor_bps / 1e4, costs.half_spread_cap_bps / 1e4
    month_end = rebal.to_period("M").to_timestamp("M")
    rf = rf_monthly.reindex(month_end).fillna(0.0).to_numpy()

    by_month = {k: g.set_index("ticker") for k, g in hold.groupby("rebal")}
    # Market stats for every ticker (exits are priced with their own liquidity too).
    stats_by_month = {
        k: g.set_index("ticker")[["half_spread", "sigma60", "adv20", "raw_px"]]
        for k, g in panel.loc[panel["ticker"].isin(hold["ticker"].unique())].groupby("rebal")
    }
    prev_w = pd.Series(dtype=float)  # drifted weights carried into this rebalance
    gross, net, cash_w, n_h, turn, cost = [], [], [], [], [], []
    for i, r in enumerate(rebal):
        g = by_month.get(r)
        target = g["w"] if g is not None else pd.Series(dtype=float)
        traded = target.subtract(prev_w, fill_value=0.0).abs()
        c = 0.0
        if len(traded) and traded.sum() > 0:
            notional = traded * spec.aum_usd
            st = stats_by_month.get(r, pd.DataFrame()).reindex(traded.index)
            # Missing stats -> conservative defaults (max spread, 5% daily vol).
            hs_t = st["half_spread"].fillna(hi).clip(lo, hi) if len(st.columns) else hi
            sig = np.minimum(
                st["sigma60"].fillna(0.05) if len(st.columns) else 0.05, costs.max_sigma_daily
            )
            px = st["raw_px"].fillna(10.0).clip(lower=0.01) if len(st.columns) else 10.0
            # Participation = notional / ADV; unknown ADV -> assume the cap. Bounded at one
            # full day of volume (larger trades would be worked over several days).
            adv = st["adv20"] if len(st.columns) else pd.Series(np.nan, index=traded.index)
            part = (
                (notional / adv.where(adv > 0)).fillna(costs.max_adv_participation).clip(upper=1.0)
            )
            impact = costs.impact_coefficient * sig * np.sqrt(part)
            commission = np.maximum(
                costs.commission_min_usd, costs.commission_per_share * notional / px
            )
            # A trade can never cost more than its own notional.
            per_name = np.minimum((notional * (hs_t + impact) + commission) * mult, notional)
            c = float(per_name.sum()) / spec.aum_usd
        w_sum = float(target.sum())
        cash = 1.0 - w_sum
        port = float((target * g["ret"]).sum()) if g is not None else 0.0
        gr = port + cash * rf[i]
        gross.append(gr)
        net.append(gr - c)
        cash_w.append(cash)
        n_h.append(len(target))
        turn.append(float(traded.sum()) / 2.0)
        cost.append(c)
        if g is not None and len(target):
            drifted = target * (1 + g["ret"])
            # Delisted names leave the book (proceeds go to cash).
            drifted = drifted.loc[~g["delisted"].reindex(drifted.index).fillna(False).astype(bool)]
            prev_w = drifted / (1 + gr)
        else:
            prev_w = pd.Series(dtype=float)
    idx = pd.DatetimeIndex(month_end)
    return MonthlyResult(
        gross=pd.Series(gross, index=idx),
        net=pd.Series(net, index=idx),
        cash_weight=pd.Series(cash_w, index=idx),
        n_holdings=pd.Series(n_h, index=idx),
        turnover=pd.Series(turn, index=idx),
        cost=pd.Series(cost, index=idx),
    )
