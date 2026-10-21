from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from insider_study.backtest.engine import PortfolioSpec, holdings_table, run_monthly
from insider_study.config import StudyConfig
from insider_study.data.market import corwin_schultz, monthly_panel_for_ticker, rebalance_sessions


def _spec(**kw: object) -> PortfolioSpec:
    base: dict[str, object] = {
        "name": "t",
        "horizon_months": 2,
        "aum_usd": 1e6,
        "max_position_weight": 0.5,
        "delisting_return": -0.3,
    }
    return PortfolioSpec(**(base | kw))  # type: ignore[arg-type]


def test_event_filed_on_last_day_enters_next_month_only() -> None:
    rebal = rebalance_sessions(date(2010, 1, 1), date(2010, 6, 30))
    ev = pd.DataFrame({"filing_date": [date(2010, 1, 29)], "yf_ticker": ["A"]})
    h = holdings_table(ev, rebal, _spec())
    months = sorted(pd.DatetimeIndex(h["rebal"]).month)
    assert months == [2, 3]  # never January: filed after the January rebalance


def test_skip_month_delays_entry() -> None:
    rebal = rebalance_sessions(date(2010, 1, 1), date(2010, 6, 30))
    ev = pd.DataFrame({"filing_date": [date(2010, 1, 15)], "yf_ticker": ["A"]})
    h = holdings_table(ev, rebal, _spec(skip_months=1))
    assert sorted(pd.DatetimeIndex(h["rebal"]).month) == [3, 4]


def _panel(rebal: pd.DatetimeIndex, ticker: str, ret: float, adv: float = 1e9) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rebal": rebal,
            "ticker": ticker,
            "ret_nodl": ret,
            "delisted": False,
            "adv20": adv,
            "sigma60": 0.02,
            "half_spread": 0.001,
            "raw_px": 20.0,
        }
    )


def test_returns_costs_and_liquidity_cap(cfg: StudyConfig) -> None:
    rebal = rebalance_sessions(date(2010, 1, 1), date(2010, 4, 30))
    panel = pd.concat([_panel(rebal, "A", 0.10), _panel(rebal, "B", 0.0, adv=1e5)])
    ev = pd.DataFrame({"filing_date": [date(2009, 12, 15)] * 2, "yf_ticker": ["A", "B"]})
    rf = pd.Series(0.0, index=rebal.to_period("M").to_timestamp("M"))
    res = run_monthly(ev, panel, rebal, rf, _spec(), cfg.costs)
    jan = res.gross.iloc[0]
    # A gets 50%; B capped at 10% * 1e5 / 1e6 = 1%; rest in cash.
    assert jan == pytest.approx(0.5 * 0.10)
    assert res.cash_weight.iloc[0] == pytest.approx(0.49)
    assert res.net.iloc[0] < jan  # costs are charged on entry
    assert res.n_holdings.iloc[2] == 0  # horizon of two months is over


def test_delisting_return_applied(cfg: StudyConfig) -> None:
    rebal = rebalance_sessions(date(2010, 1, 1), date(2010, 2, 28))
    panel = _panel(rebal, "A", 0.0)
    panel["delisted"] = True
    ev = pd.DataFrame({"filing_date": [date(2009, 12, 15)], "yf_ticker": ["A"]})
    rf = pd.Series(0.0, index=rebal.to_period("M").to_timestamp("M"))
    res = run_monthly(ev, panel, rebal, rf, _spec(max_position_weight=1.0), cfg.costs)
    assert res.gross.iloc[0] == pytest.approx(-0.3)


def test_corwin_schultz_non_negative() -> None:
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    high, low = close * 1.01, close * 0.99
    assert (corwin_schultz(high, low).dropna() >= 0).all()


def test_monthly_panel_detects_delisting_and_uses_past_liquidity() -> None:
    idx = pd.bdate_range("2010-01-04", "2010-03-15")
    n = len(idx)
    frame = pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.0,
            "adj_close": 10.0,
            "adj_open": 10.0,
            "raw_close": 10.0,
            "dollar_volume": np.r_[np.full(n - 1, 1e6), 1e12],  # spike on last day
        },
        index=idx,
    )
    rebal = rebalance_sessions(date(2010, 1, 1), date(2010, 6, 30))
    p = monthly_panel_for_ticker(frame, rebal, pd.Timestamp("2010-06-30"))
    assert bool(p.loc[p["rebal"] == pd.Timestamp("2010-03-01"), "delisted"].iloc[0])
    assert p["adv20"].dropna().max() < 1e7  # the future spike never leaks back


def test_exit_without_market_data_has_bounded_cost(cfg: StudyConfig) -> None:
    """A holding that stops trading must not be charged an absurd impact cost."""
    rebal = rebalance_sessions(date(2010, 1, 1), date(2010, 3, 31))
    panel = _panel(rebal[:1], "A", 0.0)  # A has no market data after January
    ev = pd.DataFrame({"filing_date": [date(2009, 12, 15)], "yf_ticker": ["A"]})
    rf = pd.Series(0.0, index=rebal.to_period("M").to_timestamp("M"))
    res = run_monthly(ev, panel, rebal, rf, _spec(max_position_weight=1.0), cfg.costs)
    # Exit in February: at most cap spread (5%) + sigma(5%) * sqrt(1) + commission.
    assert res.cost.iloc[1] < 0.11


def test_investability_screen_and_cost_cap(cfg: StudyConfig) -> None:
    """Sub-$1 names are not bought; no trade ever costs more than its notional."""
    rebal = rebalance_sessions(date(2010, 1, 1), date(2010, 3, 31))
    penny = _panel(rebal, "P", 100.0, adv=1e9)
    penny["raw_px"] = 0.5
    wild = _panel(rebal, "W", 0.0, adv=1e9)
    wild["sigma60"] = 16.0  # bad-tick volatility
    ev = pd.DataFrame({"filing_date": [date(2009, 12, 15)] * 2, "yf_ticker": ["P", "W"]})
    rf = pd.Series(0.0, index=rebal.to_period("M").to_timestamp("M"))
    spec = _spec(max_position_weight=1.0, min_price_usd=1.0, min_adv_usd=1e4)
    res = run_monthly(ev, pd.concat([penny, wild]), rebal, rf, spec, cfg.costs)
    assert res.n_holdings.iloc[0] == 1  # the penny stock is screened out
    assert res.cost.iloc[0] <= 1.0
    assert res.cost.iloc[0] < 0.2  # sigma input capped at 25%/day
