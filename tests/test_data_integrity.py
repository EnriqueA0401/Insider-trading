"""F4: temporal-integrity checks on the REAL processed data (skipped if absent, e.g. CI)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from insider_study.backtest.engine import PortfolioSpec, holdings_table
from insider_study.data.market import rebalance_sessions
from insider_study.temporal import assert_no_lookahead

EVENTS = Path("data/processed/events.parquet")
ENRICHED = Path("data/processed/dev/events_enriched.parquet")
pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    if not EVENTS.exists():
        pytest.skip("processed SEC events not available")
    return pd.read_parquet(EVENTS)


def test_every_event_enters_strictly_after_filing(events: pd.DataFrame) -> None:
    assert_no_lookahead(events)


def test_no_event_is_dated_by_transaction_date(events: pd.DataFrame) -> None:
    """Entry is a function of the filing date only; a late filing must enter late."""
    late = events.loc[events["filing_lag_days"] >= 5]
    assert (
        pd.to_datetime(late["entry_date"])
        > pd.to_datetime(late["first_trade_date"]) + pd.Timedelta(days=4)
    ).all()


def test_filing_lag_cap_respected(events: pd.DataFrame) -> None:
    assert events["filing_lag_days"].between(0, 30).all()


def test_only_open_market_purchases(events: pd.DataFrame) -> None:
    assert set(events["form_type"]) <= {"4", "4/A"}


def test_portfolio_holdings_only_use_past_filings(events: pd.DataFrame) -> None:
    sample = events.sample(20000, random_state=0).assign(yf_ticker=lambda d: d["ticker"])
    rebal = rebalance_sessions(pd.Timestamp("2006-01-01").date(), pd.Timestamp("2019-12-31").date())
    spec = PortfolioSpec("x", 6, 1e6, 0.05, -0.3)
    h = holdings_table(sample, rebal, spec)
    # Join back: every held (rebal, ticker) must have an event filed before the rebalance.
    first_filing = sample.groupby("yf_ticker")["filing_date"].min()
    assert (
        pd.to_datetime(first_filing.reindex(h["ticker"]).to_numpy()) < h["rebal"].to_numpy()
    ).all()


def test_dev_artifacts_never_contain_holdout_events() -> None:
    if not ENRICHED.exists():
        pytest.skip("dev artifacts not built")
    ev = pd.read_parquet(ENRICHED, columns=["filing_date"])
    assert pd.to_datetime(ev["filing_date"]).max() < pd.Timestamp("2020-01-01")
