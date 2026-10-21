"""Temporal-integrity tests: the signal must be dated by FILING date, never transaction date."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from insider_study.domain import Col
from insider_study.temporal import (
    LookAheadError,
    _nyse,
    assert_no_lookahead,
    entry_dates,
    known_as_of,
    next_session_after,
)


@pytest.mark.parametrize(
    ("filing", "expected"),
    [
        (date(2024, 7, 12), date(2024, 7, 15)),  # Friday -> Monday
        (date(2024, 7, 3), date(2024, 7, 5)),  # day before Independence Day
        (date(2024, 7, 6), date(2024, 7, 8)),  # Saturday filing date
        (date(2024, 3, 28), date(2024, 4, 1)),  # Good Friday closure
        (date(2012, 10, 26), date(2012, 10, 31)),  # Hurricane Sandy closure
        (date(2024, 7, 10), date(2024, 7, 11)),  # ordinary Wednesday
    ],
)
def test_entry_is_first_session_strictly_after_filing(filing: date, expected: date) -> None:
    assert next_session_after(filing) == expected


def test_lag_two_sessions() -> None:
    assert next_session_after(date(2024, 7, 12), n=2) == date(2024, 7, 16)


def test_zero_lag_is_forbidden() -> None:
    with pytest.raises(LookAheadError):
        next_session_after(date(2024, 7, 12), n=0)
    with pytest.raises(LookAheadError):
        entry_dates(pd.Series([date(2024, 7, 12)]), lag_sessions=0)


@settings(max_examples=300, deadline=None)
@given(st.dates(min_value=date(2003, 1, 1), max_value=date(2025, 12, 31)))
def test_entry_always_strictly_after_filing_and_a_session(filing: date) -> None:
    entry = next_session_after(filing)
    assert entry > filing
    assert _nyse().is_session(pd.Timestamp(entry))
    assert entry - filing <= timedelta(days=6)


def _events(transaction: list[str], filing: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            Col.TRANSACTION_DATE: pd.to_datetime(transaction).date,
            Col.FILING_DATE: pd.to_datetime(filing).date,
        }
    )


def test_detects_transaction_date_leakage() -> None:
    """A naive builder that enters after the TRANSACTION date must be caught."""
    events = _events(
        transaction=["2010-03-01", "2015-06-10", "2019-11-04"],
        filing=["2010-03-03", "2015-06-19", "2019-11-06"],  # filed 2+ days later
    )
    events[Col.ENTRY_DATE] = entry_dates(events[Col.TRANSACTION_DATE], lag_sessions=1)
    with pytest.raises(LookAheadError):
        assert_no_lookahead(events)


def test_filing_date_entry_passes_check() -> None:
    events = _events(
        transaction=["2010-03-01", "2015-06-10", "2019-11-04"],
        filing=["2010-03-03", "2015-06-19", "2019-11-06"],
    )
    events[Col.ENTRY_DATE] = entry_dates(events[Col.FILING_DATE], lag_sessions=1)
    assert_no_lookahead(events)
    assert (pd.to_datetime(events[Col.ENTRY_DATE]) > pd.to_datetime(events[Col.FILING_DATE])).all()


def test_same_day_entry_is_flagged_as_leak() -> None:
    events = _events(transaction=["2020-01-02"], filing=["2020-01-03"])
    events[Col.ENTRY_DATE] = events[Col.FILING_DATE]
    with pytest.raises(LookAheadError):
        assert_no_lookahead(events)


def test_known_as_of_excludes_same_day_filings() -> None:
    events = _events(
        transaction=["2020-01-01", "2020-01-01", "2020-01-01"],
        filing=["2020-01-02", "2020-01-03", "2020-01-06"],
    )
    visible = known_as_of(events, date(2020, 1, 3))
    assert list(pd.to_datetime(visible[Col.FILING_DATE]).dt.date) == [date(2020, 1, 2)]
