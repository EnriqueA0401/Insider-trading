"""Point-in-time primitives. The single most important module in the study.

Rule: a signal derived from a Form 4 is knowable only after its FILING date.
EDGAR accepts Section 16 filings until 22:00 ET with that day's date, so the
earliest tradable price is the open of the first NYSE session strictly after the
filing date. `transaction_date` must never drive a portfolio decision.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from insider_study.domain import Col


class LookAheadError(AssertionError):
    """Raised when information is used before it was publicly available."""


@lru_cache(maxsize=1)
def _nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS", start="1999-01-01")


def next_session_after(day: date, n: int = 1) -> date:
    """Return the n-th NYSE session strictly after `day` (n >= 1)."""
    if n < 1:
        raise LookAheadError("entry must be strictly after the filing date (n >= 1)")
    cal = _nyse()
    ts = pd.Timestamp(day)
    # date_to_session('previous') maps weekends/holidays to the last session <= day,
    # so offsetting by n always lands strictly after `day`.
    anchor = cal.date_to_session(ts, direction="previous")
    session: pd.Timestamp = cal.session_offset(anchor, n)
    return session.date()


def entry_dates(filing_dates: pd.Series, lag_sessions: int) -> pd.Series:
    """Vectorised `next_session_after` for a Series of filing dates."""
    if lag_sessions < 1:
        raise LookAheadError("lag_sessions must be >= 1")
    unique = pd.Series(pd.to_datetime(filing_dates).dt.date.unique())
    mapping = {d: next_session_after(d, lag_sessions) for d in unique}
    return pd.to_datetime(filing_dates).dt.date.map(mapping)


def assert_no_lookahead(
    df: pd.DataFrame,
    decision_col: str = Col.ENTRY_DATE,
    knowledge_col: str = Col.FILING_DATE,
) -> None:
    """Fail if any decision happens on or before the information became public.

    Every row must satisfy decision > knowledge (strict: filing-day prices are
    contaminated because the filing may arrive after the close).
    """
    decision = pd.to_datetime(df[decision_col])
    knowledge = pd.to_datetime(df[knowledge_col])
    bad = decision <= knowledge
    if bad.any():
        sample = df.loc[bad, [knowledge_col, decision_col]].head(5).to_dict("records")
        raise LookAheadError(
            f"{int(bad.sum())} rows decide on/before {knowledge_col}; e.g. {sample}"
        )


def known_as_of(
    df: pd.DataFrame, as_of: date, knowledge_col: str = Col.FILING_DATE
) -> pd.DataFrame:
    """Rows publicly known strictly before the session `as_of` (point-in-time slice)."""
    return df.loc[pd.to_datetime(df[knowledge_col]) < pd.Timestamp(as_of)]
