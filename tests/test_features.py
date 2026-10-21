from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from insider_study.domain import Col
from insider_study.signals.features import (
    attach_cmp,
    cluster_size,
    cmp_classification,
    expanding_terciles,
    size_bucket,
)


def _hist(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=[Col.INSIDER_CIK, Col.TRANSACTION_DATE, Col.FILING_DATE, Col.TRANS_CODE]
    )


def test_cmp_routine_vs_opportunistic() -> None:
    h = _hist(
        [
            # routine: every March of 2007-2009
            ("r", "2007-03-05", "2007-03-06", "S"),
            ("r", "2008-03-05", "2008-03-06", "S"),
            ("r", "2009-03-05", "2009-03-06", "P"),
            # opportunistic: traded each year, different months
            ("o", "2007-01-05", "2007-01-06", "P"),
            ("o", "2008-06-05", "2008-06-06", "P"),
            ("o", "2009-11-05", "2009-11-06", "P"),
            # unclassified: gap year
            ("u", "2007-01-05", "2007-01-06", "P"),
            ("u", "2009-01-05", "2009-01-06", "P"),
        ]
    )
    cls = cmp_classification(h, range(2010, 2011)).set_index("insider_cik")["pattern"]
    assert cls["r"] == "ROUTINE"
    assert cls["o"] == "OPPORTUNISTIC"
    assert "u" not in cls.index


def test_cmp_ignores_trades_filed_after_cutoff() -> None:
    """A Dec trade filed in January is not known at the Jan 1 classification date."""
    h = _hist(
        [
            ("x", "2007-03-05", "2007-03-06", "P"),
            ("x", "2008-03-05", "2008-03-06", "P"),
            ("x", "2009-12-30", "2010-01-04", "P"),  # filed after cutoff
        ]
    )
    assert cmp_classification(h, range(2010, 2011)).empty
    events = pd.DataFrame({Col.INSIDER_CIK: ["x"], Col.FILING_DATE: [date(2010, 6, 1)]})
    assert attach_cmp(events, cmp_classification(h, range(2010, 2011))).iloc[0] == "UNCLASSIFIED"


def test_expanding_terciles_use_only_past() -> None:
    dates = pd.date_range("2010-01-01", periods=60, freq="D")
    ev = pd.DataFrame({Col.FILING_DATE: dates, "v": np.arange(60, dtype=float)})
    t = expanding_terciles(ev, "v", min_history=10)
    assert t.loc[ev[Col.FILING_DATE].dt.month == 1].isna().all()
    # February values are all larger than every January value -> top tercile.
    assert (t.loc[ev[Col.FILING_DATE].dt.month == 2] == 3).all()


def test_cluster_counts_distinct_insiders_in_window() -> None:
    ev = pd.DataFrame(
        {
            Col.ISSUER_CIK: ["a", "a", "a", "a", "b"],
            Col.INSIDER_CIK: ["1", "2", "2", "3", "1"],
            Col.FILING_DATE: pd.to_datetime(
                ["2010-01-01", "2010-01-10", "2010-01-12", "2010-03-01", "2010-01-05"]
            ),
        }
    )
    assert list(cluster_size(ev, 30)) == [1, 2, 2, 1, 1]


def test_size_bucket_uses_previous_month_breakpoints() -> None:
    bp = pd.DataFrame(
        {"p20": [100.0, 1e9], "p50": [500.0, 1e9], "p80": [2000.0, 1e9]},
        index=pd.to_datetime(["2010-01-31", "2010-02-28"]),
    )
    out = size_bucket(
        pd.Series([50.0, 300.0, 1000.0, 5000.0]), pd.Series(pd.to_datetime(["2010-02-15"] * 4)), bp
    )
    assert list(out) == ["MICRO", "SMALL", "MID", "LARGE"]
