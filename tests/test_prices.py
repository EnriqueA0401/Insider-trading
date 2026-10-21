from __future__ import annotations

import pandas as pd
import pytest

from insider_study.data.prices import derive, to_yahoo


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("BRK.B", "BRK-B"), ("acme, acmew", "ACME"), ("$XYZ", "XYZ"), ("ABC/WS", "ABC-WS")],
)
def test_to_yahoo(raw: str, expected: str) -> None:
    assert to_yahoo(raw) == expected


def test_derive_undoes_split_adjustment_and_applies_dividends() -> None:
    idx = pd.bdate_range("2020-01-01", periods=4)
    # 2-for-1 split on day 3: yfinance reports pre-split closes halved.
    frame = pd.DataFrame(
        {
            "open": [50.0, 50.0, 50.0, 50.0],
            "high": [51.0] * 4,
            "low": [49.0] * 4,
            "close": [50.0, 50.0, 50.0, 50.0],
            "adj_close": [49.0, 49.0, 50.0, 50.0],
            "volume": [100.0] * 4,
            "dividends": [0.0] * 4,
            "splits": [0.0, 0.0, 2.0, 0.0],
        },
        index=idx,
    )
    d = derive(frame)
    assert list(d["raw_close"]) == [100.0, 100.0, 50.0, 50.0]
    assert d["adj_open"].iloc[0] == pytest.approx(49.0)
    assert d["dollar_volume"].iloc[0] == 5000.0
