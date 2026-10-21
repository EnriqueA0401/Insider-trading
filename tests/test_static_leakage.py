"""Static guard: decision-layer modules may not reference the transaction date.

Signals, portfolio construction, risk and backtesting must key everything off the
filing date. A line can be allow-listed only with an explicit `# leakage-ok:` reason.
"""

from __future__ import annotations

import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "insider_study"
DECISION_LAYER = ("signals", "portfolio", "risk", "backtest")
PATTERN = re.compile(r"transaction_date|TRANSACTION_DATE")


def find_violations(source: str) -> list[int]:
    return [
        i
        for i, line in enumerate(source.splitlines(), start=1)
        if PATTERN.search(line) and "# leakage-ok:" not in line
    ]


def test_scanner_catches_usage() -> None:
    src = "x = df[Col.TRANSACTION_DATE]\ny = df['transaction_date']  # leakage-ok: lag stat\n"
    assert find_violations(src) == [1]


def test_decision_layer_never_uses_transaction_date() -> None:
    offenders: list[str] = []
    for sub in DECISION_LAYER:
        for path in (PKG / sub).rglob("*.py") if (PKG / sub).exists() else []:
            offenders += [f"{path}:{n}" for n in find_violations(path.read_text())]
    assert not offenders, f"transaction_date used in decision layer: {offenders}"
