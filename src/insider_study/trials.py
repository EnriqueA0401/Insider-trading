"""Append-only trial registry: the honest count of configurations tried.

Every evaluation (including failed ideas and sensitivity runs) appends one line.
The Deflated Sharpe Ratio and PBO read N and the Sharpe distribution from here,
so the multiple-testing correction cannot be understated by hand.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from insider_study.domain import Split


@dataclass(frozen=True)
class TrialRecord:
    config_hash: str
    hypothesis: str
    split: Split
    sharpe: float
    n_periods: int
    description: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class TrialBudgetExceededError(RuntimeError):
    pass


class TrialRegistry:
    def __init__(self, path: Path, max_trials: int) -> None:
        self.path = path
        self.max_trials = max_trials

    def _iter(self) -> Iterator[TrialRecord]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    raw = json.loads(line)
                    raw["split"] = Split(raw["split"])
                    yield TrialRecord(**raw)

    def records(self, split: Split | None = None) -> list[TrialRecord]:
        return [r for r in self._iter() if split is None or r.split == split]

    def count(self, exclude_holdout: bool = True) -> int:
        return sum(1 for r in self._iter() if not (exclude_holdout and r.split == Split.HOLDOUT))

    def sharpes(self, split: Split | None = None) -> list[float]:
        return [r.sharpe for r in self.records(split)]

    def append(self, record: TrialRecord) -> None:
        if record.split != Split.HOLDOUT and self.count() >= self.max_trials:
            raise TrialBudgetExceededError(
                f"pre-registered budget of {self.max_trials} trials reached; "
                "file an amendment in docs/HYPOTHESES.md before continuing"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(record) | {"split": record.split.value}
        with self.path.open("a") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
