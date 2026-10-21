from __future__ import annotations

from pathlib import Path

import pytest

from insider_study.domain import Split
from insider_study.trials import TrialBudgetExceededError, TrialRecord, TrialRegistry


def _rec(split: Split = Split.TRAIN, sharpe: float = 0.5) -> TrialRecord:
    return TrialRecord("abc", "H1", split, sharpe, 120, "test")


def test_registry_is_append_only_and_counts(tmp_path: Path) -> None:
    reg = TrialRegistry(tmp_path / "r.jsonl", max_trials=10)
    reg.append(_rec(sharpe=0.1))
    reg.append(_rec(Split.VALIDATION, sharpe=0.2))
    reg.append(_rec(Split.HOLDOUT, sharpe=0.3))
    assert reg.count() == 2
    assert reg.count(exclude_holdout=False) == 3
    assert reg.sharpes(Split.VALIDATION) == [0.2]
    # A fresh instance reads the same history from disk.
    assert TrialRegistry(tmp_path / "r.jsonl", 10).count() == 2


def test_budget_is_enforced(tmp_path: Path) -> None:
    reg = TrialRegistry(tmp_path / "r.jsonl", max_trials=2)
    reg.append(_rec())
    reg.append(_rec())
    with pytest.raises(TrialBudgetExceededError):
        reg.append(_rec())
