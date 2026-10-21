from __future__ import annotations

from insider_study.config import StudyConfig
from insider_study.tracking import NullTracker, flatten


def test_flatten_nested() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_null_tracker_records_config_hash(cfg: StudyConfig) -> None:
    tracker = NullTracker()
    with tracker.start_run(cfg, "smoke"):
        tracker.log_metrics({"sharpe": 0.1})
    assert tracker.params["config_hash"] == cfg.config_hash()
    assert tracker.params["signal.entry_lag_sessions"] == 1
    assert tracker.metrics == {"sharpe": 0.1}
