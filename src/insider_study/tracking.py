"""Experiment tracking. MLflow by default; a null tracker for tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol

import mlflow

from insider_study.config import StudyConfig
from insider_study.reproducibility import environment_fingerprint


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out |= flatten(value, name)
        else:
            out[name] = value
    return out


class ExperimentTracker(Protocol):
    def start_run(self, cfg: StudyConfig, run_name: str) -> Any: ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...


class NullTracker:
    def __init__(self) -> None:
        self.metrics: dict[str, float] = {}
        self.params: dict[str, Any] = {}

    @contextmanager
    def start_run(self, cfg: StudyConfig, run_name: str) -> Iterator[None]:
        self.params = flatten(cfg.model_dump(mode="json")) | {"config_hash": cfg.config_hash()}
        yield

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self.metrics |= dict(metrics)


class MLflowTracker:
    @contextmanager
    def start_run(self, cfg: StudyConfig, run_name: str) -> Iterator[None]:
        mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
        mlflow.set_experiment(cfg.tracking.experiment_name)
        with mlflow.start_run(run_name=run_name):
            params = flatten(cfg.model_dump(mode="json"))
            # MLflow caps param values at 6000 chars; stringify for safety.
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.set_tags(
                {"config_hash": cfg.config_hash()}
                | {k: str(v) for k, v in environment_fingerprint().items()}
            )
            mlflow.log_text(cfg.canonical_json(), "config.json")
            yield

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        mlflow.log_metrics(dict(metrics))
