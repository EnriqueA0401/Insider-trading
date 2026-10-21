from __future__ import annotations

from pathlib import Path

import pytest

from insider_study.config import StudyConfig, load_config

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


@pytest.fixture
def cfg(tmp_path: Path) -> StudyConfig:
    return load_config(
        BASE_CONFIG,
        overrides={"tracking": {"trial_registry_path": str(tmp_path / "registry.jsonl")}},
    )
