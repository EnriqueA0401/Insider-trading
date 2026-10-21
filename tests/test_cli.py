from __future__ import annotations

import json

import pytest

from insider_study.cli import main
from tests.conftest import BASE_CONFIG


def test_validate_config_prints_hash(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate-config", str(BASE_CONFIG)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is True
    assert len(out["config_hash"]) == 64
