from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from insider_study.config import StudyConfig
from insider_study.domain import Split
from insider_study.holdout import HoldoutAccessError, HoldoutGuard, assign_split


def test_sealed_holdout_blocks_access(cfg: StudyConfig, tmp_path: Path) -> None:
    guard = HoldoutGuard.from_config(cfg, unseal_file=tmp_path / "HOLDOUT_UNSEALED")
    guard.check_range(date(2004, 1, 1), date(2019, 12, 31))
    with pytest.raises(HoldoutAccessError):
        guard.check_range(date(2018, 1, 1), date(2020, 1, 1))


def test_clip_removes_holdout_rows(cfg: StudyConfig, tmp_path: Path) -> None:
    guard = HoldoutGuard.from_config(cfg, unseal_file=tmp_path / "HOLDOUT_UNSEALED")
    df = pd.DataFrame({"d": pd.to_datetime(["2019-12-31", "2020-01-01", "2023-05-05"])})
    assert len(guard.clip(df, "d")) == 1


def test_unseal_only_for_frozen_config(cfg: StudyConfig, tmp_path: Path) -> None:
    unseal_file = tmp_path / "HOLDOUT_UNSEALED"
    HoldoutGuard.unseal(cfg, reason="final evaluation", unseal_file=unseal_file)
    assert json.loads(unseal_file.read_text())["config_hash"] == cfg.config_hash()

    HoldoutGuard.from_config(cfg, unseal_file).check_range(date(2020, 1, 1), date(2025, 12, 31))

    tweaked = cfg.model_copy(update={"seed": cfg.seed + 1})
    with pytest.raises(HoldoutAccessError, match="different config hash"):
        HoldoutGuard.from_config(tweaked, unseal_file).check_range(
            date(2020, 1, 1), date(2025, 12, 31)
        )


def test_holdout_can_be_unsealed_only_once(cfg: StudyConfig, tmp_path: Path) -> None:
    unseal_file = tmp_path / "HOLDOUT_UNSEALED"
    HoldoutGuard.unseal(cfg, reason="first", unseal_file=unseal_file)
    with pytest.raises(HoldoutAccessError, match="only once"):
        HoldoutGuard.unseal(cfg, reason="second", unseal_file=unseal_file)


def test_assign_split_uses_boundaries(cfg: StudyConfig) -> None:
    dates = pd.Series(pd.to_datetime(["2003-12-31", "2014-12-31", "2015-01-01", "2020-01-01"]))
    out = assign_split(dates, cfg)
    assert out.isna().iloc[0]
    assert list(out.iloc[1:]) == [Split.TRAIN, Split.VALIDATION, Split.HOLDOUT]


def test_second_look_keeps_history(cfg: StudyConfig, tmp_path: Path) -> None:
    from insider_study.holdout import second_look

    f = tmp_path / "HOLDOUT_UNSEALED"
    HoldoutGuard.unseal(cfg, reason="first", unseal_file=f)
    fixed = cfg.model_copy(update={"seed": cfg.seed + 1})
    second_look(fixed, reason="data bug fix", unseal_file=f)
    rec = json.loads(f.read_text())
    assert rec["look_number"] == 2
    assert rec["history"][0]["config_hash"] == cfg.config_hash()
    HoldoutGuard.from_config(fixed, f).check_range(date(2020, 1, 1), date(2025, 12, 31))
    with pytest.raises(HoldoutAccessError):
        HoldoutGuard.from_config(cfg, f).check_range(date(2020, 1, 1), date(2025, 12, 31))
    with pytest.raises(HoldoutAccessError, match="already"):
        second_look(cfg, reason="again", unseal_file=f)
