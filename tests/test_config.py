from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from insider_study.config import StudyConfig, load_config
from tests.conftest import BASE_CONFIG


def test_base_config_is_valid_and_matches_preregistration() -> None:
    cfg = load_config(BASE_CONFIG)
    assert cfg.signal.primary_horizon_months == 6
    assert cfg.signal.horizons_months == (1, 3, 6, 12)
    assert cfg.portfolio.aum_usd == 1_000_000
    assert cfg.portfolio.long_only is True
    assert cfg.filters.eligible_trans_codes == ("P",)
    assert cfg.benchmark.market_tickers == ("IWM",)
    assert "mid_large" in cfg.portfolio.universes


def test_same_day_entry_is_rejected() -> None:
    with pytest.raises(ValidationError, match="entry_lag_sessions"):
        load_config(BASE_CONFIG, overrides={"signal": {"entry_lag_sessions": 0}})


def test_overlapping_splits_are_rejected() -> None:
    with pytest.raises(ValidationError, match="strictly ordered"):
        load_config(BASE_CONFIG, overrides={"splits": {"validation_start": "2014-06-01"}})


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(BASE_CONFIG, overrides={"signal": {"secret_knob": 1}})


def test_primary_horizon_must_be_tested() -> None:
    with pytest.raises(ValidationError, match="primary_horizon"):
        load_config(BASE_CONFIG, overrides={"signal": {"primary_horizon_months": 9}})


def test_long_short_is_not_allowed() -> None:
    with pytest.raises(ValidationError):
        load_config(BASE_CONFIG, overrides={"portfolio": {"long_only": False}})


def test_config_hash_is_stable_and_sensitive(cfg: StudyConfig) -> None:
    again = StudyConfig.model_validate(cfg.model_dump(mode="json"))
    assert again.config_hash() == cfg.config_hash()
    changed = cfg.model_copy(update={"seed": cfg.seed + 1})
    assert changed.config_hash() != cfg.config_hash()


def test_config_is_immutable(cfg: StudyConfig) -> None:
    with pytest.raises(ValidationError):
        cfg.seed = 1  # type: ignore[misc]


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)
