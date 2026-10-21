"""Declarative, validated study configuration.

Every run is fully determined by a `StudyConfig`. Its canonical hash identifies the
run in the trial registry, in MLflow and in the holdout unseal record, so any
change to a parameter is visible and auditable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitConfig(_Frozen):
    """Fixed train / validation / holdout windows, assigned by filing date."""

    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    holdout_start: date
    holdout_end: date

    @model_validator(mode="after")
    def _ordered_and_disjoint(self) -> Self:
        bounds = [
            self.train_start,
            self.train_end,
            self.validation_start,
            self.validation_end,
            self.holdout_start,
            self.holdout_end,
        ]
        if any(a >= b for a, b in pairwise(bounds)):
            raise ValueError("splits must be strictly ordered and non-overlapping")
        return self


class DataConfig(_Frozen):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    sec_user_agent: str = Field(
        min_length=5,
        description="SEC fair-access policy requires 'Name email' in the User-Agent.",
    )
    sec_max_requests_per_second: float = Field(default=8.0, gt=0, le=10)
    first_year: int = Field(ge=2003)
    last_year: int
    price_source: Literal["yfinance"]
    delisting_return_primary: float = Field(ge=-1.0, le=0.0)
    delisting_return_sensitivities: tuple[float, ...]
    max_unpriced_event_share: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _years(self) -> Self:
        if self.last_year < self.first_year:
            raise ValueError("last_year must be >= first_year")
        return self


class FilterConfig(_Frozen):
    eligible_trans_codes: tuple[str, ...]
    eligible_form_types: tuple[str, ...]
    max_filing_lag_days: int | None = Field(gt=0)
    exclude_10b5_1: bool
    max_price_deviation: float = Field(
        gt=0, description="Max |form price / market price - 1| for ticker validation."
    )


class SignalConfig(_Frozen):
    entry_lag_sessions: int = Field(
        ge=1,
        description="Sessions strictly after filing date. 0 is forbidden (look-ahead).",
    )
    entry_price: Literal["open", "close"]
    horizons_months: tuple[int, ...]
    primary_horizon_months: int
    routine_lookback_years: int = Field(ge=2)
    cluster_min_insiders: int = Field(ge=2)
    cluster_window_days: int = Field(gt=0)
    conviction_terciles_expanding: bool

    @model_validator(mode="after")
    def _primary_in_horizons(self) -> Self:
        if self.primary_horizon_months not in self.horizons_months:
            raise ValueError("primary_horizon_months must be one of horizons_months")
        return self


class PortfolioConfig(_Frozen):
    aum_usd: float = Field(gt=0)
    capacity_grid_usd: tuple[float, ...]
    rebalance: Literal["monthly"]
    weighting: Literal["equal"]
    max_position_weight: float = Field(gt=0, le=1)
    long_only: Literal[True]
    universes: tuple[Literal["all", "mid_large"], ...]


class CostConfig(_Frozen):
    commission_per_share: float = Field(ge=0)
    commission_min_usd: float = Field(ge=0)
    spread_model: Literal["corwin_schultz"]
    half_spread_floor_bps: float = Field(ge=0)
    half_spread_cap_bps: float = Field(gt=0)
    impact_coefficient: float = Field(ge=0)
    max_adv_participation: float = Field(gt=0, le=1)
    adv_window_days: int = Field(gt=0)
    cost_multipliers: tuple[float, ...]


class BenchmarkConfig(_Frozen):
    market_tickers: tuple[str, ...]
    baseline: Literal["all_insiders"]
    factor_model: Literal["ff5_umd"]


class ValidationConfig(_Frozen):
    max_trials: int = Field(gt=0)
    alpha_level: float = Field(gt=0, lt=1)
    holdout_alpha_level: float = Field(gt=0, lt=1)
    min_dsr: float = Field(gt=0, lt=1)
    max_pbo: float = Field(gt=0, lt=1)
    cpcv_groups: int = Field(ge=4)
    cpcv_test_groups: int = Field(ge=1)


class TrackingConfig(_Frozen):
    experiment_name: str
    tracking_uri: str
    trial_registry_path: Path


class StudyConfig(_Frozen):
    seed: int
    splits: SplitConfig
    data: DataConfig
    filters: FilterConfig
    signal: SignalConfig
    portfolio: PortfolioConfig
    costs: CostConfig
    benchmark: BenchmarkConfig
    validation: ValidationConfig
    tracking: TrackingConfig

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON form; identifies a run unambiguously."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> StudyConfig:
    """Load a YAML config, optionally deep-merging overrides (e.g. sensitivity runs)."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    if overrides:
        raw = _deep_merge(raw, overrides)
    return StudyConfig.model_validate(raw)
