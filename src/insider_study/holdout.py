"""Holdout guard: the out-of-sample window stays sealed until the final evaluation.

Data access layers call `HoldoutGuard.check_range` / `clip` before returning any
time-indexed data. Unsealing requires a committed `HOLDOUT_UNSEALED` file whose
content is the hash of the frozen config; any later config change re-seals it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from insider_study.config import StudyConfig
from insider_study.domain import Split

UNSEAL_FILE = Path("HOLDOUT_UNSEALED")


class HoldoutAccessError(PermissionError):
    """Raised on any attempt to read holdout-period data while sealed."""


@dataclass(frozen=True)
class HoldoutGuard:
    holdout_start: date
    config_hash: str
    unseal_file: Path = UNSEAL_FILE

    @classmethod
    def from_config(cls, cfg: StudyConfig, unseal_file: Path = UNSEAL_FILE) -> HoldoutGuard:
        return cls(cfg.splits.holdout_start, cfg.config_hash(), unseal_file)

    @property
    def is_unsealed(self) -> bool:
        if not self.unseal_file.exists():
            return False
        record = json.loads(self.unseal_file.read_text())
        if record.get("config_hash") != self.config_hash:
            raise HoldoutAccessError(
                "HOLDOUT_UNSEALED was issued for a different config hash; the holdout "
                "can only be evaluated with the exact configuration that was frozen."
            )
        return True

    def check_range(self, start: date, end: date) -> None:
        if end >= self.holdout_start and not self.is_unsealed:
            raise HoldoutAccessError(
                f"requested data up to {end} but holdout starts {self.holdout_start} and is sealed"
            )

    def clip(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Drop holdout-period rows unless unsealed (used for purging label windows)."""
        if self.is_unsealed:
            return df
        return df.loc[pd.to_datetime(df[date_col]) < pd.Timestamp(self.holdout_start)]

    @staticmethod
    def unseal(cfg: StudyConfig, reason: str, unseal_file: Path = UNSEAL_FILE) -> Path:
        """Write the one-time unseal record. Must be committed to git to be auditable."""
        if unseal_file.exists():
            raise HoldoutAccessError("holdout already unsealed; it can be used only once")
        record = {
            "config_hash": cfg.config_hash(),
            "unsealed_at_utc": datetime.now(UTC).isoformat(),
            "reason": reason,
        }
        unseal_file.write_text(json.dumps(record, indent=2) + "\n")
        return unseal_file


def second_look(cfg: StudyConfig, reason: str, unseal_file: Path = UNSEAL_FILE) -> Path:
    """Re-open the holdout for a NEW config hash after it has already been looked at.

    Only for documented fixes (e.g. a data bug found in the first holdout run). The
    previous unseal records are kept in `history`, so every look at the holdout stays
    visible in the committed file; reports must show the results of every look.
    """
    if not unseal_file.exists():
        raise HoldoutAccessError("holdout was never unsealed; use unseal()")
    record = json.loads(unseal_file.read_text())
    history = [*record.pop("history", []), record]
    if any(r.get("config_hash") == cfg.config_hash() for r in history):
        raise HoldoutAccessError("this config hash already had its holdout look")
    new = {
        "config_hash": cfg.config_hash(),
        "unsealed_at_utc": datetime.now(UTC).isoformat(),
        "reason": reason,
        "look_number": len(history) + 1,
        "history": history,
    }
    unseal_file.write_text(json.dumps(new, indent=2) + "\n")
    return unseal_file


def assign_split(filing_dates: pd.Series, cfg: StudyConfig) -> pd.Series:
    """Map filing dates to train / validation / holdout (NaN-like None outside)."""
    s = cfg.splits
    d = pd.to_datetime(filing_dates)
    out = pd.Series(pd.NA, index=filing_dates.index, dtype="object")
    for split, lo, hi in (
        (Split.TRAIN, s.train_start, s.train_end),
        (Split.VALIDATION, s.validation_start, s.validation_end),
        (Split.HOLDOUT, s.holdout_start, s.holdout_end),
    ):
        mask = (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
        out[mask] = split.value
    return out
