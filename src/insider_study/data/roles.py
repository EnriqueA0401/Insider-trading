"""Map free-text insider titles to canonical roles.

Rules (pre-registered, docs/HYPOTHESES.md section 5):
- CFO: chief financial officer / CFO / principal financial officer (incl. interim).
  Deputy/assistant CFO, treasurer, VP finance and controller are NOT CFO.
- CFO_BROAD (sensitivity only): CFO or principal/chief accounting officer.
- CEO: chief executive officer / CEO / principal executive officer.
- DIRECTOR (only): director flag and no officer role.
Dual roles keep every flag; `role` is a single label with precedence
CFO > CEO > OTHER_OFFICER > DIRECTOR > TEN_PCT_OWNER > OTHER.
"""

from __future__ import annotations

import re

import pandas as pd

from insider_study.domain import Col, InsiderRole

_SUBORDINATE = r"(?<!deputy )(?<!assistant )(?<!asst )(?<!asst\. )(?<!associate )(?<!vice )"

RE_CFO = re.compile(
    _SUBORDINATE
    + r"(?:\bc\.?\s?f\.?\s?o\b|chief\s+financ(?:ial|e)\s+officer|principal\s+financial\s+officer"
    r"|chief\s+financial\b|chief\s+fin\.?\s+off)",
    re.IGNORECASE,
)
RE_CAO = re.compile(r"(?:principal|chief)\s+accounting\s+officer|\bc\.?a\.?o\b", re.IGNORECASE)
RE_CEO = re.compile(
    _SUBORDINATE
    + r"(?:\bc\.?\s?e\.?\s?o\b|chief\s+exec(?:utive|\.)?(?:\s|$)|principal\s+executive\s+officer)",
    re.IGNORECASE,
)


def _flags(title: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    # Normalise separators so "EVP/CFO", "SVP-CFO", "EVP,CFO" all expose word boundaries.
    clean = title.fillna("").str.replace(r"[/,&;\-\(\)]", " ", regex=True)
    clean = clean.str.replace(r"\s+", " ", regex=True).str.lower()
    return clean.str.contains(pattern)


def map_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean role flags and a single `role` label."""
    title = df[Col.RAW_TITLE]
    out = df.copy()
    officer_or_titled = df[Col.IS_OFFICER] | title.fillna("").str.len().gt(0)
    out["is_cfo"] = _flags(title, RE_CFO)
    out["is_ceo"] = _flags(title, RE_CEO)
    out["is_cfo_broad"] = out["is_cfo"] | _flags(title, RE_CAO)
    out["is_officer_any"] = df[Col.IS_OFFICER] | out["is_cfo"] | out["is_ceo"]
    out["is_director_only"] = df[Col.IS_DIRECTOR] & ~out["is_officer_any"]
    out["is_ten_pct_only"] = (
        df[Col.IS_TEN_PCT_OWNER] & ~out["is_officer_any"] & ~df[Col.IS_DIRECTOR]
    )
    out["is_dual_ceo_cfo"] = out["is_cfo"] & out["is_ceo"]

    role = pd.Series(InsiderRole.OTHER.value, index=df.index, dtype="object")
    role[df[Col.IS_TEN_PCT_OWNER]] = InsiderRole.TEN_PCT_OWNER.value
    role[df[Col.IS_DIRECTOR]] = InsiderRole.DIRECTOR.value
    role[officer_or_titled & df[Col.IS_OFFICER]] = InsiderRole.OTHER_OFFICER.value
    role[out["is_ceo"]] = InsiderRole.CEO.value
    role[out["is_cfo"]] = InsiderRole.CFO.value
    out[Col.ROLE] = role
    return out
