"""Domain vocabulary and tabular schemas shared across pipeline stages.

Stages exchange pandas DataFrames; the column sets below are the contract between
them. `validate_columns` is called at every stage boundary.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class TransactionCode(StrEnum):
    """SEC Form 4 transaction codes relevant to this study."""

    OPEN_MARKET_PURCHASE = "P"
    OPEN_MARKET_SALE = "S"
    GRANT = "A"
    OPTION_EXERCISE = "M"
    DERIVATIVE_EXERCISE = "X"
    CONVERSION = "C"
    TAX_WITHHOLDING = "F"
    GIFT = "G"
    INHERITANCE = "W"
    OTHER = "J"


class InsiderRole(StrEnum):
    CEO = "CEO"
    CFO = "CFO"
    DIRECTOR = "DIRECTOR"
    OTHER_OFFICER = "OTHER_OFFICER"
    TEN_PCT_OWNER = "TEN_PCT_OWNER"
    OTHER = "OTHER"


class TradingPattern(StrEnum):
    """Cohen-Malloy-Pomorski (2012) classification."""

    ROUTINE = "ROUTINE"
    OPPORTUNISTIC = "OPPORTUNISTIC"
    UNCLASSIFIED = "UNCLASSIFIED"


class SizeBucket(StrEnum):
    """NYSE market-cap percentile buckets (Ken French ME breakpoints)."""

    MICRO = "MICRO"  # < p20
    SMALL = "SMALL"  # p20 - p50
    MID = "MID"  # p50 - p80
    LARGE = "LARGE"  # > p80


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


class Col:
    """Canonical column names. Use these constants, never string literals."""

    ACCESSION = "accession_number"
    FORM_TYPE = "form_type"
    FILING_DATE = "filing_date"
    TRANSACTION_DATE = "transaction_date"
    ISSUER_CIK = "issuer_cik"
    TICKER = "ticker"
    INSIDER_CIK = "insider_cik"
    INSIDER_NAME = "insider_name"
    RAW_TITLE = "raw_title"
    IS_DIRECTOR = "is_director"
    IS_OFFICER = "is_officer"
    IS_TEN_PCT_OWNER = "is_ten_pct_owner"
    TRANS_CODE = "trans_code"
    ACQ_DISP = "acquired_disposed"
    SHARES = "shares"
    PRICE = "price"
    SHARES_OWNED_AFTER = "shares_owned_after"
    IS_10B5_1 = "is_10b5_1"
    IS_AMENDMENT = "is_amendment"
    ROLE = "role"
    ROLES = "roles"
    ENTRY_DATE = "entry_date"
    SPLIT = "split"


RAW_TRANSACTION_COLUMNS: frozenset[str] = frozenset(
    {
        Col.ACCESSION,
        Col.FORM_TYPE,
        Col.FILING_DATE,
        Col.TRANSACTION_DATE,
        Col.ISSUER_CIK,
        Col.TICKER,
        Col.INSIDER_CIK,
        Col.INSIDER_NAME,
        Col.RAW_TITLE,
        Col.IS_DIRECTOR,
        Col.IS_OFFICER,
        Col.IS_TEN_PCT_OWNER,
        Col.TRANS_CODE,
        Col.ACQ_DISP,
        Col.SHARES,
        Col.PRICE,
        Col.SHARES_OWNED_AFTER,
        Col.IS_10B5_1,
        Col.IS_AMENDMENT,
    }
)

NORMALIZED_TRANSACTION_COLUMNS: frozenset[str] = RAW_TRANSACTION_COLUMNS | {Col.ROLE, Col.ROLES}

EVENT_COLUMNS: frozenset[str] = NORMALIZED_TRANSACTION_COLUMNS | {Col.ENTRY_DATE, Col.SPLIT}


class SchemaError(ValueError):
    """Raised when a DataFrame does not satisfy a stage contract."""


def validate_columns(df: pd.DataFrame, required: frozenset[str], stage: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise SchemaError(f"{stage}: missing columns {sorted(missing)}")
