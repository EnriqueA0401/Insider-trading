from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from insider_study.config import StudyConfig
from insider_study.data.events import build_events
from insider_study.data.filters import (
    SECTransactionFilter,
    drop_cross_filed_duplicates,
    resolve_amendments,
)
from insider_study.data.roles import map_roles
from insider_study.domain import Col


def _line(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        Col.ACCESSION: "a1",
        "line_id": "1",
        Col.FORM_TYPE: "4",
        Col.FILING_DATE: date(2010, 3, 3),
        "date_of_orig_sub": None,
        Col.TRANSACTION_DATE: date(2010, 3, 1),
        Col.ISSUER_CIK: "100",
        "issuer_name": "Acme",
        Col.TICKER: "ACME",
        Col.INSIDER_CIK: "900",
        Col.INSIDER_NAME: "Doe John",
        "relationship": "Officer",
        Col.RAW_TITLE: "CFO",
        Col.IS_DIRECTOR: False,
        Col.IS_OFFICER: True,
        Col.IS_TEN_PCT_OWNER: False,
        "n_owners": 1,
        "security_title": "Common Stock",
        Col.TRANS_CODE: "P",
        "timeliness": "",
        Col.ACQ_DISP: "A",
        Col.SHARES: 1000.0,
        Col.PRICE: 10.0,
        Col.SHARES_OWNED_AFTER: 5000.0,
        "direct_indirect": "D",
        "is_10b5_1_checkbox": False,
        Col.IS_10B5_1: False,
        Col.IS_AMENDMENT: False,
    }
    return base | kw


def _frame(*rows: dict[str, Any]) -> pd.DataFrame:
    return map_roles(pd.DataFrame(list(rows)))


def test_matched_amendment_is_dropped_and_orphan_kept() -> None:
    df = _frame(
        _line(),
        _line(
            **{
                Col.ACCESSION: "a2",
                Col.FORM_TYPE: "4/A",
                Col.FILING_DATE: date(2010, 4, 1),
                "date_of_orig_sub": date(2010, 3, 3),
                Col.IS_AMENDMENT: True,
            }
        ),
        _line(
            **{
                Col.ACCESSION: "a3",
                Col.FORM_TYPE: "4/A",
                Col.FILING_DATE: date(2010, 5, 1),
                "date_of_orig_sub": date(2009, 1, 1),
                Col.INSIDER_CIK: "901",
                Col.IS_AMENDMENT: True,
            }
        ),
    )
    out, dropped, orphans = resolve_amendments(df)
    assert dropped == 1
    assert orphans == 1
    assert set(out[Col.ACCESSION]) == {"a1", "a3"}


def test_cross_filed_duplicate_keeps_officer() -> None:
    df = _frame(
        _line(
            **{
                Col.ACCESSION: "fund",
                Col.INSIDER_CIK: "1",
                Col.IS_OFFICER: False,
                Col.IS_TEN_PCT_OWNER: True,
                Col.RAW_TITLE: "",
            }
        ),
        _line(**{Col.ACCESSION: "cfo", Col.INSIDER_CIK: "2"}),
    )
    out, n = drop_cross_filed_duplicates(df)
    assert n == 1
    assert list(out[Col.ACCESSION]) == ["cfo"]


def test_filter_rules(cfg: StudyConfig) -> None:
    df = _frame(
        _line(),
        _line(**{Col.ACCESSION: "grant", Col.TRANS_CODE: "A"}),
        _line(**{Col.ACCESSION: "sale", Col.TRANS_CODE: "S", Col.ACQ_DISP: "D"}),
        _line(**{Col.ACCESSION: "f5", Col.FORM_TYPE: "5"}),
        _line(**{Col.ACCESSION: "late", Col.FILING_DATE: date(2010, 6, 1)}),
        _line(**{Col.ACCESSION: "pref", "security_title": "Series A Preferred"}),
        _line(**{Col.ACCESSION: "plan", Col.IS_10B5_1: True}),
        _line(**{Col.ACCESSION: "zero", Col.PRICE: 0.0}),
        _line(**{Col.ACCESSION: "noticker", Col.TICKER: "NONE"}),
    )
    res = SECTransactionFilter(cfg.filters).apply(df)
    assert list(res.eligible[Col.ACCESSION]) == ["a1"]
    assert list(res.sales[Col.ACCESSION]) == ["sale"]
    assert list(res.plan_10b5_1[Col.ACCESSION]) == ["plan"]
    c = res.exclusion_counts
    assert c["form_type_not_eligible"] == 1
    assert c["filing_lag_too_long"] == 1
    assert c["non_common_security"] == 1
    assert c["non_positive_price_or_shares"] == 1
    assert c["missing_ticker"] == 1


def test_events_aggregate_lines_and_enter_after_filing(cfg: StudyConfig) -> None:
    df = _frame(
        _line(**{"line_id": "1", Col.SHARES: 1000.0, Col.SHARES_OWNED_AFTER: 3000.0}),
        _line(
            **{"line_id": "2", Col.SHARES: 1000.0, Col.PRICE: 12.0, Col.SHARES_OWNED_AFTER: 4000.0}
        ),
    )
    res = SECTransactionFilter(cfg.filters).apply(df)
    ev = build_events(res.eligible, cfg)
    assert len(ev) == 1
    row = ev.iloc[0]
    assert row[Col.SHARES] == 2000
    assert row["vwap"] == 11.0
    assert row["delta_h"] == 1.0  # bought 2000 on top of 2000 held before
    assert row[Col.ENTRY_DATE] == date(2010, 3, 4)  # session after filing, not after trade
    assert row[Col.SPLIT] == "train"
