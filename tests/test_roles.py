from __future__ import annotations

import pandas as pd
import pytest

from insider_study.data.roles import map_roles
from insider_study.domain import Col, InsiderRole


def _df(
    title: str, officer: bool = True, director: bool = False, ten: bool = False
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            Col.RAW_TITLE: [title],
            Col.IS_OFFICER: [officer],
            Col.IS_DIRECTOR: [director],
            Col.IS_TEN_PCT_OWNER: [ten],
        }
    )


@pytest.mark.parametrize(
    "title",
    [
        "CFO",
        "EVP/CFO",
        "SVP-CFO",
        "Exec. VP & Chief Financial Officer",
        "Senior Vice President, CFO and Treasurer",
        "Interim Chief Financial Officer",
        "Principal Financial Officer",
        "C.F.O.",
        "Chief Financial & Administrative Officer",
        "Chief Finance Officer",
        "Chief Fin. Officer",
    ],
)
def test_cfo_titles(title: str) -> None:
    out = map_roles(_df(title))
    assert out["is_cfo"].iloc[0]
    assert out[Col.ROLE].iloc[0] == InsiderRole.CFO


@pytest.mark.parametrize(
    "title",
    [
        "Deputy CFO",
        "Assistant CFO",
        "Treasurer",
        "VP Finance",
        "Controller",
        "Chief Accounting Officer",
        "Chief Operating Officer",
        "Vice President",
        "Chief Executive Officer",
    ],
)
def test_non_cfo_titles(title: str) -> None:
    assert not map_roles(_df(title))["is_cfo"].iloc[0]


def test_cao_counts_only_in_broad_definition() -> None:
    out = map_roles(_df("Principal Accounting Officer"))
    assert not out["is_cfo"].iloc[0]
    assert out["is_cfo_broad"].iloc[0]


@pytest.mark.parametrize(
    "title",
    ["CEO", "President & CEO", "Chief Executive Officer", "Co-CEO", "Chief Exec Officer, Director"],
)
def test_ceo_titles(title: str) -> None:
    out = map_roles(_df(title))
    assert out["is_ceo"].iloc[0]
    assert out[Col.ROLE].iloc[0] == InsiderRole.CEO


def test_dual_role_keeps_both_flags() -> None:
    out = map_roles(_df("CEO and CFO"))
    assert out["is_ceo"].iloc[0]
    assert out["is_cfo"].iloc[0]
    assert out["is_dual_ceo_cfo"].iloc[0]
    assert out[Col.ROLE].iloc[0] == InsiderRole.CFO


def test_director_only_and_ten_pct() -> None:
    d = map_roles(_df("", officer=False, director=True))
    assert d[Col.ROLE].iloc[0] == InsiderRole.DIRECTOR
    assert d["is_director_only"].iloc[0]
    t = map_roles(_df("", officer=False, ten=True))
    assert t[Col.ROLE].iloc[0] == InsiderRole.TEN_PCT_OWNER
    assert t["is_ten_pct_only"].iloc[0]


def test_officer_director_is_not_director_only() -> None:
    out = map_roles(_df("CFO", officer=True, director=True))
    assert not out["is_director_only"].iloc[0]
