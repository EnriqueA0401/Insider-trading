from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insider_study.evaluation.stats import (
    FACTORS,
    alpha_regression,
    deflated_sharpe,
    expected_max_sharpe,
    holm,
    max_drawdown,
    pbo_cscv,
)


def _factors(n: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    return pd.DataFrame(
        rng.normal(0, 0.04, size=(n, len(FACTORS))), index=idx, columns=list(FACTORS)
    )


def test_alpha_regression_recovers_alpha_and_betas() -> None:
    rng = np.random.default_rng(0)
    f = _factors(600, rng)
    y = 0.01 + 1.2 * f["mkt_rf"] + 0.5 * f["smb"] + rng.normal(0, 0.005, 600)
    res = alpha_regression(pd.Series(y, index=f.index), f, nw_lags=3)
    assert res.alpha_monthly == pytest.approx(0.01, abs=0.001)
    assert res.betas["mkt_rf"] == pytest.approx(1.2, abs=0.05)
    assert res.p_one_sided < 0.001


def test_pure_factor_exposure_has_no_alpha() -> None:
    """Size/value disguised as alpha must vanish once factors are controlled."""
    rng = np.random.default_rng(1)
    f = _factors(600, rng)
    y = 1.0 * f["smb"] + 0.8 * f["hml"] + rng.normal(0, 0.01, 600)
    res = alpha_regression(pd.Series(y, index=f.index), f, nw_lags=3)
    assert res.p_one_sided > 0.05


def test_holm_matches_textbook() -> None:
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)


def test_expected_max_sharpe_grows_with_trials() -> None:
    assert expected_max_sharpe(0.01, 100) > expected_max_sharpe(0.01, 10) > 0


def test_deflated_sharpe_penalises_many_trials() -> None:
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.01, 0.04, 120))
    few, _ = deflated_sharpe(r, n_trials=2, var_sr_trials=0.01)
    many, _ = deflated_sharpe(r, n_trials=500, var_sr_trials=0.01)
    assert few > many


def test_pbo_high_for_noise_low_for_real_skill() -> None:
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(rng.normal(0, 0.04, size=(240, 20)))
    pbo_noise, _ = pbo_cscv(noise, n_splits=8)
    skilled = noise.copy()
    skilled[0] += 0.03
    pbo_skill, _ = pbo_cscv(skilled, n_splits=8)
    assert pbo_noise > 0.3
    assert pbo_skill < 0.1


def test_max_drawdown() -> None:
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
