from __future__ import annotations

from pathlib import Path

import numpy as np

from insider_study.reproducibility import (
    DataManifest,
    environment_fingerprint,
    set_global_seed,
)


def test_seed_is_deterministic() -> None:
    a = set_global_seed(7).normal(size=5)
    b = set_global_seed(7).normal(size=5)
    np.testing.assert_array_equal(a, b)


def test_manifest_detects_changed_data(tmp_path: Path) -> None:
    f = tmp_path / "raw.csv"
    f.write_text("a,b\n1,2\n")
    manifest = DataManifest(tmp_path / "manifest.json")
    manifest.record(f)
    manifest.save()
    assert DataManifest(tmp_path / "manifest.json").verify() == []
    f.write_text("a,b\n1,3\n")
    assert DataManifest(tmp_path / "manifest.json").verify() == [f.as_posix()]


def test_fingerprint_has_core_fields() -> None:
    fp = environment_fingerprint()
    assert fp["python"]
    assert "pkg.pandas" in fp
