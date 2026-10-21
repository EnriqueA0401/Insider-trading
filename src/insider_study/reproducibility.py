"""Seeds, environment fingerprint and data manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

_TRACKED_PACKAGES = ("numpy", "pandas", "pydantic", "exchange-calendars", "mlflow-skinny")


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed stdlib and numpy; return a Generator to be passed explicitly downstream."""
    random.seed(seed)
    np.random.seed(seed)  # legacy global, for third-party code
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def environment_fingerprint() -> dict[str, str | bool | None]:
    versions: dict[str, str | bool | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[f"pkg.{pkg}"] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[f"pkg.{pkg}"] = None
    status = _git("status", "--porcelain")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        **versions,
    }


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class DataManifest:
    """Records the hash of every raw data file so results can be tied to exact inputs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, str] = json.loads(path.read_text()) if path.exists() else {}

    def record(self, file: Path) -> str:
        digest = file_sha256(file)
        self.entries[file.as_posix()] = digest
        return digest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2, sort_keys=True) + "\n")

    def verify(self) -> list[str]:
        """Return the files whose content no longer matches the manifest."""
        return [
            name
            for name, digest in self.entries.items()
            if not Path(name).exists() or file_sha256(Path(name)) != digest
        ]
