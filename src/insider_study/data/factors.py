"""Ken French Data Library adapter: FF5 + momentum factors and NYSE ME breakpoints."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FILES = {
    ("ff5", "monthly"): "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    ("ff5", "daily"): "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    ("umd", "monthly"): "F-F_Momentum_Factor_CSV.zip",
    ("umd", "daily"): "F-F_Momentum_Factor_daily_CSV.zip",
    ("me_breakpoints", "monthly"): "ME_Breakpoints_CSV.zip",
}


def _first_block(text: str, n_date_digits: int) -> pd.DataFrame:
    """Parse the first data block whose rows start with an n-digit date."""
    rows: list[list[str]] = []
    header: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if rows:
                break
            continue
        parts = [p.strip() for p in line.split(",")]
        if parts[0].isdigit() and len(parts[0]) == n_date_digits:
            rows.append(parts)
        elif rows:
            break
        elif len(parts) > 1 and parts[0] == "":
            header = parts[1:]
    if not rows:
        raise ValueError("no data block found")
    frame = pd.DataFrame([r[1:] for r in rows], index=[r[0] for r in rows]).astype(float)
    if header is not None and len(header) == frame.shape[1]:
        frame.columns = [h.lower().replace("-", "_") for h in header]
    return frame


@dataclass
class FrenchFactorProvider:
    cache_dir: Path

    def _text(self, fname: str) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / fname
        if not path.exists():
            resp = requests.get(f"{FRENCH_URL}/{fname}", timeout=120)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        with zipfile.ZipFile(path) as zf:
            name = zf.namelist()[0]
            return io.TextIOWrapper(zf.open(name), encoding="latin-1").read()

    def get_factors(self, start: date, end: date, frequency: str) -> pd.DataFrame:
        digits = 6 if frequency == "monthly" else 8
        fmt = "%Y%m" if frequency == "monthly" else "%Y%m%d"
        ff5 = _first_block(self._text(FILES[("ff5", frequency)]), digits)
        umd = _first_block(self._text(FILES[("umd", frequency)]), digits)
        umd.columns = ["umd"]
        df = ff5.join(umd, how="inner") / 100.0
        df.index = pd.to_datetime(df.index, format=fmt)
        if frequency == "monthly":
            df.index = df.index.to_period("M").to_timestamp("M")
        df = df.rename(columns={"mkt_rf": "mkt_rf"})
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask, ["mkt_rf", "smb", "hml", "rmw", "cma", "umd", "rf"]]

    def get_size_breakpoints(self, start: date, end: date) -> pd.DataFrame:
        """Monthly NYSE ME percentiles (USD millions): columns p5..p100."""
        raw = _first_block(self._text(FILES[("me_breakpoints", "monthly")]), 6)
        bp = raw.iloc[:, 1:21]
        bp.columns = [f"p{5 * (i + 1)}" for i in range(20)]
        bp.index = pd.to_datetime(bp.index, format="%Y%m").to_period("M").to_timestamp("M")
        mask = (bp.index >= pd.Timestamp(start)) & (bp.index <= pd.Timestamp(end))
        return bp.loc[mask]
