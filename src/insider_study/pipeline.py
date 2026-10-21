"""Pipeline stages. Each stage reads/writes parquet under data/ so runs are resumable."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from insider_study.config import StudyConfig
from insider_study.data.events import build_events
from insider_study.data.filters import SECTransactionFilter
from insider_study.data.roles import map_roles
from insider_study.data.sec import SECForm345Ingestor, parse_all
from insider_study.logging_setup import get_logger

log = get_logger(__name__)


def stage_ingest_sec(cfg: StudyConfig) -> Path:
    ing = SECForm345Ingestor(
        cfg.data.raw_dir / "sec",
        cfg.data.sec_user_agent,
        cfg.data.sec_max_requests_per_second,
        Path("data/manifest_sec.json"),
    )
    paths = ing.fetch(range(cfg.data.first_year, cfg.data.last_year + 1))
    lines_path, _ = parse_all(paths, cfg.data.interim_dir)
    return lines_path


def stage_events(cfg: StudyConfig) -> Path:
    lines = pd.read_parquet(cfg.data.interim_dir / "open_market_lines.parquet")
    lines = map_roles(lines)
    result = SECTransactionFilter(cfg.filters).apply(lines)
    out = cfg.data.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    events = build_events(result.eligible, cfg)
    events.to_parquet(out / "events.parquet", index=False)
    build_events(result.plan_10b5_1, cfg).to_parquet(out / "events_10b5_1.parquet", index=False)
    result.sales.to_parquet(out / "sales_lines.parquet", index=False)
    # All open-market trades (P and S), for the routine/opportunistic history.
    om = lines.loc[lines["trans_code"].isin(["P", "S"])]
    om.to_parquet(out / "open_market_history.parquet", index=False)
    counts = dict(result.exclusion_counts) | {
        "input_lines": len(lines),
        "eligible_lines": len(result.eligible),
        "eligible_events": len(events),
        "sales_lines": len(result.sales),
    }
    (out / "filter_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    log.info("events_built", **counts)
    return out / "events.parquet"
