"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from insider_study.config import load_config
from insider_study.holdout import HoldoutGuard
from insider_study.logging_setup import configure_logging
from insider_study.reproducibility import environment_fingerprint
from insider_study.trials import TrialRegistry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="insider-study")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cfg = sub.add_parser("validate-config", help="validate a config and print its hash")
    p_cfg.add_argument("path", type=Path)

    sub.add_parser("fingerprint", help="print the environment fingerprint")

    p_status = sub.add_parser("status", help="trial count and holdout status")
    p_status.add_argument("path", type=Path)

    p_ing = sub.add_parser("build-data", help="download/parse SEC data and build events")
    p_ing.add_argument("path", type=Path)

    p_run = sub.add_parser("run-study", help="run the study (dev = pre-holdout, final = holdout)")
    p_run.add_argument("path", type=Path)
    p_run.add_argument("--mode", choices=["dev", "final"], required=True)

    p_unseal = sub.add_parser("unseal-holdout", help="one-time holdout unseal for this config")
    p_unseal.add_argument("path", type=Path)
    p_unseal.add_argument("--reason", required=True)

    args = parser.parse_args(argv)

    if args.command == "validate-config":
        cfg = load_config(args.path)
        print(json.dumps({"valid": True, "config_hash": cfg.config_hash()}))
    elif args.command == "fingerprint":
        print(json.dumps(environment_fingerprint(), indent=2))
    elif args.command == "status":
        cfg = load_config(args.path)
        registry = TrialRegistry(cfg.tracking.trial_registry_path, cfg.validation.max_trials)
        guard = HoldoutGuard.from_config(cfg)
        print(
            json.dumps(
                {
                    "config_hash": cfg.config_hash(),
                    "trials_used": registry.count(),
                    "trial_budget": cfg.validation.max_trials,
                    "holdout_unsealed": guard.is_unsealed,
                },
                indent=2,
            )
        )
    elif args.command == "build-data":
        from insider_study.pipeline import stage_events, stage_ingest_sec

        cfg = load_config(args.path)
        stage_ingest_sec(cfg)
        stage_events(cfg)
    elif args.command == "run-study":
        from insider_study.reporting.tables import render
        from insider_study.study import run

        configure_logging()
        cfg = load_config(args.path)
        print(run(cfg, args.mode))
        print(render(args.mode))
    elif args.command == "unseal-holdout":
        cfg = load_config(args.path)
        print(HoldoutGuard.unseal(cfg, args.reason))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
