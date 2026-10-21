"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from insider_study.config import load_config
from insider_study.holdout import HoldoutGuard
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
