"""Project Alpha CLI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from project_alpha.config import get_cfg, reload_cfg


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="project_alpha", description="Project Alpha — composite ETF backtester")
    ap.add_argument("command", choices=["backtest", "print-config"], help="Action to perform.")
    ap.add_argument("--config", help="Path to alpha.yaml (overrides default search).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    if args.config:
        os.environ["PROJECT_ALPHA_CONFIG"] = str(Path(args.config).resolve())
        reload_cfg()
    cfg = get_cfg()
    _setup_logging(args.log_level)

    if args.command == "print-config":
        print(json.dumps({k: v for k, v in cfg.items() if k != "_loaded_from"}, indent=2, default=str))
        print(f"\n# loaded_from: {cfg.get('_loaded_from')}", file=sys.stderr)
        return 0

    if args.command == "backtest":
        from project_alpha.backtest.run import run
        run()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
