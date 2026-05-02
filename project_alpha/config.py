"""
project_alpha.config — single canonical loader.

Loading priority:
  1. PROJECT_ALPHA_CONFIG env var (absolute path)
  2. <repo>/config/alpha.yaml relative to this file
  3. <cwd>/config/alpha.yaml

The loader does NOT silently fall back to in-code defaults: if no config
file is found OR the file fails JSON-schema validation, it raises.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml
from jsonschema import Draft202012Validator


_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["paths", "backtest", "universe", "cadence", "scoring", "portfolio"],
    "properties": {
        "paths": {
            "type": "object",
            "required": [
                "data_root", "prices_dir", "fundamentals_dir", "news_dir",
                "macro_dir", "russell_xlsx", "reports_dir", "cache_dir",
            ],
            "additionalProperties": {"type": "string"},
        },
        "backtest": {
            "type": "object",
            "required": ["start", "end", "initial_nav", "benchmark_ticker"],
            "properties": {
                "start": {"type": "string"},
                "end":   {"type": "string"},
                "initial_nav": {"type": "number", "exclusiveMinimum": 0},
                "benchmark_ticker": {"type": "string"},
                "warmup_days": {"type": "integer", "minimum": 0},
            },
        },
        "universe": {
            "type": "object",
            "required": ["source", "reconstitution_dates"],
            "properties": {
                "source": {"type": "string"},
                "reconstitution_dates": {
                    "type": "array",
                    "items": {"type": "string", "pattern": r"^\d{2}-\d{2}$"},
                    "minItems": 1,
                },
                "reconstitution_anchor": {"type": "string"},
                "snapshot_tolerance_days": {"type": "integer", "minimum": 0},
                "min_history_days": {"type": "integer", "minimum": 0},
                "exclude_asset_types": {"type": "array", "items": {"type": "string"}},
                "sector_source": {"type": "string"},
            },
        },
        "cadence": {
            "type": "object",
            "required": [
                "monthly_rebalance",
                "quarterly_fundamental_months",
                "semiannual_reconstitution_months",
            ],
            "properties": {
                "monthly_rebalance": {"type": "boolean"},
                "quarterly_fundamental_months": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 12},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "semiannual_reconstitution_months": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1, "maximum": 12},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
        "scoring": {
            "type": "object",
            "required": ["technical", "fundamental", "sentiment", "macro", "composite"],
            "properties": {
                "composite": {
                    "type": "object",
                    "required": ["base_weights"],
                    "properties": {
                        "base_weights": {
                            "type": "object",
                            "required": ["technical", "fundamental", "sentiment", "macro"],
                            "additionalProperties": {"type": "number"},
                        },
                        "risk_off_tilt": {"type": "object"},
                        "risk_on_tilt":  {"type": "object"},
                        "sector_neutralize": {"type": "boolean"},
                    },
                },
            },
        },
        "modeling": {"type": "object"},
        "portfolio": {
            "type": "object",
            "required": ["max_name_weight", "beta_target"],
            "properties": {
                "long_only": {"type": "boolean"},
                "max_name_weight": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                "min_name_weight": {"type": "number", "minimum": 0, "maximum": 1},
                "max_sector_active": {"type": "number", "minimum": 0, "maximum": 1},
                "adv_consume_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                "adv_lookback_days": {"type": "integer", "minimum": 1},
                "beta_target": {"type": "number"},
                "beta_tol": {"type": "number", "minimum": 0},
                "lambda_risk": {"type": "number", "minimum": 0},
                "lambda_turnover": {"type": "number", "minimum": 0},
                "trade_cost_bps": {"type": "number", "minimum": 0},
                "turnover_no_trade_band": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "reporting": {"type": "object"},
        "logging":   {"type": "object"},
    },
}


def _candidate_paths() -> List[str]:
    here = Path(__file__).resolve().parent
    repo_cfg = here.parent / "config" / "alpha.yaml"
    cwd_cfg = Path.cwd() / "config" / "alpha.yaml"
    paths: List[str] = []
    env = os.environ.get("PROJECT_ALPHA_CONFIG")
    if env:
        paths.append(env)
    paths.append(str(repo_cfg))
    paths.append(str(cwd_cfg))
    return paths


def _resolve_path() -> str:
    for p in _candidate_paths():
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"alpha.yaml not found. Looked in: {_candidate_paths()}. "
        "Set PROJECT_ALPHA_CONFIG to override."
    )


def _validate(cfg: Dict[str, Any]) -> None:
    validator = Draft202012Validator(_SCHEMA)
    errors = sorted(validator.iter_errors(cfg), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors:
            loc = ".".join(str(x) for x in e.absolute_path) or "<root>"
            msgs.append(f"  - {loc}: {e.message}")
        raise ValueError("alpha.yaml failed schema validation:\n" + "\n".join(msgs))


def _post_validate(cfg: Dict[str, Any]) -> None:
    """Cross-field invariants that JSON-schema can't easily express."""
    bw = cfg["scoring"]["composite"]["base_weights"]
    s = sum(float(v) for v in bw.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"composite base_weights must sum to 1.0; got {s:.4f}")
    if cfg["backtest"]["start"] >= cfg["backtest"]["end"]:
        raise ValueError("backtest.start must be < backtest.end")


@lru_cache(maxsize=1)
def get_cfg() -> Dict[str, Any]:
    path = _resolve_path()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    _validate(cfg)
    _post_validate(cfg)
    cfg["_loaded_from"] = path
    return cfg


def reload_cfg() -> Dict[str, Any]:
    get_cfg.cache_clear()  # type: ignore[attr-defined]
    return get_cfg()
