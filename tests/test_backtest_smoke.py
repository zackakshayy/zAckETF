"""Smoke tests for the full backtest orchestrator.

Uses a short config window so the test runs in well under a minute and verifies
the end-to-end pipeline can:
  - load data,
  - emit rebalance events,
  - compute composite scores,
  - run the optimizer,
  - produce a non-empty NAV curve and KPI dict.

Heavy assertions are minimal — the goal is to catch wiring breakage, not to
validate strategy quality.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture(scope="module")
def short_window_cfg(cfg, tmp_path_factory):
    """Build an alpha.yaml override for a 6-month window with 100 names."""
    cfg = json.loads(json.dumps({k: v for k, v in cfg.items() if k != "_loaded_from"}, default=str))
    cfg["backtest"]["start"] = "2023-07-01"
    cfg["backtest"]["end"] = "2024-01-31"
    cfg["paths"]["reports_dir"] = str(tmp_path_factory.mktemp("reports"))
    cfg["paths"]["cache_dir"] = str(tmp_path_factory.mktemp("cache"))

    p = tmp_path_factory.mktemp("cfg") / "alpha.yaml"
    yaml.safe_dump(cfg, open(p, "w"))
    return str(p)


@pytest.mark.slow
def test_backtest_runs_short_window(short_window_cfg, monkeypatch):
    """End-to-end smoke: short window, ~7 monthly rebalances."""
    monkeypatch.setenv("PROJECT_ALPHA_CONFIG", short_window_cfg)
    from project_alpha.config import reload_cfg
    reload_cfg()

    # Cap the universe size to keep the smoke test fast.
    from project_alpha.data import universe as _uni
    real_universe_for_date = _uni.universe_for_date

    def small_universe_for_date(*args, **kwargs):
        df = real_universe_for_date(*args, **kwargs)
        return df.head(60).reset_index(drop=True)

    monkeypatch.setattr(_uni, "universe_for_date", small_universe_for_date)
    from project_alpha.backtest import run as run_module
    monkeypatch.setattr(run_module, "universe_for_date", small_universe_for_date)

    result = run_module.run()
    assert "kpis" in result
    assert not result["daily_returns"].empty
    assert "cagr" in result["kpis"]
    assert "information_ratio" in result["kpis"]

    # Reports should exist
    out = Path(reload_cfg()["paths"]["reports_dir"])
    assert (out / "alpha_daily_returns.csv").exists()
    assert (out / "alpha_nav.csv").exists()
    assert (out / "regime_log.csv").exists()
    assert (out / "kpis.json").exists()
