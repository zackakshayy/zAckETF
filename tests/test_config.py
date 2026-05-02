"""Schema and loader tests for the canonical alpha.yaml."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml


def _write_tmp(cfg: dict) -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, fh)
    fh.close()
    return fh.name


def test_canonical_loads(cfg):
    assert "_loaded_from" in cfg
    assert cfg["_loaded_from"].endswith("alpha.yaml")
    assert cfg["scoring"]["composite"]["base_weights"]


def test_pillar_weights_sum_to_one(cfg):
    w = cfg["scoring"]["composite"]["base_weights"]
    # Iter-4: pillar set may include lowvol; post_validate sums all keys.
    s = sum(float(v) for v in w.values())
    assert abs(s - 1.0) < 1e-6


def test_invalid_weights_raise():
    from project_alpha.config import get_cfg, reload_cfg
    bad = yaml.safe_load(open(os.environ["PROJECT_ALPHA_CONFIG"]))
    bad["scoring"]["composite"]["base_weights"]["technical"] = 0.5  # break the sum
    p = _write_tmp(bad)
    os.environ["PROJECT_ALPHA_CONFIG"] = p
    try:
        reload_cfg.__wrapped__ if False else None
        with pytest.raises(ValueError):
            from project_alpha.config import reload_cfg as _reload
            _reload()
    finally:
        os.environ["PROJECT_ALPHA_CONFIG"] = str(Path(__file__).resolve().parent.parent / "config" / "alpha.yaml")
        from project_alpha.config import reload_cfg as _reload
        _reload()


def test_missing_required_field_raises():
    from project_alpha.config import reload_cfg
    base = yaml.safe_load(open(os.environ["PROJECT_ALPHA_CONFIG"]))
    base.pop("portfolio")
    p = _write_tmp(base)
    os.environ["PROJECT_ALPHA_CONFIG"] = p
    try:
        with pytest.raises(ValueError):
            reload_cfg()
    finally:
        os.environ["PROJECT_ALPHA_CONFIG"] = str(Path(__file__).resolve().parent.parent / "config" / "alpha.yaml")
        reload_cfg()


def test_start_before_end_validation():
    from project_alpha.config import reload_cfg
    base = yaml.safe_load(open(os.environ["PROJECT_ALPHA_CONFIG"]))
    base["backtest"]["end"] = "2010-01-01"
    p = _write_tmp(base)
    os.environ["PROJECT_ALPHA_CONFIG"] = p
    try:
        with pytest.raises(ValueError):
            reload_cfg()
    finally:
        os.environ["PROJECT_ALPHA_CONFIG"] = str(Path(__file__).resolve().parent.parent / "config" / "alpha.yaml")
        reload_cfg()
