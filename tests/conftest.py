"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pin config so all tests use the canonical alpha.yaml.
os.environ.setdefault("PROJECT_ALPHA_CONFIG", str(REPO_ROOT / "config" / "alpha.yaml"))


@pytest.fixture(scope="session")
def cfg():
    from project_alpha.config import get_cfg
    return get_cfg()


@pytest.fixture(scope="session")
def data_paths(cfg):
    p = cfg["paths"]
    # Mark tests that need a path as skipped if the path is missing rather
    # than hard-failing — the data lives off-repo and may not be present in CI.
    return {k: Path(v) for k, v in p.items()}


def _path_or_skip(p: Path, what: str):
    if not p.exists():
        pytest.skip(f"{what} not present at {p}")


@pytest.fixture(scope="session")
def russell_panel(data_paths):
    from project_alpha.data import load_russell_panel
    _path_or_skip(data_paths["russell_xlsx"], "R1000.xlsx")
    return load_russell_panel(data_paths["russell_xlsx"])


@pytest.fixture(scope="session")
def macro_panel(data_paths):
    from project_alpha.data import load_macro_panel
    _path_or_skip(data_paths["macro_dir"], "macro_dir")
    return load_macro_panel(data_paths["macro_dir"])
