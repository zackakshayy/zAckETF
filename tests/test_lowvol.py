"""Tests for the low-vol pillar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _eod_frame(n_days: int = 400, daily_vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-12-31", periods=n_days)
    rets = rng.normal(0.0003, daily_vol, n_days)
    px = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame({
        "date": dates,
        "adjusted_close": px,
        "close": px,
        "volume": rng.integers(100_000, 1_000_000, n_days),
    })


def test_lowvol_features_returns_real_numbers():
    from project_alpha.scores.lowvol import compute_lowvol_features
    eod = _eod_frame(seed=1)
    feats = compute_lowvol_features(eod, asof=eod["date"].iloc[-1])
    assert np.isfinite(feats["vol_252d"])
    assert np.isfinite(feats["vol_stability"])
    # Beta is NaN without bench_returns
    assert np.isnan(feats["beta_252d"])


def test_lowvol_beta_with_bench():
    from project_alpha.scores.lowvol import compute_lowvol_features
    eod = _eod_frame(seed=2)
    bench_dates = eod["date"].iloc[1:]  # match daily-return alignment
    bench_rets = pd.Series(np.random.default_rng(99).normal(0.0003, 0.008, len(bench_dates)),
                           index=pd.DatetimeIndex(bench_dates.values))
    feats = compute_lowvol_features(eod, asof=eod["date"].iloc[-1], bench_returns=bench_rets)
    assert np.isfinite(feats["beta_252d"])


def test_lowvol_panel_score_inverts_vol():
    """Higher vol/beta/instability → lower lowvol_score."""
    from project_alpha.scores.lowvol import compute_lowvol_panel
    rows = [
        {"ticker": "LOW",  "vol_252d": 0.10, "beta_252d": 0.6, "vol_stability": 0.001},
        {"ticker": "MID",  "vol_252d": 0.20, "beta_252d": 1.0, "vol_stability": 0.005},
        {"ticker": "HIGH", "vol_252d": 0.40, "beta_252d": 1.4, "vol_stability": 0.012},
        {"ticker": "TINY", "vol_252d": 0.08, "beta_252d": 0.5, "vol_stability": 0.0008},
    ]
    sector_map = {"LOW": "Utilities", "MID": "Utilities", "HIGH": "Utilities", "TINY": "Utilities"}
    panel = compute_lowvol_panel(rows, sector_map=sector_map)
    assert panel.loc["LOW", "lowvol_score"]  > panel.loc["HIGH", "lowvol_score"]
    assert panel.loc["TINY", "lowvol_score"] > panel.loc["MID",  "lowvol_score"]


def test_lowvol_panel_handles_empty():
    from project_alpha.scores.lowvol import compute_lowvol_panel
    out = compute_lowvol_panel([], sector_map={})
    assert "lowvol_score" in out.columns
    assert len(out) == 0
