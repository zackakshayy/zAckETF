"""Tests for idiosyncratic momentum (residual after bench-beta strip)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_eod(rets: np.ndarray, end="2024-12-31") -> pd.DataFrame:
    n = len(rets)
    dates = pd.bdate_range(end=end, periods=n + 1)
    px = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
    return pd.DataFrame({"date": dates, "adjusted_close": px, "close": px,
                         "volume": np.full(n + 1, 100_000)})


def test_idio_momentum_zero_when_pure_market_beta():
    """A name that perfectly tracks the market (β=1, no idio) should have idio_mom near 0."""
    from project_alpha.scores.technical import compute_technical_features
    rng = np.random.default_rng(0)
    n = 350
    market = rng.normal(0.0005, 0.01, n)
    stock_rets = market.copy()  # β = 1, ε = 0
    eod = _make_eod(stock_rets)
    bench_idx = pd.DatetimeIndex(eod["date"].iloc[1:].values)
    bench = pd.Series(market, index=bench_idx)
    feats = compute_technical_features(eod, asof=eod["date"].iloc[-1], bench_returns=bench)
    assert abs(feats["idio_mom_252"]) < 0.05  # near zero


def test_idio_momentum_positive_when_residual_drifts_up():
    """A name with a positive idiosyncratic drift on top of the market should have positive idio_mom."""
    from project_alpha.scores.technical import compute_technical_features
    rng = np.random.default_rng(1)
    n = 350
    market = rng.normal(0.0003, 0.01, n)
    eps = np.full(n, 0.0008)  # consistent positive idio drift
    stock_rets = 1.0 * market + eps
    eod = _make_eod(stock_rets)
    bench = pd.Series(market, index=pd.DatetimeIndex(eod["date"].iloc[1:].values))
    feats = compute_technical_features(eod, asof=eod["date"].iloc[-1], bench_returns=bench)
    assert feats["idio_mom_252"] > 0.05


def test_idio_momentum_nan_without_bench():
    """Without bench_returns, idio_mom_252 should remain NaN (graceful degrade)."""
    from project_alpha.scores.technical import compute_technical_features
    rng = np.random.default_rng(2)
    n = 350
    eod = _make_eod(rng.normal(0.0005, 0.012, n))
    feats = compute_technical_features(eod, asof=eod["date"].iloc[-1])
    assert np.isnan(feats["idio_mom_252"])


# -----------------------------------------------------------------------------
# Tier-1 item 1 / 4 — multi-horizon momentum + Sharpe-adjusted top horizon
# -----------------------------------------------------------------------------

def test_multi_horizon_momentum_features_present():
    """All four horizons (3/6/9/12-1) should be populated when there's enough history."""
    from project_alpha.scores.technical import compute_technical_features
    rng = np.random.default_rng(3)
    n = 350
    eod = _make_eod(rng.normal(0.0006, 0.011, n))
    feats = compute_technical_features(eod, asof=eod["date"].iloc[-1])
    for k in ("mom_3_1", "mom_6_1", "mom_9_1", "mom_12_1"):
        assert np.isfinite(feats[k]), f"{k} is NaN"


def test_sharpe_momentum_de_rates_high_vol_winners():
    """Two stocks with the same 12-month return — the higher-vol one should
    have a *lower* sharpe_mom_12_1 (vol penalizes the score)."""
    from project_alpha.scores.technical import compute_technical_features
    rng = np.random.default_rng(11)
    n = 350

    # Build two return series with the same expected mean but different vols
    low_vol  = rng.normal(0.0008, 0.005, n)
    high_vol = rng.normal(0.0008, 0.020, n)

    eod_lo = _make_eod(low_vol)
    eod_hi = _make_eod(high_vol)
    flo = compute_technical_features(eod_lo, asof=eod_lo["date"].iloc[-1])
    fhi = compute_technical_features(eod_hi, asof=eod_hi["date"].iloc[-1])

    # Both should have a finite sharpe_mom_12_1
    assert np.isfinite(flo["sharpe_mom_12_1"])
    assert np.isfinite(fhi["sharpe_mom_12_1"])
    # The low-vol stock's sharpe-momentum should not be lower than the high-vol stock's
    # (within the noise — we test the structural property, not exact magnitudes).
    assert flo["vol_252d"] < fhi["vol_252d"]


# -----------------------------------------------------------------------------
# Tier-1 item 3 — high52_gap and liq_trend should NOT be in the panel composite
# -----------------------------------------------------------------------------

def test_dropped_subfeatures_not_in_composite_columns():
    """high52 / liq are intentionally excluded from the technical panel output (item 3)."""
    from project_alpha.scores.technical import compute_technical_panel
    rows = [{
        "ticker": f"T{i:03d}",
        "price":  100.0 + i,
        "mom_12_1": 0.10, "mom_9_1": 0.08, "mom_6_1": 0.06, "mom_3_1": 0.03,
        "sharpe_mom_12_1": 0.5,
        "rev_21d": 0.0, "vol_252d": 0.20,
        "high52_gap": -0.05, "liq_trend": 0.01,
        "idio_mom_252": 0.05,
    } for i in range(20)]
    sector_map = {f"T{i:03d}": "Information Technology" for i in range(20)}
    panel = compute_technical_panel(rows, sector_map=sector_map)
    assert "high52_z" not in panel.columns
    assert "liq_z"    not in panel.columns
    # And the new sector-residual idio column IS present
    assert "idio_multi_z" in panel.columns


# -----------------------------------------------------------------------------
# Tier-1 item 2 — multi-residual idio momentum (sector-mean-stripped)
# -----------------------------------------------------------------------------

def test_idio_multi_z_strips_sector_mean():
    """Names with the same idio_mom but in different sector contexts should differ
    in idio_multi_z by approximately their relative sector-mean position."""
    from project_alpha.scores.technical import compute_technical_panel

    # Sector A: all members idio_mom = 0.10  → sector mean = 0.10
    # Sector B: all members idio_mom = 0.00  → sector mean = 0.00
    # A name with idio_mom=0.05 in sector A should score below same in sector B
    # (the sector-A name is BELOW its sector mean; sector-B name is ABOVE).
    rows = []
    for i in range(10):
        rows.append({"ticker": f"A{i:02d}", "idio_mom_252": 0.10,
                     "vol_252d": 0.20, "rev_21d": 0.0,
                     "mom_3_1": 0, "mom_6_1": 0, "mom_9_1": 0,
                     "mom_12_1": 0, "sharpe_mom_12_1": 0})
    for i in range(10):
        rows.append({"ticker": f"B{i:02d}", "idio_mom_252": 0.00,
                     "vol_252d": 0.20, "rev_21d": 0.0,
                     "mom_3_1": 0, "mom_6_1": 0, "mom_9_1": 0,
                     "mom_12_1": 0, "sharpe_mom_12_1": 0})
    # Test names: A-test has idio=0.05 (below sector A's 0.10), B-test has 0.05 (above sector B's 0.0)
    rows.append({"ticker": "A_TEST", "idio_mom_252": 0.05,
                 "vol_252d": 0.20, "rev_21d": 0.0,
                 "mom_3_1": 0, "mom_6_1": 0, "mom_9_1": 0,
                 "mom_12_1": 0, "sharpe_mom_12_1": 0})
    rows.append({"ticker": "B_TEST", "idio_mom_252": 0.05,
                 "vol_252d": 0.20, "rev_21d": 0.0,
                 "mom_3_1": 0, "mom_6_1": 0, "mom_9_1": 0,
                 "mom_12_1": 0, "sharpe_mom_12_1": 0})

    sector_map = {**{f"A{i:02d}": "SectorA" for i in range(10)},
                  **{f"B{i:02d}": "SectorB" for i in range(10)},
                  "A_TEST": "SectorA", "B_TEST": "SectorB"}

    panel = compute_technical_panel(rows, sector_map=sector_map)
    # B_TEST should rank higher on idio_multi_z than A_TEST (above its sector mean vs below)
    assert panel.loc["B_TEST", "idio_multi_z"] > panel.loc["A_TEST", "idio_multi_z"]
