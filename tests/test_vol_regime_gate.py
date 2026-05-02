"""Unit test for the vol-regime gating logic (Tier-2 item 5).

Mirrors the algorithm in backtest/run.py: when VIX percentile ≥ threshold,
shift weight from technical → lowvol, scaled by how far above the threshold
the regime is, capped at `gate_shift_max`, and never draining more than half
of technical's current weight.
"""

from __future__ import annotations

import numpy as np
import pytest


def _apply_vol_gate(weights: dict, vix_pctile: float,
                    gate_threshold: float = 0.75,
                    gate_shift_max: float = 0.10) -> dict:
    """Standalone copy of the gating logic for unit testing."""
    if not (gate_shift_max > 0 and np.isfinite(vix_pctile)
            and vix_pctile >= gate_threshold and 1.0 - gate_threshold > 1e-9):
        return dict(weights)
    shift_strength = (vix_pctile - gate_threshold) / (1.0 - gate_threshold)
    shift_strength = float(np.clip(shift_strength, 0.0, 1.0))
    tech_w = float(weights.get("technical", 0.0))
    shift = min(gate_shift_max * shift_strength, 0.5 * tech_w)
    if shift <= 0:
        return dict(weights)
    out = dict(weights)
    out["technical"] = tech_w - shift
    out["lowvol"]    = float(out.get("lowvol", 0.0)) + shift
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def test_no_shift_below_threshold():
    w = {"technical": 0.30, "fundamental": 0.15, "sentiment": 0.10,
         "lowvol": 0.30, "macro": 0.15}
    out = _apply_vol_gate(w, vix_pctile=0.50)  # below 0.75 threshold
    assert out == w


def test_full_shift_at_top_pctile():
    w = {"technical": 0.30, "fundamental": 0.15, "sentiment": 0.10,
         "lowvol": 0.30, "macro": 0.15}
    out = _apply_vol_gate(w, vix_pctile=1.0, gate_threshold=0.75, gate_shift_max=0.10)
    # At pctile=1.0 the full max shift (0.10) applies (capped at 0.5 * 0.30 = 0.15 — fine).
    assert out["technical"] == pytest.approx(0.20, abs=1e-6)
    assert out["lowvol"]    == pytest.approx(0.40, abs=1e-6)
    # Other pillars unchanged
    assert out["fundamental"] == pytest.approx(0.15, abs=1e-6)
    # Sum still 1
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)


def test_partial_shift_at_mid_pctile():
    w = {"technical": 0.30, "lowvol": 0.30, "macro": 0.15,
         "fundamental": 0.15, "sentiment": 0.10}
    # pctile = 0.875 → halfway between 0.75 and 1.0 → 50% of max shift = 0.05
    out = _apply_vol_gate(w, vix_pctile=0.875, gate_threshold=0.75, gate_shift_max=0.10)
    assert out["technical"] == pytest.approx(0.25, abs=1e-6)
    assert out["lowvol"]    == pytest.approx(0.35, abs=1e-6)


def test_shift_capped_at_half_of_technical():
    """If gate_shift_max would exceed half of technical, shift is capped."""
    w = {"technical": 0.10, "lowvol": 0.30, "macro": 0.15,
         "fundamental": 0.30, "sentiment": 0.15}
    # pctile=1.0, gate_shift_max=0.10 — but tech is only 0.10, so cap at 0.05
    out = _apply_vol_gate(w, vix_pctile=1.0, gate_threshold=0.75, gate_shift_max=0.10)
    assert out["technical"] == pytest.approx(0.05, abs=1e-6)
    assert out["lowvol"]    == pytest.approx(0.35, abs=1e-6)


def test_disabled_when_shift_max_zero():
    w = {"technical": 0.30, "lowvol": 0.30, "macro": 0.15,
         "fundamental": 0.15, "sentiment": 0.10}
    out = _apply_vol_gate(w, vix_pctile=1.0, gate_shift_max=0.0)
    assert out == w


def test_handles_nan_pctile():
    w = {"technical": 0.30, "lowvol": 0.30, "macro": 0.15,
         "fundamental": 0.15, "sentiment": 0.10}
    out = _apply_vol_gate(w, vix_pctile=float("nan"))
    assert out == w
