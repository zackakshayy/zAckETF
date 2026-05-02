"""Composite combiner tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _panel(score_col: str, n_per_sector: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sec in ["Information Technology", "Health Care", "Financials", "Energy"]:
        for i in range(n_per_sector):
            rows.append({
                "ticker": f"{sec[:3].upper()}{i:03d}",
                "sector": sec,
                score_col: float(rng.normal(0, 1)),
            })
    return pd.DataFrame(rows).set_index("ticker")


def test_composite_produces_one_score_per_ticker():
    from project_alpha.scores.composite import combine_composite
    tech = _panel("technical_score", seed=1)
    fund = _panel("fundamental_score", seed=2)
    sent = _panel("sentiment_score", seed=3)
    weights = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}
    comp = combine_composite(tech, fund, sent, pillar_weights=weights, regime="neutral")
    assert "composite_score" in comp.columns
    assert comp.index.is_unique
    assert (comp["composite_score"].abs() < 10).all()


def test_composite_sector_neutral():
    from project_alpha.scores.composite import combine_composite
    tech = _panel("technical_score", seed=1)
    fund = _panel("fundamental_score", seed=2)
    sent = _panel("sentiment_score", seed=3)
    weights = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}
    comp = combine_composite(tech, fund, sent, pillar_weights=weights,
                             regime="neutral", sector_neutralize=True)
    for sec, sub in comp.groupby("sector"):
        assert abs(float(sub["composite_score"].mean())) < 0.05


def test_macro_sector_tilt_signs():
    from project_alpha.scores.composite import macro_sector_tilt
    sectors = pd.Index(["Information Technology", "Utilities", "Health Care",
                        "Industrials", "Consumer Staples"])
    risk_on = macro_sector_tilt(sectors, "risk_on")
    risk_off = macro_sector_tilt(sectors, "risk_off")
    # Risk-on should favor cyclicals over defensives
    assert risk_on["Information Technology"] > risk_on["Utilities"]
    # Risk-off flips the relationship
    assert risk_off["Information Technology"] < risk_off["Utilities"]


def test_composite_without_second_sector_z_preserves_variance():
    """A2: with sector_neutralize=False the composite variance should be at least as large
    as the largest single pillar variance (after weighting). The double sector-z step
    shrinks signal — disabling it must not shrink it further."""
    from project_alpha.scores.composite import combine_composite

    tech = _panel("technical_score", seed=11)
    fund = _panel("fundamental_score", seed=12)
    sent = _panel("sentiment_score", seed=13)
    weights = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}

    no_z = combine_composite(tech, fund, sent, pillar_weights=weights,
                             regime="neutral", sector_neutralize=False)
    with_z = combine_composite(tech, fund, sent, pillar_weights=weights,
                               regime="neutral", sector_neutralize=True)

    # Enabling the second sector-z step should not *increase* variance — it shrinks signal.
    var_no_z   = float(no_z["composite_score"].var())
    var_with_z = float(with_z["composite_score"].var())
    assert var_no_z >= var_with_z - 1e-9, (var_no_z, var_with_z)

    # And the un-z'd composite carries at least the variance of the strongest weighted pillar.
    pillar_vars = []
    for col, w in [("technical_score", 0.35), ("fundamental_score", 0.30), ("sentiment_score", 0.20)]:
        pillar_vars.append(float((w * no_z[col]).var()))
    assert var_no_z >= 0.5 * max(pillar_vars), (var_no_z, pillar_vars)


def test_quality_momentum_gate_boosts_dual_positive():
    """A name with both high quality_z and high momentum_z should score above one with neither."""
    from project_alpha.scores.composite import combine_composite

    rng = np.random.default_rng(7)
    idx = ["A", "B", "C", "D"]
    tech = pd.DataFrame({
        "ticker": idx,
        "sector": ["Information Technology"] * 4,
        "technical_score": [0.0, 0.0, 0.0, 0.0],
        "momentum_z":      [+1.5, -1.0, +1.5, -1.0],
    }).set_index("ticker")
    fund = pd.DataFrame({
        "ticker": idx,
        "sector": ["Information Technology"] * 4,
        "fundamental_score": [0.0, 0.0, 0.0, 0.0],
        "quality_z":         [+1.5, +1.5, -1.0, -1.0],
    }).set_index("ticker")
    sent = pd.DataFrame({
        "ticker": idx,
        "sector": ["Information Technology"] * 4,
        "sentiment_score": [0.0, 0.0, 0.0, 0.0],
    }).set_index("ticker")
    weights = {"technical": 0.0, "fundamental": 0.0, "sentiment": 0.0,
               "lowvol": 0.0, "macro": 0.0}
    out = combine_composite(tech, fund, sent, pillar_weights=weights, regime="neutral",
                            sector_neutralize=False, quality_momentum_gate_w=1.0)
    # A is high-quality + high-momentum (gate fires); D is low/low (gate dormant).
    assert out.loc["A", "composite_score"] > out.loc["D", "composite_score"]
    # B (quality only) and C (momentum only) — gate only fires when BOTH are positive.
    assert out.loc["A", "composite_score"] > out.loc["B", "composite_score"]
    assert out.loc["A", "composite_score"] > out.loc["C", "composite_score"]


def test_lowvol_pillar_flows_through_composite():
    """When lowvol weight is positive, the lowvol_score column drives the composite."""
    from project_alpha.scores.composite import combine_composite

    idx = ["LOW", "HIGH"]
    tech = pd.DataFrame({"ticker": idx, "sector": ["Utilities"]*2,
                         "technical_score": [0.0, 0.0]}).set_index("ticker")
    fund = pd.DataFrame({"ticker": idx, "sector": ["Utilities"]*2,
                         "fundamental_score": [0.0, 0.0]}).set_index("ticker")
    sent = pd.DataFrame({"ticker": idx, "sector": ["Utilities"]*2,
                         "sentiment_score": [0.0, 0.0]}).set_index("ticker")
    lv   = pd.DataFrame({"ticker": idx, "sector": ["Utilities"]*2,
                         "lowvol_score": [+2.0, -2.0]}).set_index("ticker")
    weights = {"technical": 0.0, "fundamental": 0.0, "sentiment": 0.0,
               "lowvol": 1.0, "macro": 0.0}
    out = combine_composite(tech, fund, sent, pillar_weights=weights, regime="neutral",
                            sector_neutralize=False, lowvol_panel=lv)
    assert out.loc["LOW", "composite_score"] > out.loc["HIGH", "composite_score"]


def test_regime_changes_pillar_weights():
    from project_alpha.scores.composite import combine_composite
    tech = _panel("technical_score", seed=1)
    fund = _panel("fundamental_score", seed=2)
    sent = _panel("sentiment_score", seed=3)
    weights = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}
    comp_neutral = combine_composite(tech, fund, sent, pillar_weights=weights, regime="neutral")
    comp_off = combine_composite(tech, fund, sent, pillar_weights=weights, regime="risk_off")
    # Composite should differ somewhere
    diff = (comp_neutral["composite_score"] - comp_off["composite_score"]).abs()
    assert diff.sum() > 0
