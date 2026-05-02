"""Tests for IC-weighted dynamic pillar blending (Phase A1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_panel(scores_per_pillar: dict, n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = [f"T{i:03d}" for i in range(n)]
    df = pd.DataFrame(index=idx)
    df.index.name = "ticker"
    for pillar, gen in scores_per_pillar.items():
        df[f"{pillar}_score"] = gen(rng, n)
    return df


def test_warmup_returns_fixed_weights():
    from project_alpha.scores.dynamic_weights import PillarICTracker

    tracker = PillarICTracker(lookback_events=12, floor=0.05, cap=0.60)
    fixed = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}

    panel = _make_panel({
        "technical":   lambda rng, n: rng.normal(0, 1, n),
        "fundamental": lambda rng, n: rng.normal(0, 1, n),
        "sentiment":   lambda rng, n: rng.normal(0, 1, n),
    })
    prices = pd.Series(100.0, index=panel.index)

    # Run 5 events — under lookback, must return the fixed weights verbatim.
    for k in range(5):
        prices_k = prices * (1 + 0.001 * k)
        w = tracker.update_and_weights(
            asof=pd.Timestamp("2020-06-30") + pd.offsets.MonthEnd(k),
            score_panel=panel,
            prices_at_t=prices_k,
            fixed_weights=fixed,
        )
    assert w == pytest.approx(fixed)


def test_negative_ic_pillar_floored_not_zero():
    """A pillar that consistently anti-predicts forward returns should get the floor weight, not zero."""
    from project_alpha.scores.dynamic_weights import PillarICTracker

    tracker = PillarICTracker(lookback_events=12, floor=0.05, cap=0.60)
    fixed = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}

    rng = np.random.default_rng(42)
    n = 80
    idx = [f"T{i:03d}" for i in range(n)]

    # Synthesize a path: technical predicts well, fundamental anti-predicts, sentiment is noise.
    # Each event: pick scores at t, generate prices at t+1 such that
    #   ret = +0.5*tech_score - 0.5*fund_score + noise
    base_prices = pd.Series(100.0, index=idx)
    prev_prices = base_prices.copy()

    for k in range(13):  # 13 events → 12 realized ICs (≥ lookback)
        tech = rng.normal(0, 1, n)
        fund = rng.normal(0, 1, n)
        sent = rng.normal(0, 1, n)
        panel = pd.DataFrame({
            "technical_score":   tech,
            "fundamental_score": fund,
            "sentiment_score":   sent,
        }, index=idx)

        # Pre-compute next-event return based on these scores
        ret = 0.05 * tech - 0.05 * fund + 0.005 * rng.normal(0, 1, n)
        cur_prices = prev_prices * (1.0 + ret)

        w = tracker.update_and_weights(
            asof=pd.Timestamp("2020-06-30") + pd.offsets.MonthEnd(k),
            score_panel=panel,
            prices_at_t=prev_prices,
            fixed_weights=fixed,
        )
        prev_prices = cur_prices

    # After lookback events of history, we should have switched to dynamic weights.
    # technical must dominate, fundamental hits the floor (no zero).
    assert w["technical"] > w["fundamental"], f"got {w}"
    assert w["technical"] > w["sentiment"], f"got {w}"
    assert w["fundamental"] >= 0.05 * (1.0 - w["macro"]) - 1e-9, f"floor violated: {w}"
    # Macro is fixed at the input weight
    assert w["macro"] == pytest.approx(fixed["macro"])
    # All sum to ~1
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


def test_cap_caps_a_dominant_pillar():
    from project_alpha.scores.dynamic_weights import PillarICTracker

    tracker = PillarICTracker(lookback_events=12, floor=0.05, cap=0.60)
    fixed = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}

    rng = np.random.default_rng(7)
    n = 80
    idx = [f"T{i:03d}" for i in range(n)]
    prev_prices = pd.Series(100.0, index=idx)

    for k in range(13):
        tech = rng.normal(0, 1, n)
        fund = rng.normal(0, 1, n)
        sent = rng.normal(0, 1, n)
        panel = pd.DataFrame({
            "technical_score":   tech,
            "fundamental_score": fund,
            "sentiment_score":   sent,
        }, index=idx)

        # Only technical predicts; fundamental + sentiment pure noise.
        ret = 0.20 * tech + 0.001 * rng.normal(0, 1, n)
        cur_prices = prev_prices * (1.0 + ret)

        w = tracker.update_and_weights(
            asof=pd.Timestamp("2020-06-30") + pd.offsets.MonthEnd(k),
            score_panel=panel,
            prices_at_t=prev_prices,
            fixed_weights=fixed,
        )
        prev_prices = cur_prices

    # Technical should not exceed the cap
    assert w["technical"] <= 0.60 + 1e-6, f"cap violated: {w}"
    # The other two pillars should still receive at least floor
    assert w["fundamental"] >= 0.05 * (1.0 - w["macro"]) - 1e-9, f"floor: {w}"
    assert w["sentiment"]   >= 0.05 * (1.0 - w["macro"]) - 1e-9, f"floor: {w}"


def test_weights_always_sum_to_one():
    from project_alpha.scores.dynamic_weights import PillarICTracker

    tracker = PillarICTracker(lookback_events=4, floor=0.05, cap=0.60)
    fixed = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}

    rng = np.random.default_rng(123)
    n = 60
    idx = [f"T{i:03d}" for i in range(n)]
    prev_prices = pd.Series(100.0, index=idx)

    for k in range(8):
        panel = pd.DataFrame({
            "technical_score":   rng.normal(0, 1, n),
            "fundamental_score": rng.normal(0, 1, n),
            "sentiment_score":   rng.normal(0, 1, n),
        }, index=idx)
        cur_prices = prev_prices * (1.0 + 0.01 * rng.normal(0, 1, n))
        w = tracker.update_and_weights(
            asof=pd.Timestamp("2020-06-30") + pd.offsets.MonthEnd(k),
            score_panel=panel,
            prices_at_t=prev_prices,
            fixed_weights=fixed,
        )
        prev_prices = cur_prices
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6), f"sum != 1 at k={k}: {w}"
        for v in w.values():
            assert v >= 0


def test_ic_history_dataframe_shape():
    from project_alpha.scores.dynamic_weights import PillarICTracker, PILLARS

    tracker = PillarICTracker(lookback_events=4, floor=0.05, cap=0.60)
    fixed = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}

    rng = np.random.default_rng(0)
    n = 50
    idx = [f"T{i:03d}" for i in range(n)]
    prev_prices = pd.Series(100.0, index=idx)
    for k in range(5):
        panel = pd.DataFrame({
            f"{p}_score": rng.normal(0, 1, n) for p in PILLARS
        }, index=idx)
        cur_prices = prev_prices * (1.0 + 0.005 * rng.normal(0, 1, n))
        tracker.update_and_weights(
            asof=pd.Timestamp("2020-06-30") + pd.offsets.MonthEnd(k),
            score_panel=panel,
            prices_at_t=prev_prices,
            fixed_weights=fixed,
        )
        prev_prices = cur_prices

    df = tracker.to_dataframe()
    # 5 events → 4 IC rows (one per realized event after the first)
    assert len(df) == 4
    for p in PILLARS:
        assert p in df.columns
