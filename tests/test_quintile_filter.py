"""Tests for the top-quintile within-sector pre-filter (Phase A3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _build(n_per_sector: dict, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for sec, n in n_per_sector.items():
        for i in range(n):
            rows.append({
                "ticker": f"{sec[:3].upper()}{i:03d}",
                "sector": sec,
                "score":  float(rng.normal(0, 1)),
            })
    df = pd.DataFrame(rows).set_index("ticker")
    return df["score"], df["sector"]


def test_quintile_filter_keeps_top_20pct_per_sector():
    from project_alpha.portfolio.construct import filter_top_quintile_by_sector

    scores, sector = _build({"IT": 100, "Financials": 50, "Energy": 25}, seed=1)
    keep = filter_top_quintile_by_sector(scores, sector, quintile=0.20, min_per_sector=3)

    kept_sector = sector.reindex(keep)
    counts = kept_sector.value_counts()
    # 20% of 100=20, 20% of 50=10, 20% of 25=5 (still >= min_per_sector=3)
    assert counts["IT"] == 20
    assert counts["Financials"] == 10
    assert counts["Energy"] == 5


def test_quintile_filter_enforces_min_per_sector_floor():
    from project_alpha.portfolio.construct import filter_top_quintile_by_sector

    # Tiny sector with only 5 names — 20% rounds to 1, but min_per_sector=3 should win.
    scores, sector = _build({"IT": 100, "Tiny": 5}, seed=2)
    keep = filter_top_quintile_by_sector(scores, sector, quintile=0.20, min_per_sector=3)

    kept_sector = sector.reindex(keep)
    counts = kept_sector.value_counts()
    assert counts["Tiny"] == 3
    assert counts["IT"] == 20


def test_quintile_filter_keeps_highest_scores():
    from project_alpha.portfolio.construct import filter_top_quintile_by_sector

    # Construct deterministic scores
    scores = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0,
                        "F": -1.0, "G": -2.0, "H": -3.0, "I": -4.0, "J": -5.0})
    sector = pd.Series({k: "IT" for k in scores.index})
    keep = filter_top_quintile_by_sector(scores, sector, quintile=0.20, min_per_sector=2)
    # 20% of 10 = 2 → top 2 by score: A, B
    assert set(keep) == {"A", "B"}


def test_quintile_filter_handles_empty_sector():
    """Sector in target list with no members must be skipped without raising."""
    from project_alpha.portfolio.construct import filter_top_quintile_by_sector

    scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0})
    sector = pd.Series({"A": "IT", "B": "IT", "C": "IT"})
    keep = filter_top_quintile_by_sector(scores, sector, quintile=0.20, min_per_sector=2)
    assert len(keep) == 2


def test_quintile_filter_drops_NaN_scores():
    from project_alpha.portfolio.construct import filter_top_quintile_by_sector

    scores = pd.Series({"A": 1.0, "B": np.nan, "C": 3.0, "D": 2.0, "E": np.nan})
    sector = pd.Series({k: "IT" for k in scores.index})
    keep = filter_top_quintile_by_sector(scores, sector, quintile=0.50, min_per_sector=1)
    # 3 valid scores, 50% → 2; should be C (3.0) and D (2.0)
    assert set(keep) == {"C", "D"}
