"""Regression test for the no-trade band when universe membership shifts.

The original bug: at the first rebalance after a semi-annual reconstitution,
some names from the previous portfolio (e.g. ATI, DDD, JAZZ, TGI) drop out of
the universe entirely. The naive implementation was

    diff = weights_w.reindex(last_weights.index, fill_value=0.0) - last_weights
    small = diff.abs() < nt_band
    weights_w.loc[small.index[small]] = last_weights.loc[small.index[small]]

which tries to assign back to `weights_w` rows that were never in
`weights_w.index`, raising
   KeyError: "['ATI','DDD','JAZZ','TGI'] not in index"

The corrected logic restricts the band to the index intersection.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _apply_no_trade_band(weights_w: pd.Series, last_weights: pd.Series, nt_band: float) -> pd.Series:
    """Mirror of the band logic in backtest/run.py."""
    if last_weights is None or nt_band <= 0:
        return weights_w
    common = weights_w.index.intersection(last_weights.index)
    if len(common) == 0:
        return weights_w
    diff = (weights_w.loc[common] - last_weights.loc[common]).abs()
    freeze = common[diff < nt_band]
    if len(freeze) == 0:
        return weights_w
    weights_w = weights_w.copy()
    weights_w.loc[freeze] = last_weights.loc[freeze]
    total = float(weights_w.sum())
    if total > 0:
        weights_w = weights_w / total
    return weights_w


def test_band_drops_names_that_left_the_universe():
    """Names from prev portfolio that aren't in new index must be sold (not raise)."""
    last_w = pd.Series({"AAPL": 0.20, "MSFT": 0.20, "ATI": 0.15, "DDD": 0.15, "JAZZ": 0.15, "TGI": 0.15})
    new_w  = pd.Series({"AAPL": 0.30, "MSFT": 0.30, "NVDA": 0.20, "AMZN": 0.20})
    out = _apply_no_trade_band(new_w, last_w, nt_band=0.005)
    # No KeyError, output is well-formed
    assert set(out.index) == {"AAPL", "MSFT", "NVDA", "AMZN"}
    assert abs(float(out.sum()) - 1.0) < 1e-9
    # Names that left the universe are gone
    assert not any(t in out.index for t in ("ATI", "DDD", "JAZZ", "TGI"))


def test_band_freezes_only_small_changes_in_overlap():
    """Names whose target moves <0.5% should retain their previous weight."""
    last_w = pd.Series({"AAPL": 0.30, "MSFT": 0.30, "NVDA": 0.40})
    # AAPL barely changed (0.301 vs 0.30), NVDA moved a lot (0.20 vs 0.40)
    new_w  = pd.Series({"AAPL": 0.301, "MSFT": 0.30, "NVDA": 0.20, "AMZN": 0.199})
    out = _apply_no_trade_band(new_w, last_w, nt_band=0.005)
    # AAPL is frozen at the previous level (then renormalized along with everyone)
    pre_norm = pd.Series({"AAPL": 0.30, "MSFT": 0.30, "NVDA": 0.20, "AMZN": 0.199})
    expected = pre_norm / pre_norm.sum()
    pd.testing.assert_series_equal(out.sort_index(), expected.sort_index(), check_names=False)


def test_band_noop_when_disabled():
    last_w = pd.Series({"AAPL": 0.5, "ZZZ": 0.5})
    new_w  = pd.Series({"AAPL": 0.6, "BBB": 0.4})
    out = _apply_no_trade_band(new_w, last_w, nt_band=0.0)
    pd.testing.assert_series_equal(out, new_w)


def test_band_noop_with_no_previous_weights():
    new_w = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    out = _apply_no_trade_band(new_w, last_weights=None, nt_band=0.005)
    pd.testing.assert_series_equal(out, new_w)


def test_band_noop_when_intersection_empty():
    """Full universe turnover (no overlap) must not raise — band is a no-op."""
    last_w = pd.Series({"OLD1": 0.5, "OLD2": 0.5})
    new_w  = pd.Series({"NEW1": 0.5, "NEW2": 0.5})
    out = _apply_no_trade_band(new_w, last_w, nt_band=0.005)
    pd.testing.assert_series_equal(out, new_w)


# -----------------------------------------------------------------------------
# A4 — Score-aware no-trade band (mirror of the orchestrator logic in run.py)
# -----------------------------------------------------------------------------

def _apply_score_band(weights_w: pd.Series, last_weights: pd.Series,
                      cur_scores: pd.Series, last_scores: pd.Series,
                      score_band: float) -> pd.Series:
    if last_weights is None or last_scores is None or score_band <= 0:
        return weights_w
    common = weights_w.index.intersection(last_weights.index).intersection(last_scores.index)
    if len(common) == 0:
        return weights_w
    cur_s  = cur_scores.reindex(common).astype(float)
    prev_s = last_scores.reindex(common).astype(float)
    d_score = (cur_s - prev_s).abs()
    freeze = common[d_score < score_band]
    if len(freeze) == 0:
        return weights_w
    weights_w = weights_w.copy()
    weights_w.loc[freeze] = last_weights.loc[freeze]
    total = float(weights_w.sum())
    if total > 0:
        weights_w = weights_w / total
    return weights_w


def test_score_band_freezes_only_names_with_small_score_change():
    """A name whose composite score barely moves should retain its prior weight,
    while a name whose score moved materially should be allowed to trade."""
    last_w  = pd.Series({"AAPL": 0.30, "MSFT": 0.30, "NVDA": 0.40})
    new_w   = pd.Series({"AAPL": 0.20, "MSFT": 0.50, "NVDA": 0.30})
    last_s  = pd.Series({"AAPL": 1.0, "MSFT": 0.5, "NVDA": -0.2})
    cur_s   = pd.Series({"AAPL": 1.05, "MSFT": 1.5, "NVDA": -0.21})  # AAPL/NVDA tiny moves

    out = _apply_score_band(new_w, last_w, cur_s, last_s, score_band=0.25)

    # AAPL and NVDA must be frozen at their previous weights (then renormalized)
    pre_norm = pd.Series({"AAPL": 0.30, "MSFT": 0.50, "NVDA": 0.40})
    expected = pre_norm / pre_norm.sum()
    pd.testing.assert_series_equal(out.sort_index(), expected.sort_index(), check_names=False)


def test_score_band_lets_meaningful_score_change_trade():
    """When score change exceeds the band, the optimizer's new weight stands."""
    last_w = pd.Series({"AAPL": 0.40, "MSFT": 0.60})
    new_w  = pd.Series({"AAPL": 0.30, "MSFT": 0.70})
    last_s = pd.Series({"AAPL": 0.10, "MSFT": 0.10})
    cur_s  = pd.Series({"AAPL": -0.50, "MSFT": 0.80})  # both moved > 0.25

    out = _apply_score_band(new_w, last_w, cur_s, last_s, score_band=0.25)
    pd.testing.assert_series_equal(out.sort_index(), new_w.sort_index(), check_names=False)


def test_score_band_disabled_is_noop():
    last_w = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    new_w  = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
    last_s = pd.Series({"AAPL": 1.0, "MSFT": 1.0})
    cur_s  = pd.Series({"AAPL": 1.0, "MSFT": 1.0})
    out = _apply_score_band(new_w, last_w, cur_s, last_s, score_band=0.0)
    pd.testing.assert_series_equal(out, new_w)
