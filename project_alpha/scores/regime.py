"""
Macro regime classifier.

Classifies the macro environment at a given as-of date into one of three
regimes — `risk_on`, `neutral`, `risk_off` — based on a small basket of
indicators that are easy to source from FRED:

  VIX level vs its 5y rolling history
  10Y-2Y curve slope (sign + 3M change)
  Fed funds 3M change
  Unemployment 3M change

Regime is a simple sum of signed signals, then bucketed. The thresholds
are deliberately conservative: small moves stay neutral.

The regime is consumed by `scores.composite` to tilt the four-pillar
weights (e.g., risk-off → up-weight quality + macro pillar). It does
NOT score individual stocks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


REGIMES = ("risk_off", "neutral", "risk_on")


def _percentile(series: pd.Series, asof: pd.Timestamp, lookback_years: int) -> float:
    """Percentile of the latest <= asof value vs. the trailing window."""
    if series is None or series.empty:
        return float("nan")
    asof = pd.Timestamp(asof).normalize()
    sub = series.loc[series.index <= asof]
    if sub.empty:
        return float("nan")
    last = float(sub.iloc[-1])
    win = sub.tail(int(lookback_years * 252))
    if win.empty:
        return float("nan")
    return float((win <= last).mean())


def classify_regime(
    macro_panel: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    lookback_years: int = 5,
) -> Tuple[str, Dict[str, float]]:
    """Classify regime at `asof`. Returns (label, signal_dict)."""
    asof = pd.Timestamp(asof).normalize()
    sub = macro_panel.loc[macro_panel.index <= asof].ffill()
    if sub.empty:
        return "neutral", {}

    last = sub.iloc[-1]

    vix_pctile = _percentile(macro_panel.get("VIXCLS", pd.Series(dtype=float)), asof, lookback_years)
    slope_now = float(last.get("T10Y2Y", np.nan))
    slope_chg = float(macro_panel["T10Y2Y"].diff(63).loc[macro_panel.index <= asof].ffill().iloc[-1]) \
        if "T10Y2Y" in macro_panel else float("nan")
    ff_chg = float(last.get("FEDFUNDS_DIFF_3M", np.nan))
    un_chg = float(last.get("UNRATE_DIFF_3M", np.nan))

    score = 0.0
    breakdown: Dict[str, float] = {
        "vix_pctile": vix_pctile,
        "slope_now": slope_now,
        "slope_3m_chg": slope_chg,
        "fed_funds_3m_chg": ff_chg,
        "unrate_3m_chg": un_chg,
    }

    # VIX: high percentile -> risk-off
    if np.isfinite(vix_pctile):
        if vix_pctile >= 0.80:
            score -= 1.0
        elif vix_pctile <= 0.20:
            score += 1.0

    # Yield-curve sign: inversion is risk-off
    if np.isfinite(slope_now):
        if slope_now < 0:
            score -= 0.5
        elif slope_now > 0.5:
            score += 0.5
    # Steepening helps cyclicals (risk-on); flattening hurts
    if np.isfinite(slope_chg):
        if slope_chg > 0.25:
            score += 0.5
        elif slope_chg < -0.25:
            score -= 0.5

    # Hiking cycle is risk-off; cutting helps risk
    if np.isfinite(ff_chg):
        if ff_chg > 0.25:
            score -= 0.5
        elif ff_chg < -0.25:
            score += 0.5

    # Rising unemployment is risk-off
    if np.isfinite(un_chg):
        if un_chg > 0.20:
            score -= 0.5
        elif un_chg < -0.10:
            score += 0.25

    if score <= -1.5:
        label = "risk_off"
    elif score >= 1.5:
        label = "risk_on"
    else:
        label = "neutral"

    breakdown["regime_score"] = score
    return label, breakdown


def pillar_weights_for_regime(
    base_weights: Dict[str, float],
    regime: str,
    *,
    risk_off_tilt: Dict[str, float] | None = None,
    risk_on_tilt: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Apply additive tilt to base weights, then re-normalize to sum to 1.0."""
    out = dict(base_weights)
    tilt: Dict[str, float] = {}
    if regime == "risk_off" and risk_off_tilt:
        tilt = risk_off_tilt
    elif regime == "risk_on" and risk_on_tilt:
        tilt = risk_on_tilt

    for k, dv in tilt.items():
        out[k] = max(0.0, float(out.get(k, 0.0)) + float(dv))

    s = sum(out.values())
    if s <= 0:
        return base_weights
    return {k: float(v) / s for k, v in out.items()}
