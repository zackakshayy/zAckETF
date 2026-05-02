"""
Composite score combiner.

Takes the four pillar score panels (technical, fundamental, sentiment, macro
overlay) and produces a single per-ticker composite using regime-adjusted
weights. The macro pillar here is the *regime-implied sector tilt* — a
per-sector value, broadcast to each ticker by sector — not a per-stock score.

Pillar conventions (each panel has columns including {sector, <pillar>_score}):
  technical:    technical_score
  fundamental:  fundamental_score
  sentiment:    sentiment_score
  macro_sector: macro_sector_tilt (Series indexed by sector)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _winsorize_z(series: pd.Series, *, k: float = 3.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    mu = float(s.mean(skipna=True))
    sd = float(s.std(ddof=0, skipna=True))
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return ((s.clip(lower=mu - k * sd, upper=mu + k * sd) - mu) / sd)


def _z_by_sector(values: pd.Series, sectors: pd.Series, *, k: float = 3.0) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for sec, idx in sectors.groupby(sectors).groups.items():
        sub = values.loc[idx]
        if sub.dropna().shape[0] >= 3:
            out.loc[idx] = _winsorize_z(sub, k=k).values
    missing = out.isna() & values.notna()
    if missing.any():
        out.loc[missing] = _winsorize_z(values.loc[missing], k=k).values
    return out


def macro_sector_tilt(
    sectors: pd.Index,
    regime: str,
) -> pd.Series:
    """Map regime → per-sector tilt. Returns z-scaled (mean 0, scale ~1)."""
    cyclical = {"Information Technology", "Financials", "Industrials",
                "Consumer Discretionary", "Materials", "Energy",
                "Communication Services"}
    defensive = {"Consumer Staples", "Utilities", "Health Care", "Real Estate"}

    raw = {}
    if regime == "risk_on":
        for s in sectors:
            raw[s] = +1.0 if s in cyclical else (-1.0 if s in defensive else 0.0)
    elif regime == "risk_off":
        for s in sectors:
            raw[s] = -1.0 if s in cyclical else (+1.0 if s in defensive else 0.0)
    else:
        for s in sectors:
            raw[s] = 0.0
    s = pd.Series(raw)
    if s.std(ddof=0) == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def combine_composite(
    technical_panel: pd.DataFrame,
    fundamental_panel: pd.DataFrame,
    sentiment_panel: pd.DataFrame,
    *,
    pillar_weights: Dict[str, float],
    regime: str,
    sector_neutralize: bool = True,
    winsor_k: float = 3.0,
    lowvol_panel: Optional[pd.DataFrame] = None,
    quality_momentum_gate_w: float = 0.0,
) -> pd.DataFrame:
    """Build composite score panel keyed by ticker.

    `pillar_weights` keys: technical, fundamental, sentiment, lowvol, macro.
    Returns DataFrame with sector + every pillar score + composite_score.

    `quality_momentum_gate_w`: if > 0, adds an interaction term
        gate_score = z( max(0,quality_z) * max(0,momentum_z) )
    weighted by `quality_momentum_gate_w`. The interaction filters value-traps
    (high quality without trend) and junk-momentum (high trend without quality).
    """
    panels = {
        "technical":   ("technical_score",   technical_panel),
        "fundamental": ("fundamental_score", fundamental_panel),
        "sentiment":   ("sentiment_score",   sentiment_panel),
        "lowvol":      ("lowvol_score",      lowvol_panel),
    }
    # Tickers covered by at least one panel
    all_tickers = set()
    for _, (_, p) in panels.items():
        if p is not None and not p.empty:
            all_tickers.update(p.index.tolist())
    if not all_tickers:
        return pd.DataFrame(columns=["sector", "composite_score"])

    idx = pd.Index(sorted(all_tickers))

    # Sector — prefer fundamental's mapping, fall back to others
    sec_series = pd.Series(index=idx, dtype=object)
    for _, (_, p) in panels.items():
        if p is None or p.empty or "sector" not in p.columns:
            continue
        for t in p.index:
            if pd.isna(sec_series.get(t)):
                sec_series.loc[t] = p["sector"].get(t)
    sec_series = sec_series.fillna("Unknown").astype(str)

    score_df = pd.DataFrame(index=idx)
    for name, (col, p) in panels.items():
        if p is None or p.empty or col not in p.columns:
            score_df[name] = 0.0
        else:
            score_df[name] = pd.to_numeric(p[col], errors="coerce").reindex(idx).fillna(0.0)

    # Macro: per-sector tilt, broadcast to tickers
    sector_tilt = macro_sector_tilt(sec_series.unique(), regime=regime)
    score_df["macro"] = sec_series.map(sector_tilt).fillna(0.0).values

    w = pillar_weights

    composite = (
        float(w.get("technical", 0.0))   * score_df["technical"]
        + float(w.get("fundamental", 0.0)) * score_df["fundamental"]
        + float(w.get("sentiment", 0.0))   * score_df["sentiment"]
        + float(w.get("lowvol", 0.0))      * score_df["lowvol"]
        + float(w.get("macro", 0.0))       * score_df["macro"]
    )

    # Quality × Momentum gate: only fires when *both* are positive (real conviction).
    if quality_momentum_gate_w > 0:
        q = pd.to_numeric(fundamental_panel.get("quality_z", pd.Series(0.0, index=idx)), errors="coerce") \
              .reindex(idx).fillna(0.0) if fundamental_panel is not None else pd.Series(0.0, index=idx)
        m = pd.to_numeric(technical_panel.get("momentum_z", pd.Series(0.0, index=idx)), errors="coerce") \
              .reindex(idx).fillna(0.0) if technical_panel is not None else pd.Series(0.0, index=idx)
        gate_raw = q.clip(lower=0.0) * m.clip(lower=0.0)
        gate_z = _winsorize_z(gate_raw, k=winsor_k).fillna(0.0)
        composite = composite + float(quality_momentum_gate_w) * gate_z

    if sector_neutralize:
        composite = _z_by_sector(composite, sec_series, k=winsor_k).fillna(0.0)
    else:
        composite = _winsorize_z(composite, k=winsor_k).fillna(0.0)

    out = pd.DataFrame({
        "sector":            sec_series,
        "technical_score":   score_df["technical"],
        "fundamental_score": score_df["fundamental"],
        "sentiment_score":   score_df["sentiment"],
        "lowvol_score":      score_df["lowvol"],
        "macro_score":       score_df["macro"],
        "composite_score":   composite,
    })
    out.attrs["pillar_weights"] = dict(w)
    out.attrs["regime"] = regime
    return out
