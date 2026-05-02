"""
Fundamental score.

Inputs are point-in-time TTM features from `data.fundamentals.point_in_time_features`.

We do **not** combine pillars here — each cross-section returns one z-scored
fundamental signal per ticker, sector-neutralized. The composite step
combines fundamental, technical, sentiment, and macro.

Sub-signals (winsorized at ±3σ within sector before z-score):
  Quality:    roe, roa, gross_margin, operating_margin, net_margin, fcf_margin
  Value:      ev_ebitda (inv), pe (inv), pb (inv), ps (inv), fcf_yield
  Growth:     revenue YoY, eps YoY (when available)
  Leverage:   -tanh(debt_to_equity)
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


_QUALITY_FIELDS = ["roe", "roa", "gross_margin", "operating_margin", "net_margin", "fcf_margin"]
_VALUE_FIELDS_INVERSE = ["pe", "pb", "ps", "ev_ebitda"]   # lower = better, invert for ranking
_VALUE_FIELDS_DIRECT  = ["fcf_yield"]                     # higher = better
_GROWTH_FIELDS = ["rev_growth_yoy", "eps_growth_yoy"]


def _winsorize_z(series: pd.Series, *, k: float = 3.0) -> pd.Series:
    """Cross-sectional winsorize (±k stdevs) then z-score with population std."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    mu = float(s.mean(skipna=True))
    sd = float(s.std(ddof=0, skipna=True))
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    s_clip = s.clip(lower=mu - k * sd, upper=mu + k * sd)
    return (s_clip - mu) / sd


def _z_by_sector(values: pd.Series, sectors: pd.Series, *, k: float = 3.0) -> pd.Series:
    """Sector-neutral winsorized z-score. Sectors with <3 names get global z-score."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for sec, idx in sectors.groupby(sectors).groups.items():
        sub = values.loc[idx]
        if sub.dropna().shape[0] >= 3:
            out.loc[idx] = _winsorize_z(sub, k=k).values
    # Fill remaining (small sectors) with global z
    missing = out.isna() & values.notna()
    if missing.any():
        out.loc[missing] = _winsorize_z(values.loc[missing], k=k).values
    return out


def _inv(s: pd.Series) -> pd.Series:
    """Invert a positive multiple (so 'cheap' = high). NaN if non-positive."""
    out = pd.to_numeric(s, errors="coerce").astype(float).copy()
    out[out <= 0] = np.nan
    return 1.0 / out


def compute_fundamental_panel(
    feature_rows: Iterable[Dict[str, object]],
    *,
    winsor_k: float = 3.0,
    quality_w: float = 0.40,
    value_w: float = 0.30,
    growth_w: float = 0.20,
    leverage_w: float = 0.10,
) -> pd.DataFrame:
    """Cross-sectional fundamental score panel.

    Returns DataFrame indexed by ticker with columns:
      sector, quality_z, value_z, growth_z, leverage_z, fundamental_score.
    """
    df = pd.DataFrame(list(feature_rows))
    if df.empty:
        return pd.DataFrame(columns=[
            "sector", "quality_z", "value_z", "growth_z", "leverage_z", "fundamental_score"
        ])
    df = df.set_index("ticker")
    sectors = df["sector"].fillna("Unknown").astype(str)

    # Quality
    quality_signals: List[pd.Series] = []
    for f in _QUALITY_FIELDS:
        if f in df.columns:
            quality_signals.append(_z_by_sector(df[f], sectors, k=winsor_k))
    quality_z = pd.concat(quality_signals, axis=1).mean(axis=1) if quality_signals else pd.Series(0.0, index=df.index)

    # Value (invert positive multiples, then z-score)
    value_signals: List[pd.Series] = []
    for f in _VALUE_FIELDS_INVERSE:
        if f in df.columns:
            value_signals.append(_z_by_sector(_inv(df[f]), sectors, k=winsor_k))
    for f in _VALUE_FIELDS_DIRECT:
        if f in df.columns:
            value_signals.append(_z_by_sector(df[f], sectors, k=winsor_k))
    value_z = pd.concat(value_signals, axis=1).mean(axis=1) if value_signals else pd.Series(0.0, index=df.index)

    # Growth
    growth_signals: List[pd.Series] = []
    for f in _GROWTH_FIELDS:
        if f in df.columns:
            growth_signals.append(_z_by_sector(df[f], sectors, k=winsor_k))
    growth_z = pd.concat(growth_signals, axis=1).mean(axis=1) if growth_signals else pd.Series(0.0, index=df.index)

    # Leverage penalty (lower D/E → higher score)
    if "debt_to_equity" in df.columns:
        de = pd.to_numeric(df["debt_to_equity"], errors="coerce").astype(float)
        leverage_z = _z_by_sector(-np.tanh(de.clip(lower=0, upper=10)), sectors, k=winsor_k)
    else:
        leverage_z = pd.Series(0.0, index=df.index)

    composite = (
        float(quality_w)  * quality_z.fillna(0.0)
        + float(value_w)    * value_z.fillna(0.0)
        + float(growth_w)   * growth_z.fillna(0.0)
        + float(leverage_w) * leverage_z.fillna(0.0)
    )

    out = pd.DataFrame({
        "sector": sectors,
        "quality_z": quality_z,
        "value_z": value_z,
        "growth_z": growth_z,
        "leverage_z": leverage_z,
        "fundamental_score": composite,
    })
    # Final cross-sectional standardisation so fundamental_score ~ z-scale
    out["fundamental_score"] = _z_by_sector(out["fundamental_score"], sectors, k=winsor_k)
    return out
