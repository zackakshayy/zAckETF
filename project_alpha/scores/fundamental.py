"""
Fundamental score — Iter-13 rebuild.

The previous fundamental pillar blended quality + naive-value + growth + leverage
into ONE muddy score and produced negative IC (-0.037 full, -0.056 recent). Root
cause: the naive-value component (cheap PE/PB/PS/EV) bought *value traps* — cheap
stocks that were cheap because earnings were deteriorating — and dragged down the
genuinely-positive quality signal.

Iter-13 fixes this by splitting the pillar into THREE clean, separately-validated
sub-pillars, each exposed as its own column so the IC tracker can score them:

  quality_z   — durable profitability/efficiency (the persistent alpha)
                roe, roa, gross/operating/net/fcf margin, low leverage
  value_z     — DISCIPLINED value: cheapness GATED by quality, so cheap-junk
                (low-quality names) has its value signal damped, never rewarded
  catalyst_z  — PEAD: earnings-surprise drift + EPS-trend (point-in-time, from
                Earnings.History announcement dates — no look-ahead)

DATA-COVERAGE NOTE (honest caveat): `Earnings.History` exists for only ~11%
of the Russell 1000 in the source dataset, so the catalyst sub-pillar is
exposed as a column for transparency/IC-tracking but its default blend weight
is 0.0 — a signal covering 11% of names would be sparse and selection-biased.
The "catalyst" role (price-confirmed trend) is instead carried by the slow
12-1 technical-momentum pillar, which has 100% coverage. `fundamental_score`
therefore defaults to a Quality + Disciplined-Value blend — both ~94% covered
and point-in-time clean. Quality and value persist across 6-month holds
(factor half-life of years), which is what makes this pillar viable for a
semi-annual rebalance — unlike fast price momentum.

Inputs are per-ticker dict rows that merge:
  - data.fundamentals.point_in_time_features   (margins, roe, fcf_yield, pe, ...)
  - data.fundamentals.point_in_time_earnings   (surprise_avg, eps_ttm_growth, ...)
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


# Quality: all "higher = better". debt_to_equity handled separately (inverted).
_QUALITY_FIELDS = ["roe", "roa", "gross_margin", "operating_margin",
                   "net_margin", "fcf_margin"]
# Value: positive multiples inverted so "cheap = high"; fcf_yield already direct.
_VALUE_FIELDS_INVERSE = ["pe", "pb", "ps", "ev_ebitda"]
_VALUE_FIELDS_DIRECT = ["fcf_yield"]
# Catalyst: point-in-time earnings-surprise / EPS-trend fields.
_CATALYST_FIELDS = ["surprise_avg", "surprise_streak", "eps_ttm_growth"]


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
    """Sector-neutral winsorized z-score. Sectors with <3 names get global z."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for _, idx in sectors.groupby(sectors).groups.items():
        sub = values.loc[idx]
        if sub.dropna().shape[0] >= 3:
            out.loc[idx] = _winsorize_z(sub, k=k).values
    missing = out.isna() & values.notna()
    if missing.any():
        out.loc[missing] = _winsorize_z(values.loc[missing], k=k).values
    return out


def _inv(s: pd.Series) -> pd.Series:
    """Invert a positive multiple (so 'cheap' = high). NaN if non-positive —
    a negative PE/EV is meaningless as 'cheap' and must not score well."""
    out = pd.to_numeric(s, errors="coerce").astype(float).copy()
    out[out <= 0] = np.nan
    return 1.0 / out


def _mean_z(df: pd.DataFrame, fields: List[str], sectors: pd.Series,
            *, k: float, invert: bool = False) -> pd.Series:
    """Average of sector-neutral z-scores over `fields` present in `df`."""
    sigs: List[pd.Series] = []
    for f in fields:
        if f in df.columns:
            col = _inv(df[f]) if invert else df[f]
            sigs.append(_z_by_sector(col, sectors, k=k))
    if not sigs:
        return pd.Series(0.0, index=df.index)
    return pd.concat(sigs, axis=1).mean(axis=1)


def compute_fundamental_panel(
    feature_rows: Iterable[Dict[str, object]],
    *,
    winsor_k: float = 3.0,
    quality_w: float = 0.60,
    value_w: float = 0.40,
    catalyst_w: float = 0.0,   # 0 by default — Earnings.History covers only ~11%
    value_quality_gate: float = 0.60,
) -> pd.DataFrame:
    """Cross-sectional fundamental score panel — Iter-13 three-pillar rebuild.

    Returns a DataFrame indexed by ticker with columns:
      sector, quality_z, value_z, catalyst_z, growth_z, fundamental_score

    `value_z` is DISCIPLINED value: raw cheapness multiplied by a quality gate
    in [1-g, 1+g] (g = `value_quality_gate`), so a cheap low-quality name has
    its value score damped toward — or below — zero, while a cheap high-quality
    name keeps (or amplifies) it. This is the value-trap fix.

    `growth_z` is kept as an alias of `catalyst_z` for backward compatibility
    with any caller still expecting the old column name.
    """
    df = pd.DataFrame(list(feature_rows))
    cols = ["sector", "quality_z", "value_z", "catalyst_z", "growth_z", "fundamental_score"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.set_index("ticker")
    sectors = df["sector"].fillna("Unknown").astype(str)

    # ---- Quality: profitability/efficiency + low leverage --------------------
    quality_core = _mean_z(df, _QUALITY_FIELDS, sectors, k=winsor_k)
    if "debt_to_equity" in df.columns:
        de = pd.to_numeric(df["debt_to_equity"], errors="coerce").astype(float)
        leverage_z = _z_by_sector(-np.tanh(de.clip(lower=0, upper=10)), sectors, k=winsor_k)
    else:
        leverage_z = pd.Series(0.0, index=df.index)
    # Leverage is a modest quality input (0.80 core / 0.20 leverage).
    quality_z = (0.80 * quality_core.fillna(0.0) + 0.20 * leverage_z.fillna(0.0))
    quality_z = _z_by_sector(quality_z, sectors, k=winsor_k)

    # ---- Value: cheapness, then GATED by quality -----------------------------
    value_raw = pd.concat([
        _mean_z(df, _VALUE_FIELDS_INVERSE, sectors, k=winsor_k, invert=True),
        _mean_z(df, _VALUE_FIELDS_DIRECT, sectors, k=winsor_k),
    ], axis=1).mean(axis=1)
    # Quality gate: map quality_z (~N(0,1)) through a sigmoid to [1-g, 1+g].
    g = float(np.clip(value_quality_gate, 0.0, 1.0))
    q_clip = quality_z.fillna(0.0).clip(-3.0, 3.0)
    gate = 1.0 + g * np.tanh(q_clip / 1.5)          # cheap-junk damped, cheap-quality boosted
    value_z = _z_by_sector(value_raw.fillna(0.0) * gate, sectors, k=winsor_k)

    # ---- Catalyst: PEAD earnings-surprise drift + EPS trend ------------------
    catalyst_z = _z_by_sector(
        _mean_z(df, _CATALYST_FIELDS, sectors, k=winsor_k), sectors, k=winsor_k)

    # ---- Quality-led blend ---------------------------------------------------
    total_w = float(quality_w + value_w + catalyst_w) or 1.0
    qw, vw, cw = quality_w / total_w, value_w / total_w, catalyst_w / total_w
    composite = (qw * quality_z.fillna(0.0)
                 + vw * value_z.fillna(0.0)
                 + cw * catalyst_z.fillna(0.0))

    out = pd.DataFrame({
        "sector": sectors,
        "quality_z": quality_z,
        "value_z": value_z,
        "catalyst_z": catalyst_z,
        "growth_z": catalyst_z,                      # back-compat alias
        "fundamental_score": _z_by_sector(composite, sectors, k=winsor_k),
    })
    return out
