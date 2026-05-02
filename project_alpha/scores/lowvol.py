"""
Low-vol pillar.

Three sub-features per ticker, all where *lower is better*:

  vol_252d         : 1y realized daily-return vol × √252
  beta_252d        : CAPM beta vs IWB over the same window
  vol_stability    : stdev of rolling-21d vol over the last 252d (vol-of-vol)

Each is sector-z-scored (lower-is-better → multiply by −1 so sign aligns with
the rest of the composite, where higher = better). The pillar score is the
equal-weighted average, then re-z'd cross-sectionally (NOT sector-z, so
defensives that earn the low-vol premium stay structurally over-weighted).

The low-vol anomaly is one of the most replicated single-factor effects
(Frazzini-Pedersen "Betting Against Beta"; Ang-Hodrick-Xing-Zhang). It is
*uncorrelated* with momentum and earns its own risk premium, especially in
draw-down periods — exactly when our other pillars decay.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

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


def compute_lowvol_features(
    eod: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    bench_returns: Optional[pd.Series] = None,
    vol_lookback_days: int = 252,
) -> Dict[str, float]:
    """Per-ticker low-vol features at `asof`."""
    out: Dict[str, float] = {
        "vol_252d":      float("nan"),
        "beta_252d":     float("nan"),
        "vol_stability": float("nan"),
    }
    if eod is None or eod.empty:
        return out

    df = eod.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        return out
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[df["date"] <= pd.Timestamp(asof).tz_localize(None)]
    if df.empty:
        return out

    px_col = "adjusted_close" if "adjusted_close" in df.columns else "close"
    px = pd.to_numeric(df[px_col], errors="coerce").astype(float)
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.shape[0] < 60:
        return out

    rets_dated = pd.Series(rets.values, index=pd.DatetimeIndex(df["date"].iloc[1:].values))
    tail = rets_dated.tail(vol_lookback_days)
    if len(tail) >= 60:
        out["vol_252d"] = float(tail.std(ddof=0) * np.sqrt(252))
        # Vol-stability: stdev of rolling-21d realized vol
        rv21 = tail.rolling(21).std(ddof=0).dropna()
        if len(rv21) >= 30:
            out["vol_stability"] = float(rv21.std(ddof=0))

    if bench_returns is not None and len(tail) >= 60:
        common = tail.index.intersection(bench_returns.index)
        if len(common) >= 60:
            r = tail.loc[common]
            m = bench_returns.loc[common]
            if r.std(ddof=0) > 0 and m.var() > 0:
                beta = float(r.cov(m) / m.var())
                if np.isfinite(beta):
                    out["beta_252d"] = beta

    return out


def compute_lowvol_panel(
    feature_rows: Iterable[Dict[str, object]],
    sector_map: Dict[str, str],
    *,
    winsor_k: float = 3.0,
    w_vol:    float = 0.40,
    w_beta:   float = 0.40,
    w_stab:   float = 0.20,
) -> pd.DataFrame:
    """Cross-sectional low-vol pillar panel.

    Output columns: sector, vol_z, beta_z, vol_stab_z, lowvol_score.
    Higher score = lower expected risk (and historically higher risk-adjusted return).
    """
    df = pd.DataFrame(list(feature_rows))
    if df.empty:
        return pd.DataFrame(columns=["sector", "lowvol_score"])
    if "ticker" in df.columns:
        df = df.set_index("ticker")
    sectors = pd.Series({t: sector_map.get(t, "Unknown") for t in df.index}, name="sector")

    # All three are "lower is better" → invert sign by negating after z.
    vol_z   = _z_by_sector(df.get("vol_252d",      pd.Series(np.nan, index=df.index)), sectors, k=winsor_k)
    beta_z  = _z_by_sector(df.get("beta_252d",     pd.Series(np.nan, index=df.index)), sectors, k=winsor_k)
    stab_z  = _z_by_sector(df.get("vol_stability", pd.Series(np.nan, index=df.index)), sectors, k=winsor_k)

    score = (
        - float(w_vol)  * vol_z.fillna(0.0)
        - float(w_beta) * beta_z.fillna(0.0)
        - float(w_stab) * stab_z.fillna(0.0)
    )

    # Global z (NOT sector-z) — the low-vol *premium* is concentrated in defensive
    # sectors, and we want that structural signal to flow into the composite.
    lowvol_score = _winsorize_z(score, k=winsor_k)

    out = pd.DataFrame({
        "sector":       sectors,
        "vol_z":        vol_z,
        "beta_z":       beta_z,
        "vol_stab_z":   stab_z,
        "lowvol_score": lowvol_score,
    })
    return out
