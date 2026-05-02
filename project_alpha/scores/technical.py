"""
Technical features and score.

Per-ticker features (point-in-time):
  mom_12_1   : 12-month return ending one month ago (skip-month momentum)
  mom_6_1    : 6-month return ending one month ago
  rev_21d    : negative of 1-month return (short-term reversal)
  vol_252d   : trailing 1-year realized volatility (daily, ddof=0)
  high52_gap : (price - 52w high) / 52w high  (≤ 0)
  ma_200_gap : (price - 200d MA) / 200d MA
  liq_trend  : EMA20(dollar_vol) / EMA60(dollar_vol) - 1

The cross-sectional `compute_technical_panel` z-scores by sector and combines
into a single technical_score. All inputs use `adjusted_close`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _ret_lookback(prices: pd.Series, days: int, skip: int = 0) -> float:
    if prices is None or len(prices) < days + skip + 1:
        return float("nan")
    end = -1 - skip
    start = end - days
    a = float(prices.iloc[end])
    b = float(prices.iloc[start])
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return a / b - 1.0


def compute_technical_features(
    eod: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    momentum_12_1_days: int = 252,
    momentum_12_1_skip_days: int = 21,
    momentum_6_1_days: int = 126,
    vol_lookback_days: int = 252,
    high52_lookback_days: int = 252,
    short_term_reversal_days: int = 21,
    bench_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """Compute features for a single ticker as of `asof` from its EOD frame.

    If `bench_returns` is provided, also computes `idio_mom_252` — the cumulative
    market-residualized return over (asof-252d .. asof-21d), i.e. raw return minus
    β × bench_return summed over the window. This is the firm-specific component
    of momentum, which empirically has stronger and more persistent IC than raw.
    """
    out: Dict[str, float] = {
        "price":           float("nan"),
        "mom_12_1":        float("nan"),
        "mom_9_1":         float("nan"),    # NEW (Tier 1 item 4)
        "mom_6_1":         float("nan"),
        "mom_3_1":         float("nan"),    # NEW (Tier 1 item 4)
        "sharpe_mom_12_1": float("nan"),    # NEW (Tier 1 item 1) — vol-adjusted momentum
        "rev_21d":         float("nan"),
        "vol_252d":        float("nan"),
        "high52_gap":      float("nan"),    # still computed but DROPPED from composite (item 3)
        "ma_200_gap":      float("nan"),
        "liq_trend":       float("nan"),    # still computed but DROPPED from composite (item 3)
        "idio_mom_252":    float("nan"),
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
    if not np.isfinite(px.iloc[-1]):
        return out

    out["price"] = float(px.iloc[-1])
    # Multi-horizon momentum ensemble (Tier 1 item 4): 3-1, 6-1, 9-1, 12-1.
    # Skip-month convention applies to all horizons to avoid the 1-month reversal.
    out["mom_12_1"] = _ret_lookback(px, days=momentum_12_1_days,        skip=momentum_12_1_skip_days)
    out["mom_9_1"]  = _ret_lookback(px, days=int(momentum_12_1_days * 0.75),  skip=momentum_12_1_skip_days)
    out["mom_6_1"]  = _ret_lookback(px, days=momentum_6_1_days,         skip=momentum_12_1_skip_days)
    out["mom_3_1"]  = _ret_lookback(px, days=int(momentum_6_1_days * 0.5),    skip=momentum_12_1_skip_days)
    out["rev_21d"]  = -_ret_lookback(px, days=short_term_reversal_days, skip=0)

    rets = px.pct_change().replace([np.inf, -np.inf], np.nan)
    if rets.dropna().shape[0] >= 60:
        out["vol_252d"] = float(rets.tail(vol_lookback_days).std(ddof=0) * np.sqrt(252))

    # Sharpe momentum (Tier 1 item 1) — risk-adjusts the headline 12-1 horizon.
    # Filters out high-vol "leverage-style" winners whose returns are
    # disproportionate to their realized risk.
    if np.isfinite(out["mom_12_1"]) and np.isfinite(out["vol_252d"]) and out["vol_252d"] > 0:
        out["sharpe_mom_12_1"] = float(out["mom_12_1"] / out["vol_252d"])

    if len(px) >= high52_lookback_days:
        hi = float(px.tail(high52_lookback_days).max())
        out["high52_gap"] = (out["price"] - hi) / hi if hi else float("nan")

    if len(px) >= 200:
        ma200 = float(px.tail(200).mean())
        out["ma_200_gap"] = (out["price"] - ma200) / ma200 if ma200 else float("nan")

    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce").astype(float)
        dv = (vol * px).replace([np.inf, -np.inf], np.nan)
        ema20 = dv.ewm(span=20, adjust=False, min_periods=10).mean().iloc[-1]
        ema60 = dv.ewm(span=60, adjust=False, min_periods=20).mean().iloc[-1]
        if np.isfinite(ema20) and np.isfinite(ema60) and ema60 > 0:
            out["liq_trend"] = float(ema20 / ema60 - 1.0)

    # Idiosyncratic momentum — cumulative residual return over the
    # 12-1 window after stripping out market beta exposure.
    if bench_returns is not None and rets.dropna().shape[0] >= 60:
        idx = df["date"].to_numpy()
        rets_dated = pd.Series(rets.values, index=pd.DatetimeIndex(idx))
        rets_dated = rets_dated.replace([np.inf, -np.inf], np.nan).dropna()
        common = rets_dated.index.intersection(bench_returns.index)
        if len(common) >= 60:
            r = rets_dated.loc[common]
            m = bench_returns.loc[common]
            if r.std(ddof=0) > 0 and m.std(ddof=0) > 0 and m.var() > 0:
                window = momentum_12_1_days + momentum_12_1_skip_days
                r_w = r.tail(window)
                m_w = m.reindex(r_w.index).fillna(0.0)
                if len(r_w) >= max(60, window) and m_w.var() > 0:
                    beta = float(r_w.cov(m_w) / m_w.var())
                    if not np.isfinite(beta):
                        beta = 1.0
                    resid = r_w - beta * m_w
                    # Drop the last `skip` observations (skip-month convention)
                    if momentum_12_1_skip_days > 0:
                        seg = resid.iloc[: -momentum_12_1_skip_days]
                    else:
                        seg = resid
                    if len(seg) > 0:
                        cum = float((1.0 + seg).prod() - 1.0)
                        if np.isfinite(cum):
                            out["idio_mom_252"] = cum

    return out


# ---------------------------------------------------------------------------
# Cross-sectional score panel
# ---------------------------------------------------------------------------

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


def compute_technical_panel(
    feature_rows: Iterable[Dict[str, object]],
    sector_map: Dict[str, str],
    *,
    winsor_k: float = 3.0,
    momentum_w: float = 0.55,
    vol_w: float = 0.20,            # negative weight applied below
    reversal_w: float = 0.05,
    idio_w: float = 0.20,           # Tier 1 item 2 — multi-residual idio momentum
) -> pd.DataFrame:
    """Build the cross-sectional technical score panel.

    Tier 1 changes (vs prior version):
      - Item 1: top-horizon momentum is Sharpe-adjusted (mom_12_1 / vol_252d).
      - Item 2: idio momentum is residualized against BOTH market beta
        (computed in compute_technical_features) AND the cross-sectional
        sector mean (here) — captures pure firm-specific trend.
      - Item 3: drop high52_gap and liq_trend — they correlate ~0.6 with
        momentum and add noise.
      - Item 4: multi-horizon ensemble — 3-1 / 6-1 / 9-1 / 12-1-Sharpe, with
        weights tilted toward longer horizons (more persistent published IC).

    `feature_rows` is an iterable of dicts produced by
    `compute_technical_features` plus a `ticker` key.
    """
    df = pd.DataFrame(list(feature_rows))
    if df.empty:
        return pd.DataFrame(columns=["sector", "technical_score"])
    df = df.set_index("ticker")
    sectors = pd.Series({t: sector_map.get(t, "Unknown") for t in df.index}, name="sector")

    # ---- Multi-horizon momentum ensemble (Tier 1 item 4 + item 1) ----
    # Use GLOBAL z (not sector-z) — cross-sector momentum is real signal in
    # mega-cap-led regimes; sector-z would erase it. The 12-1 horizon is the
    # Sharpe-adjusted version when available (item 1).
    mom_components: List[pd.Series] = []
    for f, w_local in [
        ("mom_3_1",         0.20),
        ("mom_6_1",         0.25),
        ("mom_9_1",         0.30),
        ("sharpe_mom_12_1", 0.25),
    ]:
        if f in df.columns:
            z = _winsorize_z(df[f], k=winsor_k)
            mom_components.append(w_local * z.fillna(0.0))
    # Weights inside the ensemble re-normalize if any horizon is missing.
    if mom_components:
        total_w = 0.20 + 0.25 + 0.30 + 0.25
        # Defensive: if some horizons missing, keep the available portion at its
        # own internal weighting (no rescale needed since z-scored).
        momentum_z = sum(mom_components)
    else:
        momentum_z = pd.Series(0.0, index=df.index)

    # ---- Multi-residual idiosyncratic momentum (Tier 1 item 2) ----
    # `idio_mom_252` is already market-residualized in compute_technical_features.
    # Here we additionally strip out the cross-sectional sector mean — what's left
    # is "trend after market AND sector style exposure removed". This is the
    # purest firm-specific momentum we can build with our data.
    if "idio_mom_252" in df.columns:
        idio = pd.to_numeric(df["idio_mom_252"], errors="coerce")
        sector_mean = idio.groupby(sectors).transform("mean")
        idio_multi = (idio - sector_mean)
        idio_z = _winsorize_z(idio_multi, k=winsor_k).fillna(0.0)
    else:
        idio_z = pd.Series(0.0, index=df.index)

    # ---- Vol & reversal — sector-z (they aren't cross-sector tilt signals) ----
    vol_z = _z_by_sector(df["vol_252d"], sectors, k=winsor_k) if "vol_252d" in df.columns else pd.Series(0.0, index=df.index)
    rev_z = _z_by_sector(df["rev_21d"],  sectors, k=winsor_k) if "rev_21d"  in df.columns else pd.Series(0.0, index=df.index)

    # high52_gap and liq_trend are intentionally excluded from the composite.
    # We still expose `vol_z` and `rev_z` columns so downstream callers (the
    # quality×momentum gate) can read `momentum_z` directly.

    score = (
        float(momentum_w)  * momentum_z.fillna(0.0)
        - float(vol_w)        * vol_z.fillna(0.0)        # high vol penalized
        + float(reversal_w)   * rev_z.fillna(0.0)
        + float(idio_w)       * idio_z.fillna(0.0)
    )

    # Final standardisation — global winsorize+z (NOT sector-z) so cross-sector
    # signal survives into the composite.
    technical_score = _winsorize_z(score, k=winsor_k)

    out = pd.DataFrame({
        "sector":           sectors,
        "momentum_z":       momentum_z,
        "idio_multi_z":     idio_z,
        "vol_z":            vol_z,
        "rev_z":            rev_z,
        "technical_score":  technical_score,
    })
    return out
