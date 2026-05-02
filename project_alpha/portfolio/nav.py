"""
NAV / P&L stitching helpers.

Pure functions that take per-rebalance weights + a daily returns panel and
produce a strategy-level daily return series and NAV curve. Used by the
backtest orchestrator's reporting path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


# --------------------------
# Core helpers
# --------------------------
def realized_vol_annual(r: pd.Series) -> float:
    if r is None or r.empty:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))


def realized_beta(port: pd.Series, bench: pd.Series) -> float:
    """OLS beta of port vs bench on the intersection of dates."""
    idx = port.index.intersection(bench.index)
    if len(idx) < 20:
        return 0.0
    p = port.loc[idx]
    b = bench.loc[idx]
    vb = float(b.var(ddof=0))
    if vb <= 0:
        return 0.0
    cov = float(p.cov(b))
    beta = cov / vb
    return beta if np.isfinite(beta) else 0.0


def realized_beta_window(port: pd.Series, bench: pd.Series, window_days: int) -> float:
    """Beta on the last `window_days` calendar points of the intersection."""
    idx = port.index.intersection(bench.index)
    if len(idx) < max(20, window_days // 4):
        return realized_beta(port, bench)
    idx = idx[-window_days:]
    p = port.loc[idx]
    b = bench.loc[idx]
    vb = float(b.var(ddof=0))
    if vb <= 0:
        return 0.0
    cov = float(p.cov(b))
    beta = cov / vb
    return beta if np.isfinite(beta) else 0.0


# --------------------------
# PnL stitching
# --------------------------
def stitch_pnl(
    stock_rets: pd.DataFrame,
    weights: pd.Series,
    start_excl: pd.Timestamp,
    end_incl: pd.Timestamp,
    per_name_daily_abs_cap: float | None = None,
) -> pd.Series:
    """
    Compute daily portfolio returns using fixed weights between (start_excl, end_incl].
    Optional per-name clipping protects against stale/split spikes in vendor data.
    """
    if stock_rets.empty or weights.empty:
        return pd.Series(dtype=float)

    w = weights.reindex(stock_rets.columns).fillna(0.0)
    if float(w.sum()) == 0.0:
        return pd.Series(dtype=float)

    seg = stock_rets.loc[(stock_rets.index > start_excl) & (stock_rets.index <= end_incl)]
    if seg.empty:
        return pd.Series(dtype=float)

    if per_name_daily_abs_cap is not None and per_name_daily_abs_cap > 0:
        seg = seg.clip(lower=-per_name_daily_abs_cap, upper=per_name_daily_abs_cap)

    pnl = seg.mul(w, axis=1).sum(axis=1)
    pnl.name = "ret"
    return pnl


# --------------------------
# Simple vol targeting
# --------------------------
def vol_target(r: pd.Series, target_annual_vol: float) -> pd.Series:
    if r is None or r.empty or target_annual_vol is None:
        return r
    vol = realized_vol_annual(r)
    if vol <= 0:
        return r
    k = float(target_annual_vol / vol)
    return r * k


# --------------------------
# Dual vol + beta targeting (robust)
# --------------------------
def vol_beta_target(
    port: pd.Series,
    bench: pd.Series,
    target_vol: float | None = None,
    target_beta: float | None = None,
    beta_tol: float = 0.05,
    vol_target_window_days: int = 252,
    vol_floor_annual: float = 0.04,
    beta_floor_abs: float = 0.05,
    scale_cap: float = 3.0,
    daily_abs_cap: float = 0.25,
) -> pd.Series:
    """
    Scale daily returns to hit both annualized vol and beta targets if provided.
    - Uses a windowed vol/beta estimate
    - Floors tiny vol/beta to avoid huge multipliers
    - Caps the overall scale factor
    - Ensures no scaled daily move exceeds `daily_abs_cap`
    """
    if port is None or port.empty:
        return port

    # Windowed volatility
    r_win = port.tail(vol_target_window_days)
    vol0 = float(r_win.std(ddof=0) * np.sqrt(TRADING_DAYS))
    vol_eff = max(vol0, float(vol_floor_annual)) if target_vol else None

    # Windowed beta
    beta_eff = None
    if target_beta is not None:
        beta0 = realized_beta_window(port, bench, window_days=vol_target_window_days)
        beta_eff = beta0 if abs(beta0) >= float(beta_floor_abs) else None

    k_candidates = []
    if target_vol and vol_eff and vol_eff > 0:
        k_candidates.append(float(target_vol / vol_eff))
    if target_beta is not None and beta_eff:
        k_candidates.append(float(target_beta / beta_eff))

    if not k_candidates:
        out = port.copy()
    else:
        k = min(k_candidates)
        k = float(np.clip(k, 1.0 / scale_cap, scale_cap))  # cap both up and down
        out = port * k

    # Ensure no scaled daily return exceeds cap
    max_abs = float(np.nanmax(np.abs(out.values))) if len(out) else 0.0
    if max_abs > daily_abs_cap and max_abs > 0:
        k3 = float(daily_abs_cap / max_abs)
        out = out * (0.9 * k3 + 0.1)  # gentle correction

    # Final beta nudge if still out of band
    if target_beta is not None:
        beta1 = realized_beta_window(out, bench, window_days=vol_target_window_days)
        if beta1 < target_beta - beta_tol or beta1 > target_beta + beta_tol:
            if beta1 != 0:
                k2 = float(target_beta / beta1)
                k2 = float(np.clip(k2, 1.0 / scale_cap, scale_cap))
                out = out * (0.5 * k2 + 0.5)

    # One more safety clamp on daily magnitude
    out = out.clip(lower=-daily_abs_cap, upper=daily_abs_cap)
    return out


def beta_overlay(
    port: pd.Series,
    bench: pd.Series,
    beta_target: float,
    clamp: float = 0.35,
    window_days: int = 252,
) -> tuple[pd.Series, float]:
    """
    Return a blended series port' = (1-θ)·port + θ·bench so that beta(port', bench) ≈ beta_target.
    θ is clamped to avoid extreme overlays; returns (series, theta_used).
    """
    idx = port.index.intersection(bench.index)
    if len(idx) < 60:
        return port, 0.0
    p = port.loc[idx]
    b = bench.loc[idx]

    # β((1-θ)p + θb) = (1-θ)βp + θ*1
    vb = float(b.var(ddof=0))
    if vb <= 0:
        return port, 0.0
    # use a rolling window for more responsive beta
    use = idx[-window_days:] if len(idx) > window_days else idx
    p_w = p.loc[use]
    b_w = b.loc[use]
    vb = float(b_w.var(ddof=0))
    if vb <= 0:
        return port, 0.0
    beta_p = float(p_w.cov(b_w) / vb)
    if not np.isfinite(beta_p):
        return port, 0.0

    denom = (1.0 - beta_p) if abs(1.0 - beta_p) > 1e-9 else 1e-9
    theta = float(np.clip((beta_target - beta_p) / denom, -clamp, clamp))

    blend = (1.0 - theta) * p + theta * b
    out = port.copy()
    out.loc[idx] = blend
    return out, theta

