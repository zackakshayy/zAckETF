"""
Portfolio construction.

Two public surfaces:

  optimize_sector_matched
    Long-only convex program (cvxpy + ECOS, with SCS fallback) that maximizes
    `score · w − λ_risk · ||diag(σ_252) · w||₂ − λ_turnover · ||w − w_prev||₁`
    subject to:
      - long-only, sum-to-1
      - per-sector weight = sector_target  (or within ±sector_band)
      - β·w in [β_target ± β_tol]
      - per-name cap = min(adv_consume_fraction · ADV$/NAV, max_name_weight)
    Falls back to a proportional-by-sector allocator if cvxpy is unavailable
    or the SOCP fails to converge.

  filter_top_quintile_by_sector
    Pre-optimizer concentration filter — keeps the top X% of names per sector
    (max(min_per_sector, ⌈quintile · n⌉) per sector). Used to shrink the
    optimizer's variable space and to trade equal exposure for conviction.

The optimizer's SOCP form (cvxpy `cp.norm2`) avoids the QP `quad_form` path
that triggered numerical issues with PSD-validation in some cvxpy versions.
"""
from __future__ import annotations

from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd

# -----------------------------
# Risk & beta estimators
# -----------------------------

def estimate_beta(stock_rets: pd.DataFrame,
                  bench_rets: pd.Series,
                  lookback: int = 252) -> pd.Series:
    if stock_rets.empty or bench_rets.empty:
        return pd.Series(1.0, index=stock_rets.columns)
    idx = stock_rets.index.intersection(bench_rets.index)
    R = stock_rets.loc[idx].tail(lookback)
    m = bench_rets.loc[idx].tail(lookback)
    if R.empty or len(m) < 10 or m.var() == 0:
        return pd.Series(1.0, index=stock_rets.columns)
    var_m = float(m.var())
    betas = {}
    for col in R.columns:
        cov_im = float(R[col].cov(m)) if R[col].std(ddof=0) > 0 else 0.0
        b = cov_im / var_m if var_m > 0 else 1.0
        if not np.isfinite(b):
            b = 1.0
        betas[col] = b
    return pd.Series(betas).reindex(stock_rets.columns).fillna(1.0)

def diag_ann_variance(stock_rets: pd.DataFrame, lookback: int = 252) -> pd.Series:
    if stock_rets.empty:
        return pd.Series(0.10, index=[])
    R = stock_rets.tail(lookback)
    var = (R.var(ddof=0) * 252.0).replace([np.inf, -np.inf], np.nan)
    return var.fillna(0.10).clip(lower=1e-8)  # numeric floor

# -----------------------------
# Lazy cvxpy loader
# -----------------------------

def _get_cvxpy():
    try:
        import cvxpy as cp  # noqa: F401
        return cp
    except Exception:
        return None

# -----------------------------
# Main optimizer (SOCP risk)
# -----------------------------

def filter_top_quintile_by_sector(
    scores: pd.Series,
    sector: pd.Series,
    *,
    quintile: float = 0.20,
    min_per_sector: int = 3,
) -> pd.Index:
    """Return the index of names in the top `quintile` of `scores` within each sector.

    Guarantees `min_per_sector` names per sector when the sector has at least that
    many names — otherwise returns all of that sector's names.
    """
    keep: list = []
    quintile = float(np.clip(quintile, 1e-6, 1.0))
    min_per_sector = max(1, int(min_per_sector))
    for sec_name, group in sector.groupby(sector):
        members = group.index
        sec_scores = scores.reindex(members).dropna()
        if sec_scores.empty:
            continue
        n_total = len(sec_scores)
        n_keep = max(min_per_sector, int(math.ceil(quintile * n_total)))
        n_keep = min(n_keep, n_total)
        keep.extend(sec_scores.nlargest(n_keep).index.tolist())
    return pd.Index(keep)


def optimize_sector_matched(
    scores: pd.Series,
    sector: pd.Series,
    sector_targets: pd.Series,
    prev_w: Optional[pd.Series],
    stock_rets: pd.DataFrame,
    bench_rets: pd.Series,
    price: pd.Series,
    adv20: pd.Series,
    max_name: float,
    adv_frac: float,
    beta_target: float,
    beta_tol: float,
    lambda_risk: float,
    lambda_turnover: float,
    sector_band: float = 0.0,
) -> pd.Series:
    """
    Long-only optimizer enforcing sector weights and beta band, with name/%ADV caps
    and L1 turnover penalty. Uses SOCP risk term to avoid cvxpy quad_form path.
    Falls back to proportional-by-sector if cvxpy is unavailable.
    """
    tickers = list(scores.index)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)

    # Align inputs
    scores = scores.reindex(tickers).fillna(0.0)
    sector = sector.reindex(tickers)
    price = price.reindex(tickers).fillna(0.0)
    adv20 = adv20.reindex(tickers).fillna(0.0)
    prev = prev_w.reindex(tickers).fillna(0.0) if isinstance(prev_w, pd.Series) else pd.Series(0.0, index=tickers)

    # Risk & beta
    betas = estimate_beta(stock_rets[tickers], bench_rets).reindex(tickers).fillna(1.0)
    var_d = diag_ann_variance(stock_rets[tickers]).reindex(tickers).fillna(0.10).clip(lower=1e-8)
    sec = sector.reindex(var_d.index)
    var_d = var_d.groupby(sec).transform(lambda s: 0.5*s + 0.5*s.mean())
    std_d = np.sqrt(var_d.to_numpy(dtype=float))

    # Liquidity and name caps
    cap_liq = (adv_frac * adv20 * price).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    caps = pd.concat([cap_liq, pd.Series(max_name, index=tickers)], axis=1).min(axis=1).clip(lower=0.0, upper=max_name)

    # Try cvxpy
    cp = _get_cvxpy()
    if cp is None:
        return _fallback_proportional(scores, sector, sector_targets, caps)

    # Variables / data
    s = np.nan_to_num(scores.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(betas.to_numpy(dtype=float),  nan=1.0, posinf=1.0, neginf=1.0)
    w_prev = np.nan_to_num(prev.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    cap_vec = np.nan_to_num(caps.to_numpy(dtype=float), nan=max_name, posinf=max_name, neginf=0.0)
    std_vec = np.nan_to_num(std_d, nan=0.10, posinf=0.10, neginf=0.10)

    w = cp.Variable(n)

    # ---- SOCP risk penalty: lambda_risk * || diag(std) * w ||_2
    Dw = cp.multiply(std_vec, w)
    risk_pen = lambda_risk * cp.norm2(Dw)

    # Objective (concave): maximize s @ w - risk_pen - lambda_turnover * ||w - w_prev||_1
    obj = cp.Maximize(s @ w - risk_pen - lambda_turnover * cp.norm1(w - w_prev))

    cons = [w >= 0, cp.sum(w) == 1]

    # Sector weights — exact equality (band=0) or active band (band>0)
    band = float(max(0.0, sector_band))
    for sec_name, target in sector_targets.items():
        mask = (sector == sec_name).astype(float).to_numpy(dtype=float)
        if mask.sum() <= 0:
            continue
        sw = mask @ w
        if band <= 0:
            cons.append(sw == float(target))
        else:
            lo = max(0.0, float(target) - band)
            hi = min(1.0, float(target) + band)
            cons.append(sw >= lo)
            cons.append(sw <= hi)

    # Beta band
    cons += [b @ w >= beta_target - beta_tol, b @ w <= beta_target + beta_tol]

    # Per-name caps
    cons.append(w <= cap_vec)

    prob = cp.Problem(obj, cons)

    # Solve with cone solvers only (avoid QP → quad_form path)
    solved = False
    for solver_kwargs in (
        dict(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, feastol=1e-8, verbose=False),
        dict(solver=cp.SCS, eps=1e-5, verbose=False),
    ):
        try:
            prob.solve(**solver_kwargs)
            solved = w.value is not None and np.all(np.isfinite(w.value))
            if solved:
                break
        except Exception:
            continue

    if not solved:
        return _fallback_proportional(scores, sector, sector_targets, caps)

    sol = pd.Series(w.value, index=tickers).clip(lower=0.0)
    sm = float(sol.sum() or 1.0)
    sol = sol / sm
    # Enforce sector weights post-solve only when band==0 (exact equality).
    # When a band is active, we keep the optimizer's choice within the band.
    if band <= 0:
        for sec_name, target in sector_targets.items():
            mem = sol.index[sector == sec_name]
            if len(mem) == 0:
                continue
            sec_sum = float(sol.loc[mem].sum() or 1.0)
            sol.loc[mem] *= float(target) / sec_sum
    sol = sol.clip(lower=0.0)
    return sol / float(sol.sum() or 1.0)

# -----------------------------
# Fallback proportional allocator
# -----------------------------

def _fallback_proportional(
    scores: pd.Series,
    sector: pd.Series,
    sector_targets: pd.Series,
    caps: pd.Series,
) -> pd.Series:
    # Correct: keyword argument index=
    out = pd.Series(0.0, index=scores.index)
    for sec_name, tgt in sector_targets.items():
        mem = scores.index[sector == sec_name]
        raw = scores.reindex(mem).clip(lower=0.0)
        if raw.sum() <= 0:
            continue
        w = (raw / raw.sum()) * float(tgt)
        w = w.clip(upper=caps.reindex(w.index).fillna(1.0))
        w = (w / float(w.sum() or 1.0)) * float(tgt)
        out.loc[w.index] = w
    out = out.clip(lower=0.0)
    return out / float(out.sum() or 1.0)
