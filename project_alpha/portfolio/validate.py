# project_alpha/portfolio/validate.py
from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
import pandas as pd

__all__ = [
    "assert_daily_returns_hygiene",
    "align_for_ir",
    "assert_history_cutoff",
    "assert_no_lookahead",
    "annualized_kpis",
    "information_ratio",
    "beta_to_bench",
    "rolling_information_ratio",
    "performance_validation_report",
]

# ---------------------------
# Return hygiene & alignment
# ---------------------------

def assert_daily_returns_hygiene(rets: pd.Series) -> None:
    if rets.isna().any():
        n = int(rets.isna().sum())
        raise AssertionError(f"Found {n} NaNs in daily returns.")
    if (rets.abs() >= 0.75).any():
        bad = rets.loc[rets.abs() >= 0.75].head()
        raise AssertionError(f"Daily returns contain extreme values (>=75%). Example: {bad.to_dict()}")

def align_for_ir(port: pd.Series, bench: pd.Series) -> Tuple[pd.Series, pd.Series]:
    idx = port.index.intersection(bench.index)
    return port.loc[idx], bench.loc[idx]

# ---------------------------
# No look-ahead / cutoff guards
# ---------------------------

def assert_history_cutoff(history: pd.Index, cutoff: pd.Timestamp, label: str = "history") -> None:
    if len(history) == 0:
        return
    mx = pd.to_datetime(history.max())
    if mx > pd.to_datetime(cutoff):
        raise AssertionError(f"{label} exceeds cutoff {cutoff.date()} (max date = {mx.date()})")

def assert_no_lookahead(asof_date: pd.Timestamp,
                        feature_dates: pd.Series,
                        horizon_days: int) -> None:
    if feature_dates.empty:
        return
    bad = feature_dates[feature_dates > pd.to_datetime(asof_date)]
    if not bad.empty:
        raise AssertionError(
            f"Detected look-ahead: {len(bad)} tickers have feature_date > asof_date. "
            f"Example: {bad.head().to_dict()}"
        )

# ---------------------------
# KPIs & statistical tests
# ---------------------------

def annualized_kpis(rets: pd.Series) -> Dict[str, float]:
    if len(rets) < 10:
        return dict(CAGR=0.0, Vol=0.0, Sharpe=0.0, Sortino=0.0, MaxDD=0.0)
    vol = float(rets.std(ddof=0) * np.sqrt(252.0))
    cagr = float((1.0 + rets).prod() ** (252.0 / len(rets)) - 1.0)
    dn = rets[rets < 0]
    sortino = float((rets.mean() * 252.0) / (dn.std(ddof=0) * np.sqrt(252.0)) if dn.std(ddof=0) > 0 else 0.0)
    sharpe = float((rets.mean() * 252.0) / vol) if vol > 0 else 0.0
    cr = (1.0 + rets).cumprod()
    peak = cr.cummax()
    mdd = float((cr / peak - 1.0).min())
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, Sortino=sortino, MaxDD=mdd)

def information_ratio(port: pd.Series, bench: pd.Series) -> float:
    p, b = align_for_ir(port, bench)
    if len(p) < 10:
        return 0.0
    ex = p - b
    te = float(ex.std(ddof=0) * np.sqrt(252.0))
    return float((ex.mean() * 252.0) / te) if te > 0 else 0.0

def beta_to_bench(port: pd.Series, bench: pd.Series) -> float:
    """OLS beta of portfolio vs benchmark on aligned dates."""
    p, b = align_for_ir(port, bench)
    if len(p) < 10 or b.var(ddof=0) == 0:
        return 1.0
    cov = float(p.cov(b))
    return float(cov / float(b.var(ddof=0)))

# --- HAC / Newey–West t-stat for excess-mean & rolling IR ---

def _newey_west_tstat(excess: pd.Series, lags: int = 5) -> Tuple[float, float]:
    x = excess.dropna().astype(float).values
    n = x.size
    if n < lags + 5:
        return 0.0, 1.0
    mu = x.mean()
    x0 = x - mu
    gamma0 = np.dot(x0, x0) / n
    var_hac = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = np.dot(x0[:-k], x0[k:]) / n
        var_hac += 2.0 * w * cov
    var_mean = var_hac / n
    if var_mean <= 0:
        return 0.0, 1.0
    t = mu / np.sqrt(var_mean)
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
    return float(t), float(p)

def rolling_information_ratio(port: pd.Series, bench: pd.Series, window_days: int) -> pd.Series:
    p, b = align_for_ir(port, bench)
    ex = p - b
    return ex.rolling(window_days).apply(
        lambda s: s.mean() / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0,
        raw=False
    )

def nav_from_rets(r: pd.Series, start=1.0) -> pd.Series:
    return start * (1.0 + r.fillna(0.0)).cumprod()

def max_drawdown(r: pd.Series) -> float:
    nav = nav_from_rets(r)
    peak = nav.cummax()
    dd = (nav / peak - 1.0).min()
    return float(dd)

def sortino_ratio(r: pd.Series, rf: float = 0.0) -> float:
    dr = (r - rf/252.0).clip(upper=0)  # negative or zero
    downside = float(dr.std(ddof=0) * np.sqrt(252.0))
    downside = max(downside, 1e-6)     # floor to avoid explosion
    mu = float((r.mean() - rf/252.0) * 252.0)
    return mu / downside

def performance_validation_report(
    port: pd.Series,
    benches: Dict[str, pd.Series],
    conf_level: float = 0.95
) -> pd.DataFrame:
    rows = []
    kpi = annualized_kpis(port)
    rows.append({
        "Benchmark": "—",
        "Alpha_CAGR": kpi["CAGR"],
        "Alpha_Vol": kpi["Vol"],
        "Alpha_Sharpe": kpi["Sharpe"],
        "Alpha_Sortino": kpi["Sortino"],
        "Alpha_MaxDD": kpi["MaxDD"],
        "IR": np.nan,
        "Excess_Mean_daily": np.nan,
        "NW_tstat": np.nan,
        "NW_pvalue": np.nan,
        "Beats?": np.nan,
    })

    for name, b in benches.items():
        p, bench = align_for_ir(port, b)
        ex = (p - bench).dropna()
        if ex.empty:
            rows.append({
                "Benchmark": name, "Alpha_CAGR": np.nan, "Alpha_Vol": np.nan,
                "Alpha_Sharpe": np.nan, "Alpha_Sortino": np.nan, "Alpha_MaxDD": np.nan,
                "IR": np.nan, "Excess_Mean_daily": np.nan, "NW_tstat": np.nan,
                "NW_pvalue": np.nan, "Beats?": False,
            })
            continue
        ir = information_ratio(p, bench)
        t, pval = _newey_west_tstat(ex, lags=5)
        beats = bool((ir > 0) and (pval < (1.0 - conf_level)))
        rows.append({
            "Benchmark": name,
            "Alpha_CAGR": kpi["CAGR"],
            "Alpha_Vol": kpi["Vol"],
            "Alpha_Sharpe": kpi["Sharpe"],
            "Alpha_Sortino": kpi["Sortino"],
            "Alpha_MaxDD": kpi["MaxDD"],
            "IR": ir,
            "Excess_Mean_daily": float(ex.mean()),
            "NW_tstat": t,
            "NW_pvalue": pval,
            "Beats?": beats,
        })
    return pd.DataFrame(rows)
