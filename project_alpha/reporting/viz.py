# project_alpha/reporting/viz.py
# -*- coding: utf-8 -*-
"""
Visualization helper for Alpha vs Benchmarks.

- Reads Alpha returns/NAV from the reports directory.
- Reads benchmarks ONLY from explicit paths in alpha.yaml (no alias guessing).
- Robust to empty / sparse series (annual or monthly RECS is fine).
- Produces:
    plots/annual_returns.png
    plots/cum_nav.png
    plots/growth_1m.png
    plots/rolling_sharpe_252.png
    plots/kpi_bars.png
  And saves KPI table to viz_kpis.csv

CLI:
  python -m project_alpha.reporting.viz \
     --config "project_alpha_code_v2_1/config/alpha.yaml" \
     --reports-dir "/Users/.../_px_reports"
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # We'll run with defaults if yaml is missing


# --------------------------------------------------------------------------------------
# I/O and config
# --------------------------------------------------------------------------------------

def get_cfg(cfg_path: Optional[str]) -> Dict:
    """Load YAML config if present; otherwise return empty dict."""
    if not cfg_path:
        return {}
    p = Path(cfg_path).expanduser()
    if not p.exists():
        return {}
    if yaml is None:
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_any_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel into DataFrame without date parsing."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _pick_date_and_value_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Try to infer a date-like column and a value-like column."""
    # Candidate date columns (case-insensitive):
    date_candidates = [
        "date", "as_of", "asof", "period", "year", "month", "year_end", "calendar_year"
    ]
    value_candidates = [
        # returns
        "return", "returns", "total_return", "total returns", "nav_return",
        "alpha_daily_returns", "bench_daily_returns",
        # nav / index
        "nav", "index", "value", "price", "close",
        # percent fields
        "return_%", "returns_%", "total_return_%"
    ]

    cols = {c.lower(): c for c in df.columns}

    # date col
    dcol = None
    for k in date_candidates:
        if k in cols:
            dcol = cols[k]
            break
    if dcol is None:
        # try first object/numeric that looks like dates/years
        for c in df.columns:
            if str(c).lower().startswith("unnamed"):
                continue
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                dcol = c
                break
            # all-int years?
            if pd.api.types.is_integer_dtype(s) and s.between(1900, 2100).all():
                dcol = c
                break
    if dcol is None:
        # last resort: first column
        dcol = df.columns[0]

    # value col
    vcol = None
    for k in value_candidates:
        if k in cols:
            vcol = cols[k]
            break
    if vcol is None:
        # take the first numeric column that's not the date col
        numeric = [c for c in df.columns if c != dcol and pd.api.types.is_numeric_dtype(df[c])]
        if numeric:
            vcol = numeric[0]
        else:
            # fallback: second column
            vcol = df.columns[min(1, len(df.columns) - 1)]
    return dcol, vcol


def _series_from_table(path: Path) -> pd.Series:
    """
    Turn a table into a pd.Series with a DatetimeIndex.

    - If the 'date' column seems to be YEAR, index becomes Dec-31 of that year.
    - Value column is coerced to float; strings with '%' are handled.
    """
    df = _read_any_table(path)
    if df.empty:
        return pd.Series(dtype=float, name=path.name)

    dcol, vcol = _pick_date_and_value_columns(df)

    # value coercion (handles percent strings)
    vals = df[vcol]
    if vals.dtype == object:
        vals = vals.astype(str).str.strip()
        vals = vals.str.replace("%", "", regex=False)
    vals = pd.to_numeric(vals, errors="coerce")

    # build datetime index
    if str(dcol).lower() == "year":
        years = pd.to_numeric(df[dcol], errors="coerce").astype("Int64")
        idx = pd.to_datetime(years.astype(str) + "-12-31", errors="coerce")
    else:
        idx = pd.to_datetime(df[dcol], errors="coerce")

    s = pd.Series(vals.values, index=idx, name=os.path.basename(path))
    s = s.sort_index()
    s = s[~s.index.isna()]
    # drop duplicate dates (keep first)
    if not s.index.is_unique:
        s = s[~s.index.duplicated(keep="first")]
    return s


# --------------------------------------------------------------------------------------
# Time series transforms
# --------------------------------------------------------------------------------------

def _looks_like_returns(s: pd.Series) -> bool:
    """Heuristic: daily (or period) returns generally small around 0."""
    if s is None or len(s) == 0:
        return False
    ss = s.dropna()
    if ss.empty:
        return False
    return ss.abs().median() < 0.02 and abs(ss.mean()) < 0.01


def returns_to_nav(r: pd.Series, start_val: float = 1.0) -> pd.Series:
    if r is None or r.empty:
        return pd.Series(dtype=float, name=(r.name if r is not None else "NAV"))
    nav = start_val * (1.0 + r.fillna(0.0)).cumprod()
    nav.name = (r.name or "series") + "_NAV"
    return nav


def nav_to_returns(nav: pd.Series) -> pd.Series:
    if nav is None or nav.empty:
        return pd.Series(dtype=float, name=(nav.name if nav is not None else "RET"))
    r = nav.pct_change().fillna(0.0)
    r.name = (nav.name or "series") + "_RET"
    return r


def ensure_nav(s: pd.Series) -> pd.Series:
    """Ensure Series is a NAV/index series."""
    if s is None or s.empty:
        return pd.Series(dtype=float)
    s = s.dropna().sort_index()
    if _looks_like_returns(s):
        return returns_to_nav(s)
    return s


def ensure_returns(s: pd.Series) -> pd.Series:
    """Ensure Series is a returns series."""
    if s is None or s.empty:
        return pd.Series(dtype=float)
    s = s.dropna().sort_index()
    if _looks_like_returns(s):
        return s
    return nav_to_returns(s)

def _scan_alpha_series(reports_dir: Path) -> tuple[pd.Series, pd.Series]:
     """
     Look inside reports_dir for any files that *look* like alpha returns/NAV.
     Heuristics:
       - filenames containing 'alpha' and 'return' -> returns
       - filenames containing 'alpha' and 'nav'    -> nav
       - special case: alpha_vs_bench_validation.csv (uses ALPHA_DAILY_RETURNS)
     """
     def _find(pattern_words: list[str]) -> Optional[Path]:
         words = [w.lower() for w in pattern_words]
         for p in list(reports_dir.glob("*.csv")) + list(reports_dir.glob("*.xlsx")):
             name = p.name.lower()
             if all(w in name for w in words):
                 return p
         return None
 
     r = pd.Series(dtype=float)
     nav = pd.Series(dtype=float)
 
     # direct name scans
     rp = _find(["alpha", "return"])
     if rp:
         r = _series_from_table(rp)
     npth = _find(["alpha", "nav"])
     if npth:
         nav = _series_from_table(npth)
 
     # special fallback: alpha_vs_bench_validation.csv
     av = reports_dir / "alpha_vs_bench_validation.csv"
     if r.empty and av.exists():
         try:
             df = _read_any_table(av)
             if "ALPHA_DAILY_RETURNS" in df.columns:
                 r = _series_from_table(av.rename("ALPHA_DAILY_RETURNS"))  # type: ignore
                 r = pd.Series(df["ALPHA_DAILY_RETURNS"].values,
                               index=pd.to_datetime(df.iloc[:, 0], errors="coerce"),
                               name="Alpha_returns")
         except Exception:
             pass
     return r, nav

def annualize_calendar(returns: pd.Series) -> pd.Series:
    """
    Calendar-year total returns from period returns; works for daily/monthly/annual.
    """
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    r = returns.dropna().sort_index()
    # Resample to year-end and compound within the year: (1+r).prod() - 1
    g = r.resample("YE").apply(lambda x: (1.0 + x).prod() - 1.0)
    g.name = (r.name or "series") + "_ANNUAL"
    return g


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------

def read_alpha_series(reports_dir: Path, cfg: Dict) -> Tuple[pd.Series, pd.Series]:
    """
    Load Alpha daily returns and NAV. We accept:
      - alpha_returns.csv or alpha_returns.xlsx
      - alpha_nav.csv or alpha_nav.xlsx
    If only one is present, derive the other.
    """
    # 0) explicit paths from YAML (preferred)
    acfg = (cfg or {}).get("alpha", {})
    r = _read_optional_series(acfg.get("returns_csv"))
    nav = _read_optional_series(acfg.get("nav_csv"))

    # 1) fixed filenames in reports dir
    if r.empty:
        for p in [reports_dir / "alpha_returns.csv",
                reports_dir / "Alpha_returns.csv",
                reports_dir / "alpha_returns.xlsx"]:
            if p.exists():
                r = _series_from_table(p); break
    if nav.empty:
        for p in [reports_dir / "alpha_nav.csv",
                reports_dir / "Alpha_NAV.csv",
                reports_dir / "alpha_nav.xlsx"]:
            if p.exists():
                nav = _series_from_table(p); break

    # 2) heuristic scan in reports_dir
    if r.empty and nav.empty:
        r, nav = _scan_alpha_series(reports_dir)

    # 3) derive missing side if other exists; else warn but don't crash
    if r.empty and not nav.empty:
        r = ensure_returns(nav)
    if nav.empty and not r.empty:
        nav = ensure_nav(r)

    if r.empty and nav.empty:
        print(f"[viz] WARNING: Alpha returns/NAV not found in {reports_dir}. "
            f"Set alpha.returns_csv / alpha.nav_csv in alpha.yaml.")
        return pd.Series(dtype=float, name="Alpha"), pd.Series(dtype=float, name="Alpha_NAV")
 
    return r.dropna().sort_index(), nav.dropna().sort_index()


def _read_optional_series(path_str: Optional[str]) -> pd.Series:
    if not path_str:
        return pd.Series(dtype=float)
    p = Path(path_str).expanduser()
    if not p.exists():
        return pd.Series(dtype=float)
    return _series_from_table(p)


def read_bench_series(reports_dir: Path, cfg: Dict) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """
    Read configured benchmarks. No alias guessing.
    Expected config structure (examples):

    benchmarks:
      r1000:
        returns_csv: "/abs/path/r1000_returns.csv"   # daily or monthly/annual OK
        nav_csv:     "/abs/path/r1000_nav.csv"       # optional if returns present
      recs:
        returns_csv: "/abs/path/recs_returns.csv"    # your manually prepared CSVs
        returns_percent_csv: "/abs/path/recs_returns_percent.csv"  # optional
        nav_csv:     "/abs/path/recs_nav.csv"        # optional
    """
    out: Dict[str, Tuple[pd.Series, pd.Series]] = {}
    bcfg = (cfg or {}).get("benchmarks", {})

    def _ingest(name: str, d: Dict) -> None:
        if not isinstance(d, dict):
            return
        # try returns in preference order
        r = _read_optional_series(d.get("returns_csv"))
        if r.empty:
            r = _read_optional_series(d.get("returns_percent_csv"))
            # if percent is in whole numbers, convert (e.g., 12.3 -> 0.123)
            if not r.empty and r.abs().median() > 1.0:
                r = r / 100.0
            # returns_percent often annual; OK
        nav = _read_optional_series(d.get("nav_csv"))
        if r.empty and nav.empty:
            return
        if r.empty:
            r = ensure_returns(nav)
        if nav.empty:
            nav = ensure_nav(r)
        out[name.upper()] = (r.dropna().sort_index(), nav.dropna().sort_index())

    for key, d in bcfg.items():
        _ingest(key, d)

    return out


# --------------------------------------------------------------------------------------
# KPIs and plotting
# --------------------------------------------------------------------------------------

def kpis_from_returns(r: pd.Series) -> Dict[str, float]:
    """Compute KPIs on daily (or period) returns."""
    if r is None or r.empty:
        return {k: np.nan for k in ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD"]}

    r = r.dropna().sort_index()
    # Annualization factor: infer by median spacing (days)
    if len(r.index) > 1:
        dt = np.median(np.diff(r.index.values).astype("timedelta64[D]").astype(int))
        if dt <= 1:
            ann = 252.0  # daily
        elif dt <= 31:
            ann = 12.0   # monthly
        else:
            ann = 1.0    # annual
    else:
        ann = 252.0

    nav = (1.0 + r).cumprod()
    total = nav.iloc[-1] / nav.iloc[0] if len(nav) > 0 else np.nan
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    cagr = total ** (1.0 / years) - 1.0 if pd.notna(total) else np.nan

    vol = r.std() * np.sqrt(ann)
    sharpe = (r.mean() * ann) / (vol if vol != 0 else np.nan)
    downside = r[r < 0].std() * np.sqrt(ann)
    sortino = (r.mean() * ann) / (downside if downside != 0 else np.nan)

    dd = (nav / nav.cummax() - 1.0).min() if len(nav) else np.nan

    return {"CAGR": float(cagr), "Vol": float(vol), "Sharpe": float(sharpe),
            "Sortino": float(sortino), "MaxDD": float(dd)}


def plot_annual_bars(ax, annual_map: Dict[str, pd.Series]):
    years = sorted(set().union(*[set(s.index) for s in annual_map.values() if not s.empty]))
    if not years:
        ax.set_visible(False)
        return
    x = np.arange(len(years))
    width = 0.8 / max(len(annual_map), 1)
    for i, (name, s) in enumerate(annual_map.items()):
        s = s.groupby(s.index).last()  # ensure unique year index
        yvals = []
        for y in years:
            v = s.get(y, np.nan)
            if isinstance(v, pd.Series):
                v = v.iloc[0] if not v.empty else np.nan
            yvals.append(float(v) if pd.notna(v) else np.nan)
        ax.bar(x + i * width, yvals, width, label=name)
    ax.set_title("Calendar-Year Returns")
    ax.set_xticks(x + width * (len(annual_map) - 1) / 2)
    ax.set_xticklabels([str(y) for y in years], rotation=90)
    ax.set_ylabel("Return")
    ax.legend()


def plot_cum_nav(ax, idx_map: Dict[str, pd.Series]):
    for name, idx in idx_map.items():
        idx = ensure_nav(idx)
        idx = idx.dropna().sort_index()
        if idx.empty:
            continue
        ax.plot(idx.index, (idx / idx.iloc[0]).values, label=name)
    ax.set_title("Cumulative Growth (normalized to 1)")
    ax.set_ylabel("Index Level")
    ax.legend()


def plot_growth_1m(ax, idx_map: Dict[str, pd.Series]):
    for name, idx in idx_map.items():
        idx = ensure_nav(idx)
        idx = idx.dropna().sort_index()
        if idx.empty:
            continue
        ax.plot(idx.index, 1_000_000.0 * (idx / idx.iloc[0]).values, label=name, linewidth=1.5)
    ax.set_title("Growth of $1,000,000 Since Start")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend()


def plot_rolling_sharpe(ax, ret_map: Dict[str, pd.Series], window: int = 252):
    for name, r in ret_map.items():
        r = ensure_returns(r).dropna().sort_index()
        if r.empty or len(r) < window:
            continue
        roll = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252.0)
        ax.plot(roll.index, roll.values, label=name)
    ax.set_title(f"Rolling Sharpe (window={window} trading days)")
    ax.set_ylabel("Sharpe")
    ax.legend()


def plot_kpi_bars(ax, kpi_map: Dict[str, Dict[str, float]]):
    keys = list(kpi_map.keys())
    metrics = ["Sharpe", "Sortino", "CAGR", "Vol", "MaxDD"]
    width = 0.1
    x = np.arange(len(keys))
    for i, m in enumerate(metrics):
        vals = [kpi_map[k].get(m, np.nan) for k in keys]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_title("Performance KPIs")
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(keys, rotation=0)
    ax.set_ylabel("Value (annualized where applicable)")
    ax.legend()


# --------------------------------------------------------------------------------------
# Main chart maker
# --------------------------------------------------------------------------------------

def make_all_charts(cfg_path: Optional[str], reports_dir_str: str) -> None:
    cfg = get_cfg(cfg_path)
    reports_dir = Path(reports_dir_str).expanduser()
    plots_dir = reports_dir / "plots"
    ensure_dir(plots_dir)
    print(f"open {plots_dir}")

    # --- Alpha ---
    alpha_ret, alpha_nav = read_alpha_series(reports_dir, cfg)

    # --- Benchmarks from config only ---
    bench = read_bench_series(reports_dir, cfg)  # name -> (returns, nav)

    # Build maps for plotting
    idx_map: Dict[str, pd.Series] = {"Alpha": alpha_nav}
    ret_map: Dict[str, pd.Series] = {"Alpha": alpha_ret}
    for name, (r, nav) in bench.items():
        idx_map[name] = nav
        ret_map[name] = r

    # Annual returns
    annual_map: Dict[str, pd.Series] = {"Alpha": annualize_calendar(alpha_ret)}
    for name, (r, nav) in bench.items():
        rr = r if not r.empty else ensure_returns(nav)
        annual_map[name] = annualize_calendar(rr)

    # 1) Annual bars
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_annual_bars(ax, annual_map)
    fig.tight_layout()
    fig.savefig(plots_dir / "annual_returns.png", dpi=140)
    plt.close(fig)

    # 2) Cumulative nav
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_cum_nav(ax, idx_map)
    fig.tight_layout()
    fig.savefig(plots_dir / "cum_nav.png", dpi=140)
    plt.close(fig)

    # 3) Growth of $1,000,000
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_growth_1m(ax, idx_map)
    fig.tight_layout()
    fig.savefig(plots_dir / "growth_1m.png", dpi=140)
    plt.close(fig)

    # 4) Rolling Sharpe
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_rolling_sharpe(ax, ret_map, window=252)
    fig.tight_layout()
    fig.savefig(plots_dir / "rolling_sharpe_252.png", dpi=140)
    plt.close(fig)

    # 5) KPI bars + table
    kpi_map: Dict[str, Dict[str, float]] = {}
    for name, r in ret_map.items():
        kpi_map[name] = kpis_from_returns(ensure_returns(r))
    kpi_df = pd.DataFrame(kpi_map).T
    kpi_df.index.name = "Which"
    kpi_df.to_csv(reports_dir / "viz_kpis.csv", float_format="%.6f")

    fig, ax = plt.subplots(figsize=(13, 4))
    plot_kpi_bars(ax, kpi_map)
    fig.tight_layout()
    fig.savefig(plots_dir / "kpi_bars.png", dpi=140)
    plt.close(fig)

    # sanity dump for debugging
    bench_list = list(bench.keys())
    print(f"[viz] Benchmarks discovered: {bench_list if bench_list else '[]'}")
    print(f"Charts saved to: {plots_dir}")
    print(f"KPI table saved to: {reports_dir / 'viz_kpis.csv'}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha vs Benchmarks visualization")
    ap.add_argument("--config", type=str, default=None, help="Path to alpha.yaml")
    ap.add_argument(
        "--reports-dir",
        type=str,
        default=str(Path.home() / "Desktop/_px_reports"),
        help="Folder with alpha_* files and where plots will be saved",
    )
    args = ap.parse_args()
    make_all_charts(args.config, args.reports_dir)


if __name__ == "__main__":
    main()
