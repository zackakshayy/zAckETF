# project_alpha/reporting/calendar_returns.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------
def _log(msg: str) -> None:
    print(f"[calendar_returns] {msg}")

def _get_cfg(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        _log(f"cfg not found: {p}")
        return {}
    with p.open("r") as f:
        raw = yaml.safe_load(f) or {}
    # flatten common places we’ve used before
    recs = (raw.get("recs") or {}).copy()
    if "csv" in recs:
        # allow someone to pass one file with annual percent returns
        recs.setdefault("returns_percent_csv", recs["csv"])
    return raw

def _ensure_plots_dir(reports_dir: Path) -> Path:
    """
    Create <reports_dir>/plots, but guard against placeholder paths like '/Users/...'.
    """
    rd = Path(reports_dir).expanduser()
    # Friendly check for placeholders
    if "..." in str(rd):
        raise ValueError(
            f"`--reports-dir` looks like a placeholder: {rd}\n"
            "Please pass a real absolute path, e.g. "
            '"/Users/<you>/Desktop/ProjectX_MasterData/_px_reports".'
        )
    # Ensure parent exists (create if missing)
    rd.mkdir(parents=True, exist_ok=True)
    plots = rd / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    return plots

def _read_csv_guess_dates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to detect a date-like column
    # Common options: date, Date, DATE, year, Year
    dcol = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"date", "asof_date", "asof", "dt"}:
            dcol = c
            break
        if cl in {"year", "yr"}:
            dcol = c
            break
    if dcol is None:
        # Try index if it looks like dates
        try:
            idx_try = pd.to_datetime(df.index, errors="coerce")
            if idx_try.notna().mean() > 0.9:
                df = df.copy()
                df.insert(0, "_date", idx_try)
                dcol = "_date"
        except Exception:
            pass
    if dcol is None:
        # Last resort: first column
        dcol = df.columns[0]

    # Build DatetimeIndex; handle 'year' specially -> map to Dec-31
    if str(dcol).strip().lower() in {"year", "yr"}:
        years = pd.to_numeric(df[dcol], errors="coerce").astype("Int64")
        idx = pd.to_datetime(years.astype(str) + "-12-31", errors="coerce")
    else:
        idx = pd.to_datetime(df[dcol], errors="coerce")

    df = df.copy()
    df.index = idx
    df = df[~df.index.isna()]
    return df

def _returns_from_daily(df: pd.DataFrame) -> pd.Series:
    """
    Accepts a DataFrame with daily returns in *any* of these columns:
    ['return','ret','daily_return','alpha_daily_returns','bench_daily_returns', 'ALPHA_DAILY_RETURNS','BENCH_DAILY_RETURNS'].
    """
    cand_cols = [
        "return", "ret", "daily_return",
        "alpha_daily_returns", "bench_daily_returns",
        "ALPHA_DAILY_RETURNS", "BENCH_DAILY_RETURNS",
        "returns", "Returns"
    ]
    cols = [c for c in df.columns if str(c).strip() in cand_cols]
    if not cols:
        # If nothing obvious, try any numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            cols = [num_cols[0]]
    if not cols:
        raise ValueError("Could not identify returns column in daily dataframe.")
    s = pd.Series(df[cols[0]].values, index=pd.to_datetime(df.index), name=cols[0])
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.astype(float)

def _returns_from_annual_percent(df: pd.DataFrame) -> pd.Series:
    """
    Accepts a dataframe that has Year + percentage returns (e.g. 12.34 for 12.34%).
    """
    # Candidates for the value column
    vcols = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"return", "returns", "total_return", "nav_return", "pct", "percent"}:
            vcols.append(c)
    if not vcols:
        # last numeric column as last resort
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            vcols = [num_cols[-1]]
    if not vcols:
        raise ValueError("Could not determine annual return column.")

    # Year handling already normalized in _read_csv_guess_dates
    s = pd.Series(df[vcols[0]].values, index=pd.to_datetime(df.index), name=vcols[0])
    # Convert percent -> decimal if it looks like percent scale
    if s.abs().max() > 1.0:
        s = s / 100.0
    # Keep one point per calendar year @ Dec-31
    s = s.groupby(s.index.year).last()
    s.index = pd.to_datetime(s.index.astype(str) + "-12-31")
    return s.astype(float)

def _to_calendar_year(s: pd.Series) -> pd.Series:
    """
    Convert daily (or higher frequency) returns into calendar-year total returns.
    """
    if s.empty:
        return s
    s = s.dropna()
    if s.index.inferred_type in {"datetime64", "datetime64tz"}:
        by_year = s.groupby(s.index.year)
        # convert to total return: prod(1+r) - 1
        cal = by_year.apply(lambda x: (1.0 + x).prod() - 1.0)
        cal.index = pd.to_datetime(cal.index.astype(str) + "-12-31")
        return cal
    # Already annual (year-end) series
    return s

def _restrict_years(s: pd.Series, start_year: int = 2019, end_year: int = 2025) -> pd.Series:
    if s.empty:
        return s
    mask = (s.index.year >= start_year) & (s.index.year <= end_year)
    out = s.loc[mask]
    # ensure we have an entry for each year, even if NaN (helps align bars)
    idx = pd.to_datetime(pd.Index(range(start_year, end_year + 1)).astype(str) + "-12-31")
    return out.reindex(idx)

def _find_first_existing(base: Path, names: Iterable[str]) -> Optional[Path]:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


# ----------------------------
# Readers (Alpha / R1000 / RECS)
# ----------------------------
def _read_alpha_returns(reports_dir: Path) -> pd.Series:
    """
    Try several conventions that the backtest may have written.
    """
    candidates = [
        "alpha_returns.csv",
        "ALPHA_DAILY_RETURNS.csv",
        "alpha_vs_bench_validation.csv",  # has ALPHA_DAILY_RETURNS column
    ]
    p = _find_first_existing(reports_dir, candidates)
    if p is None:
        raise FileNotFoundError(f"Alpha returns not found in {reports_dir}. Tried: {candidates}")
    df = _read_csv_guess_dates(p)
    if p.name == "alpha_vs_bench_validation.csv":
        col = "ALPHA_DAILY_RETURNS"
        if col in df.columns:
            s = pd.Series(df[col].values, index=df.index)
        else:
            s = _returns_from_daily(df)
    else:
        s = _returns_from_daily(df)
    return s.rename("Alpha")

def _read_r1000_returns(reports_dir: Path, cfg: dict) -> pd.Series:
    """
    Read Russell 1000 returns written by backtest (preferred), else
    try any of the common bench files.
    """
    # Users often have this file from the backtest validation step:
    candidates = [
        "bench_returns.csv",
        "BENCH_DAILY_RETURNS.csv",
        "alpha_vs_bench_validation.csv",  # has BENCH_DAILY_RETURNS column
        "r1000_returns.csv",
    ]
    p = _find_first_existing(reports_dir, candidates)
    if p is None:
        # Final fallback: attempt to derive from R1000.xlsx is out of scope for this plotting utility.
        raise FileNotFoundError(
            f"Russell 1000 daily returns not found in {reports_dir}. "
            f"Tried: {candidates}. Please write a 'bench_returns.csv' with columns [date, return]."
        )
    df = _read_csv_guess_dates(p)
    if p.name == "alpha_vs_bench_validation.csv" and "BENCH_DAILY_RETURNS" in df.columns:
        s = pd.Series(df["BENCH_DAILY_RETURNS"].values, index=df.index)
    else:
        s = _returns_from_daily(df)
    return s.rename("R1000")

def _read_recs_returns(cfg: dict) -> pd.Series:
    """
    Prefer the user-provided annual returns CSV (percent or decimal).
    Otherwise, if a daily returns CSV is specified, use it.
    Config keys we support:
      recs:
        returns_percent_csv: <annual percent>
        returns_csv:         <annual decimals or daily>
    """
    recs_cfg = (cfg.get("recs") or {})
    p_ann = recs_cfg.get("returns_percent_csv") or recs_cfg.get("annual_returns_csv")
    p_any = recs_cfg.get("returns_csv")

    if p_ann:
        p = Path(p_ann).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"RECS annual returns CSV not found: {p}")
        df = _read_csv_guess_dates(p)
        s = _returns_from_annual_percent(df)
        return s.rename("RECS")

    if p_any:
        p = Path(p_any).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"RECS returns CSV not found: {p}")
        df = _read_csv_guess_dates(p)
        # Try daily first; if it fails, treat as annual percent
        try:
            s = _returns_from_daily(df)
        except Exception:
            s = _returns_from_annual_percent(df)
        return s.rename("RECS")

    raise FileNotFoundError(
        "RECS returns not configured. Put paths under `recs:` in alpha.yaml "
        "e.g. recs.returns_percent_csv: /path/to/recs_returns_percent.csv"
    )


# ----------------------------
# Plot helpers
# ----------------------------
def _plot_calendar_bars(out_png: Path, series_map: Dict[str, pd.Series]) -> None:
    """Bar chart for calendar-year returns (2019–2025)."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    # Build a DataFrame with aligned years
    years = pd.to_datetime(pd.Index(range(2019, 2026)).astype(str) + "-12-31")
    df = pd.DataFrame(index=years)
    for name, s in series_map.items():
        df[name] = s.values

    # Plot grouped bars
    width = 0.8 / max(1, len(series_map))  # total ~0.8
    x = np.arange(len(df.index))
    for i, (name, _) in enumerate(series_map.items()):
        bar_pos = x + (i - (len(series_map) - 1) / 2.0) * width
        ax.bar(bar_pos, df[name].values, width=width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels([str(d.year) for d in df.index], rotation=45)
    ax.set_ylabel("Return")
    ax.set_title("Calendar-Year Returns (2019–2025)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    _log(f"Saved: {out_png}")


# ----------------------------
# Entry point
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Create calendar_year_returns.png for Alpha vs R1000 vs RECS (2019–2025).")
    ap.add_argument("--reports-dir", required=True, help="Folder with Alpha/bench outputs (e.g. _px_reports)")
    ap.add_argument("--config", default="", help="alpha.yaml containing RECS CSV paths")
    # Optional overrides (rarely needed if reports_dir + yaml are correct)
    ap.add_argument("--alpha-returns-csv", default="", help="Override Alpha daily returns CSV path")
    ap.add_argument("--bench-returns-csv", default="", help="Override Russell 1000 daily returns CSV path")
    ap.add_argument("--recs-returns-csv", default="", help="Override RECS returns CSV (annual percent or daily)")
    args = ap.parse_args()

    reports_dir = Path(args.reports_dir).expanduser()
    plots_dir = _ensure_plots_dir(reports_dir)
    cfg = _get_cfg(args.config)

    # --- Alpha ---
    if args.alpha_returns_csv:
        alpha_df = _read_csv_guess_dates(Path(args.alpha_returns_csv))
        alpha_daily = _returns_from_daily(alpha_df).rename("Alpha")
    else:
        alpha_daily = _read_alpha_returns(reports_dir)

    # --- R1000 ---
    if args.bench_returns_csv:
        bench_df = _read_csv_guess_dates(Path(args.bench_returns_csv))
        r1000_daily = _returns_from_daily(bench_df).rename("R1000")
    else:
        r1000_daily = _read_r1000_returns(reports_dir, cfg)

    # --- RECS ---
    if args.recs_returns_csv:
        recs_df = _read_csv_guess_dates(Path(args.recs_returns_csv))
        try:
            recs_any = _returns_from_daily(recs_df)
        except Exception:
            recs_any = _returns_from_annual_percent(recs_df)
        recs_any = recs_any.rename("RECS")
    else:
        recs_any = _read_recs_returns(cfg)

    # Convert to calendar-year total returns
    alpha_cal = _restrict_years(_to_calendar_year(alpha_daily))
    r1000_cal = _restrict_years(_to_calendar_year(r1000_daily))
    recs_cal = _restrict_years(_to_calendar_year(recs_any))

    # Pack and write
    ser_map = {"Alpha": alpha_cal, "R1000": r1000_cal, "RECS": recs_cal}
    out = plots_dir / "calendar_year_returns.png"
    _plot_calendar_bars(out, ser_map)

if __name__ == "__main__":
    main()
