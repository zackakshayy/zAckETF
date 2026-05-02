# project_alpha/reporting/synthetic_showcase.py
# SPDX-License-Identifier: MIT
"""
Synthetic showcase module:
- Uses actual Russell 1000 information from R1000.xlsx and RECS CSVs you provide
- Builds a daily benchmark return series (from CSV if available; else from R1000.xlsx MV by rebalance)
- Expands annual RECS returns to business-daily
- Creates a realistic synthetic Alpha daily series that outperforms bench by ~1% annually
- Produces plots (calendar year bars, cumulative NAV, growth of $1M) and a small KPI table
- Emits a constituents table (top of RECS, Russell, Alpha) as of Dec-2024 (fallback: latest available in 2024)

CLI example:
python -m project_alpha.reporting.synthetic_showcase \
  --reports-dir "/Users/.../_px_reports" \
  --r1000-xlsx "/Users/.../R1000.xlsx" \
  --recs-returns-csv "/Users/.../RECS/recs_returns_percent.csv" \
  --recs-top-holdings "/Users/.../RECS/recs_top_holdings_2025-06-30.csv"

Outputs (under <reports-dir>/plots):
- calendar_year_returns_synth.png
- cum_nav_synth.png
- growth_1m_synth.png
- kpi_bars_synth.png

Outputs (under <reports-dir>):
- top_constituents_2024.csv
- showcase_kpis.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------- IO / FS helpers ----------------------------

def _ensure_plots_dir(reports_dir: Path) -> Path:
    reports_dir = Path(reports_dir).expanduser()
    plots = reports_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    print(f"open {plots}")
    return plots


# ---------------------------- Robust readers ----------------------------

def _read_csv_guess(path: Path) -> pd.DataFrame:
    """
    Read CSV and robustly choose a datetime-like index:
      1) common date/Year header aliases
      2) auto-scan: first col that parses to datetime for ≥50% rows
      3) 'Unnamed: 0' (common saved index)
      4) first column
    If the chosen column is "Year-like" (1900–2100), map to Dec-31 of that year.
    Always assigns an index or raises a clear error.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    cols = list(df.columns)

    aliases = [
        "date", "Date", "DATE",
        "asof_date", "as_of", "asof", "AsOf", "AsOfDate", "ASOF_DATE",
        "timestamp", "Timestamp", "TIMESTAMP",
        "dt", "DT", "time", "TIME",
        "trade_date", "TradeDate", "TRADE_DATE",
        "nav_date", "NAV_DATE",
        "Year", "year", "YR", "yr", "fiscal_year", "FiscalYear",
        "Unnamed: 0",
    ]
    dcol = next((c for c in aliases if c in df.columns), None)

    if dcol is None:
        best = None
        best_ratio = 0.0
        for c in cols:
            try:
                ts = pd.to_datetime(df[c], errors="coerce")
                ratio = float(ts.notna().mean())
                if ratio >= 0.50 and ratio > best_ratio:
                    best, best_ratio = c, ratio
            except Exception:
                continue
        dcol = best

    if dcol is None and "Unnamed: 0" in df.columns:
        dcol = "Unnamed: 0"

    if dcol is None and len(cols) > 0:
        dcol = cols[0]

    if dcol is None:
        raise ValueError(f"Could not find a date/Year column in: {path}")

    idx = None
    try:
        ser_num = pd.to_numeric(df[dcol], errors="coerce")
        if ser_num.notna().mean() > 0.8 and (ser_num.between(1900, 2100)).mean() >= 0.90:
            yrs = ser_num.astype("Int64")
            idx = pd.to_datetime(yrs.astype(str) + "-12-31", errors="coerce")
        else:
            idx = None
    except Exception:
        idx = None

    if idx is None:
        idx = pd.to_datetime(df[dcol], errors="coerce")

    if idx.isna().all():
        raise ValueError(f"Failed to parse dates from column '{dcol}' in: {path}")

    out = df.copy()
    out.index = idx
    out = out[~out.index.isna()].sort_index()
    return out


def _to_decimal_if_percent(s: pd.Series) -> pd.Series:
    if len(s) == 0:
        return s
    # If many values look like % (e.g., > 1.0), treat as percent
    if (s.abs() > 1.0).mean() >= 0.10:
        return s / 100.0
    return s


def _extract_returns_series(df: pd.DataFrame, preferred: tuple[str, ...] = ()) -> pd.Series:
    """
    Find a reasonable returns column in df and return as a float Series indexed by df.index.
    Search order:
      1) any of `preferred`
      2) common aliases for daily returns
      3) first numeric column
    Also converts %→decimal if needed.
    """
    for c in preferred:
        if c in df.columns:
            s = pd.Series(df[c].astype(float).values, index=df.index).sort_index()
            return _to_decimal_if_percent(s)

    aliases = (
        "BENCH_DAILY_RETURNS", "bench_daily_returns", "bench_returns",
        "ALPHA_DAILY_RETURNS", "alpha_daily_returns", "alpha_returns",
        "return", "returns", "ret", "daily_return"
    )
    for c in aliases:
        if c in df.columns:
            s = pd.Series(df[c].astype(float).values, index=df.index).sort_index()
            return _to_decimal_if_percent(s)

    num = [c for c in df.columns if df[c].dtype.kind in "fc"]
    if not num:
        raise ValueError("No numeric returns column found.")
    s = pd.Series(df[num[0]].astype(float).values, index=df.index).sort_index()
    return _to_decimal_if_percent(s)


# ---------------------------- Russell & RECS loaders ----------------------------

def _bench_daily_from_r1000(xlsx_path: Path) -> pd.Series:
    """
    Construct a Russell 1000 *daily* return series from the official Russell XLSX
    by:
      - summing 'Port. Ending Market Value' across all constituents for each
        rebalance 'Date'
      - computing period returns between consecutive rebalances
      - distributing each period's return evenly across business days in the interval
    """
    df = pd.read_excel(xlsx_path, sheet_name=0)
    if df.empty:
        raise ValueError(f"{xlsx_path} is empty.")

    # Date column
    date_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("date", "asof", "asofdate", "as_of", "as_of_date"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # Market Value column
    mv_col = None
    for c in df.columns:
        if "market value" in str(c).lower():
            mv_col = c
            break
    if mv_col is None:
        for c in df.columns:
            cl = str(c).lower()
            if "ending" in cl and "value" in cl:
                mv_col = c
                break
    if mv_col is None:
        raise ValueError("Could not find a 'Port. Ending Market Value' column in R1000.xlsx.")

    mv = (
        pd.Series(df[mv_col].astype(str))
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": np.nan})
        .astype(float)
    )

    g = (
        pd.DataFrame({"Date": dates, "MV": mv})
        .dropna(subset=["Date", "MV"])
        .groupby("Date", as_index=True)["MV"]
        .sum()
        .sort_index()
    )
    if len(g) < 2:
        raise ValueError("Not enough rebalance points in R1000.xlsx to form period returns.")

    daily_idx: List[pd.Timestamp] = []
    daily_vals: List[float] = []
    for prev_dt, cur_dt in zip(g.index[:-1], g.index[1:]):
        prev_mv = g.loc[prev_dt]
        cur_mv = g.loc[cur_dt]
        Rp = cur_mv / prev_mv - 1.0
        bdays = pd.bdate_range(prev_dt + pd.Timedelta(days=1), cur_dt, inclusive="right")
        n = len(bdays)
        if n <= 0:
            continue
        r_d = (1.0 + Rp) ** (1.0 / n) - 1.0
        daily_idx.extend(list(bdays))
        daily_vals.extend([r_d] * n)

    s = pd.Series(daily_vals, index=pd.DatetimeIndex(daily_idx), name="R1000_daily_ret").sort_index()
    return s


def _load_russell_daily_returns(
    reports_dir: Path,
    bench_csv: Optional[str],
    r1000_xlsx: Optional[Path] = None,
) -> pd.Series:
    """
    Try, in order:
      1) explicit path
      2) <reports>/bench_returns.csv
      3) <reports>/alpha_vs_bench_validation.csv (auto-detect the bench column)
      4) (fallback) derive daily returns from Russell 1000 XLSX (rebalance MV → daily)
    """
    candidates = []
    if bench_csv:
        candidates.append(Path(bench_csv).expanduser())
    candidates += [
        Path(reports_dir) / "bench_returns.csv",
        Path(reports_dir) / "alpha_vs_bench_validation.csv",
    ]
    last_err = None
    for p in candidates:
        try:
            if not p.exists():
                continue
            df = _read_csv_guess(p)
            if p.name == "alpha_vs_bench_validation.csv":
                s = _extract_returns_series(
                    df, preferred=("BENCH_DAILY_RETURNS", "bench_daily_returns", "bench_returns")
                )
                return s
            s = _extract_returns_series(df, preferred=("return", "returns", "ret", "daily_return"))
            return s
        except Exception as e:
            last_err = e
            continue

    if r1000_xlsx is not None and Path(r1000_xlsx).expanduser().exists():
        return _bench_daily_from_r1000(Path(r1000_xlsx).expanduser())

    raise FileNotFoundError(
        f"Could not find Russell 1000 daily returns; tried: {', '.join(str(x) for x in candidates)}. "
        f"Last error: {last_err}. Fallback also failed because R1000.xlsx was not provided or not found."
    )


def _load_recs_returns_daily_or_annual(path: str) -> Tuple[pd.Series, bool]:
    """
    Load RECS returns from CSV:
       - If daily series, return (series, False)
       - If annual series (e.g., 2019..), return (series indexed to Dec-31 each year, True)
    The function accepts percent or decimal returns and normalizes to decimal.
    """
    df = _read_csv_guess(Path(path).expanduser())
    s = _extract_returns_series(df)
    # Detect annual: one (or a few) points per calendar year, or short series
    years = s.index.year
    counts = pd.Series(1, index=years).groupby(level=0).sum()
    is_annual = (counts.median() <= 3) or (len(s) < 80)
    return s, is_annual


def _expand_annual_to_business_daily(annual: pd.Series) -> pd.Series:
    """
    Convert an annual total return series (indexed with a date in each year, any day)
    to a business-daily series by evenly distributing the year's total return
    across business days in that calendar year.
    """
    annual = annual.copy()
    # normalize index to Dec-31 of that year
    years = annual.index.year
    target_idx = pd.to_datetime(pd.Series(years.astype(int)).astype(str) + "-12-31")
    annual.index = target_idx
    annual = annual[~annual.index.duplicated(keep="last")].sort_index()

    # Build daily
    daily_idx: List[pd.Timestamp] = []
    daily_vals: List[float] = []
    # iterate years in order; for the first year, start on Jan-01 of that year
    for i, dt in enumerate(annual.index):
        year = dt.year
        # business days for the year
        bdays = pd.bdate_range(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
        n = len(bdays)
        Rp = float(annual.loc[dt])
        if n <= 0:
            continue
        r_d = (1.0 + Rp) ** (1.0 / n) - 1.0
        daily_idx.extend(list(bdays))
        daily_vals.extend([r_d] * n)
    s = pd.Series(daily_vals, index=pd.DatetimeIndex(daily_idx), name="RECS_daily_ret").sort_index()
    return s


def _read_recs_top(path_csv: str, n: int = 15) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    if df.empty:
        raise ValueError(f"{path_csv} is empty.")

    candidates = {"Ticker": None, "Name": None, "Weight": None, "Sector": None, "AsOf": None}
    for c in df.columns:
        cl = c.strip().lower()
        if "ticker" in cl and candidates["Ticker"] is None:
            candidates["Ticker"] = c
        if cl in ("name", "security", "holding", "issuer") and candidates["Name"] is None:
            candidates["Name"] = c
        if "weight" in cl and candidates["Weight"] is None:
            candidates["Weight"] = c
        if "sector" in cl and candidates["Sector"] is None:
            candidates["Sector"] = c
        if ("asof" in cl or "date" in cl) and candidates["AsOf"] is None:
            candidates["AsOf"] = c

    if candidates["Weight"] is None:
        raise ValueError("RECS top holdings CSV missing a Weight column.")

    out = df.copy()
    out["Ticker"] = out[candidates["Ticker"]] if candidates["Ticker"] else np.nan
    out["Description"] = out[candidates["Name"]] if candidates["Name"] else np.nan
    out["GICS Sector Extended"] = out[candidates["Sector"]] if candidates["Sector"] else np.nan

    wraw = out[candidates["Weight"]].astype(str).str.replace("%", "", regex=False)
    w = pd.to_numeric(wraw, errors="coerce")
    # Keep units as-is (if it's 12.3 → assume %; if 0.123 → fraction),
    # normalize to fraction for internal use
    if w.max(skipna=True) > 1.5:
        out["Weight_frac"] = w / 100.0
    else:
        out["Weight_frac"] = w

    if candidates["AsOf"] and candidates["AsOf"] in df.columns:
        out["AsOf"] = pd.to_datetime(out[candidates["AsOf"]], errors="coerce").dt.date
    out = out.sort_values("Weight_frac", ascending=False).head(n)
    return out[["Ticker", "Description", "GICS Sector Extended", "Weight_frac"]].reset_index(drop=True)


def _read_russell_top_2024(xlsx_path: Path, n: int = 15) -> pd.DataFrame:
    """
    Read the single-sheet R1000.xlsx, filter to the latest date in 2024 (or nearest),
    and return top n by 'Port. Ending Weight'.
    """
    df = pd.read_excel(xlsx_path, sheet_name=0)
    if df.empty:
        raise ValueError(f"{xlsx_path} is empty.")

    # Detect columns
    cols = {c.lower(): c for c in df.columns}
    # date
    if "date" in cols:
        dcol = cols["date"]
    else:
        # fallback: try first column
        dcol = list(df.columns)[0]
    df["_Date"] = pd.to_datetime(df[dcol], errors="coerce")
    df = df[~df["_Date"].isna()]

    # Ticker
    tcol = None
    for c in df.columns:
        if str(c).strip().lower() == "ticker":
            tcol = c
            break
    if tcol is None:
        raise ValueError("Ticker column not found in R1000.xlsx")
    # Name/Description
    ncol = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("description", "name", "security", "holding"):
            ncol = c
            break
    # Sector
    scol = None
    for c in df.columns:
        if "gics sector" in str(c).lower():
            scol = c
            break
    # Weight
    wcol = None
    for c in df.columns:
        if "ending weight" in str(c).lower() or "weight" in str(c).lower():
            wcol = c
            break
    if wcol is None:
        raise ValueError("Could not find a 'Port. Ending Weight' column in R1000.xlsx.")

    # Target: last available in 2024
    d2024 = df[df["_Date"].dt.year == 2024]
    if d2024.empty:
        # fallback: nearest date before 2025
        d2024 = df[df["_Date"] <= pd.Timestamp("2024-12-31")]
    if d2024.empty:
        # final: max date in file
        d2024 = df.copy()

    latest_date = d2024["_Date"].max()
    snap = df[df["_Date"] == latest_date].copy()

    # Clean weight; may be like "1.23%" or "1.23"
    wraw = snap[wcol].astype(str).str.replace("%", "", regex=False)
    wnum = pd.to_numeric(wraw, errors="coerce")
    if wnum.max(skipna=True) > 1.5:
        wfrac = wnum / 100.0
    else:
        wfrac = wnum

    out = pd.DataFrame({
        "Ticker": snap[tcol].values,
        "Description": snap[ncol].values if ncol else np.nan,
        "GICS Sector Extended": snap[scol].values if scol else np.nan,
        "Weight_frac": wfrac.values,
    })
    out = out.sort_values("Weight_frac", ascending=False).head(n).reset_index(drop=True)
    return out


# ---------------------------- Synthesis & KPIs ----------------------------

def _synth_alpha_from_bench(bench_daily: pd.Series, annual_excess: float = 0.01) -> pd.Series:
    """
    Create a realistic synthetic Alpha series:
    - same base path as benchmark + small positive drift to target ~1% annual excess
    - preserves overall volatility/correlation look-and-feel
    """
    bench_daily = bench_daily.sort_index().dropna().copy()
    if bench_daily.empty:
        return bench_daily

    # daily drift to get annual_excess over ~252 trading days
    drift = (1.0 + annual_excess) ** (1.0 / 252.0) - 1.0  # ≈ 0.000039 for 1%/yr
    # small noise proportional to bench vol (to avoid a perfectly parallel line)
    sigma = max(1e-6, bench_daily.std() * 0.10)
    rng = np.random.default_rng(42)  # deterministic
    noise = rng.normal(loc=0.0, scale=sigma, size=len(bench_daily))

    alpha = bench_daily + drift + noise
    # Cap extreme values (safety)
    alpha = alpha.clip(lower=-0.2, upper=0.2)
    alpha.name = "ALPHA_daily_ret"
    return pd.Series(alpha.values, index=bench_daily.index, name="ALPHA_daily_ret")


def _kpis_from_daily(ret: pd.Series) -> Dict[str, float]:
    ret = ret.dropna()
    if len(ret) == 0:
        return dict(CAGR=np.nan, Sharpe=np.nan, Sortino=np.nan)
    mean = ret.mean()
    vol = ret.std()
    downside = ret[ret < 0].std()
    ann = 252.0
    sharpe = (mean / vol) * np.sqrt(ann) if (vol is not None and vol > 0) else np.nan
    sortino = (mean / downside) * np.sqrt(ann) if (downside is not None and downside > 0) else np.nan
    tot = (1.0 + ret).prod()
    years = len(ret) / ann
    cagr = tot ** (1.0 / years) - 1.0 if years > 0 else np.nan
    return dict(CAGR=cagr, Sharpe=sharpe, Sortino=sortino)


# ---------------------------- Plot helpers ----------------------------

def _calendar_bars(ax, series_map: Dict[str, pd.Series]) -> None:
    """
    Plot calendar-year total returns bars for each series in series_map.
    """
    # compute year returns
    annual_map: Dict[str, pd.Series] = {}
    all_years = set()
    for name, s in series_map.items():
        if s.empty:
            annual_map[name] = pd.Series(dtype=float)
            continue
        ann = (1.0 + s).resample("YE").apply(lambda x: x.prod() - 1.0)
        annual_map[name] = ann
        all_years |= set(ann.index.year.tolist())

    years = sorted(all_years)
    if not years:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    # assemble matrix (names x years)
    names = list(series_map.keys())
    mat = []
    for nm in names:
        s = annual_map[nm]
        row = [float(s.get(pd.Timestamp(f"{y}-12-31"), np.nan)) for y in years]
        mat.append(row)

    mat = np.array(mat)  # shape (N, Y)
    N, Y = mat.shape
    idx = np.arange(Y)
    width = 0.8 / max(1, N)

    for i, nm in enumerate(names):
        ax.bar(idx + i * width, mat[i], width=width, label=nm)

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(idx + width * (N - 1) / 2)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Calendar Return")
    ax.set_title("Calendar-Year Returns")
    ax.legend(loc="best")


def _cum_nav(ax, series_map: Dict[str, pd.Series], start_nav: float = 100.0) -> None:
    for name, s in series_map.items():
        if s.empty:
            continue
        nav = start_nav * (1.0 + s).cumprod()
        ax.plot(nav.index, nav.values, label=name)
    ax.set_title("Cumulative NAV")
    ax.set_ylabel("NAV")
    ax.legend(loc="best")


def _growth_1m(ax, series_map: Dict[str, pd.Series]) -> None:
    for name, s in series_map.items():
        if s.empty:
            continue
        nav = 1_000_000.0 * (1.0 + s).cumprod()
        ax.plot(nav.index, nav.values, label=name)
    ax.set_title("Growth of $1,000,000")
    ax.set_ylabel("Dollars")
    ax.legend(loc="best")


def _kpi_bars(ax, kpi_map: Dict[str, Dict[str, float]]) -> None:
    metrics = ["CAGR", "Sharpe", "Sortino"]
    names = list(kpi_map.keys())
    idx = np.arange(len(metrics))
    width = 0.8 / max(1, len(names))

    for i, nm in enumerate(names):
        vals = [kpi_map[nm].get(m, np.nan) for m in metrics]
        ax.bar(idx + i * width, vals, width=width, label=nm)

    ax.set_xticks(idx + width * (len(names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_title("Performance KPIs")
    ax.legend(loc="best")


# ---------------------------- CLI / Main ----------------------------

def main():
    p = argparse.ArgumentParser(description="Synthetic showcase plots/tables.")
    p.add_argument("--reports-dir", required=True, help="Root reports dir (will write plots/ here).")
    p.add_argument("--r1000-xlsx", required=True, help="Absolute path to R1000.xlsx")
    p.add_argument("--bench-returns-csv", default=None, help="Optional path to daily benchmark returns CSV")
    p.add_argument("--recs-returns-csv", required=True, help="RECS returns CSV (daily or annual)")
    p.add_argument("--recs-top-holdings", required=True, help="RECS top holdings CSV (for table)")
    p.add_argument("--alpha-annual-excess", type=float, default=0.01, help="Alpha annual excess vs bench (default 1%)")
    args = p.parse_args()

    reports = Path(args.reports_dir).expanduser()
    plots = _ensure_plots_dir(reports)

    # 1) Benchmark daily series (CSV -> daily; else derive from R1000.xlsx rebalance MVs)
    bench_daily = _load_russell_daily_returns(
        reports_dir=reports,
        bench_csv=args.bench_returns_csv,
        r1000_xlsx=Path(args.r1000_xlsx),
    )

    # 2) RECS
    recs_daily, recs_is_annual = _load_recs_returns_daily_or_annual(args.recs_returns_csv)
    if recs_is_annual:
        recs_daily = _expand_annual_to_business_daily(recs_daily)

    # 3) Synthetic Alpha
    alpha_daily = _synth_alpha_from_bench(bench_daily, annual_excess=float(args.alpha_annual_excess))

    # Align to overlap window (prefer 2019–2025; degrade gracefully to real overlap)
    tgt_start, tgt_end = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-12-31")
    bench_daily = bench_daily.loc[tgt_start:tgt_end]
    recs_daily = recs_daily.loc[tgt_start:tgt_end]
    alpha_daily = alpha_daily.loc[tgt_start:tgt_end]

    def _overlap(slist: List[pd.Series]) -> Tuple[pd.Timestamp, pd.Timestamp]:
        lo = max([s.index.min() for s in slist if not s.empty])
        hi = min([s.index.max() for s in slist if not s.empty])
        return lo, hi

    lo, hi = _overlap([bench_daily, recs_daily, alpha_daily])
    if pd.isna(lo) or pd.isna(hi) or lo >= hi:
        # if RECS window is shorter (launched 2019), at least align Alpha/Bench
        lo, hi = _overlap([bench_daily, alpha_daily])
    bench_daily = bench_daily.loc[lo:hi]
    recs_daily = recs_daily.loc[lo:hi]
    alpha_daily = alpha_daily.loc[lo:hi]

    if bench_daily.empty or alpha_daily.empty:
        raise ValueError("Empty Alpha or Russell daily series after alignment; check inputs.")
    if recs_daily.empty:
        print("[showcase] Warning: RECS daily window empty after alignment; continuing with Alpha vs R1000 only.")

    # 4) KPI table
    kpis = {
        "Alpha": _kpis_from_daily(alpha_daily),
        "R1000": _kpis_from_daily(bench_daily),
    }
    if not recs_daily.empty:
        kpis["RECS"] = _kpis_from_daily(recs_daily)

    kpi_df = (
        pd.DataFrame(kpis)
        .reindex(["CAGR", "Sharpe", "Sortino"])
        .round(4)
    )
    kpi_df.to_csv(reports / "showcase_kpis.csv")

    # 5) Constituents table as of Dec-2024 (or nearest)
    r_top = _read_russell_top_2024(Path(args.r1000_xlsx), n=15)
    recs_top = _read_recs_top(args.recs_top_holdings, n=15)
    # Synthetic Alpha top: small tilts to R1000 top to look realistic
    alpha_top = r_top.copy()
    if not alpha_top.empty:
        # +10% relative to top 3 weights, then renormalize
        w = alpha_top["Weight_frac"].values.copy()
        k = min(3, len(w))
        w[:k] = w[:k] * 1.10
        w = w / w.sum() * alpha_top["Weight_frac"].sum()
        alpha_top["Weight_frac"] = w

    # Save table concatenated with a "Which" column
    r_top_ = r_top.copy(); r_top_["Which"] = "R1000"
    a_top_ = alpha_top.copy(); a_top_["Which"] = "Alpha"
    recs_top_ = recs_top.copy(); recs_top_["Which"] = "RECS"
    out_tbl = pd.concat([r_top_[["Which", "Ticker", "Description", "GICS Sector Extended", "Weight_frac"]],
                         a_top_[["Which", "Ticker", "Description", "GICS Sector Extended", "Weight_frac"]],
                         recs_top_[["Which", "Ticker", "Description", "GICS Sector Extended", "Weight_frac"]]],
                        ignore_index=True)
    out_tbl.rename(columns={"Weight_frac": "Weight"}, inplace=True)
    out_tbl.to_csv(reports / "top_constituents_2024.csv", index=False)

    # 6) Plots
    # Calendar-year returns
    fig, ax = plt.subplots(figsize=(12, 4))
    s_map = {"Alpha": alpha_daily, "R1000": bench_daily}
    if not recs_daily.empty:
        s_map["RECS"] = recs_daily
    _calendar_bars(ax, s_map)
    fig.tight_layout()
    fig.savefig(plots / "calendar_year_returns_synth.png", dpi=160)

    # Cumulative NAV
    fig, ax = plt.subplots(figsize=(12, 4))
    _cum_nav(ax, s_map, start_nav=100.0)
    fig.tight_layout()
    fig.savefig(plots / "cum_nav_synth.png", dpi=160)

    # Growth of $1M
    fig, ax = plt.subplots(figsize=(12, 4))
    _growth_1m(ax, s_map)
    fig.tight_layout()
    fig.savefig(plots / "growth_1m_synth.png", dpi=160)

    # KPI bars
    fig, ax = plt.subplots(figsize=(8, 4))
    _kpi_bars(ax, kpis)
    fig.tight_layout()
    fig.savefig(plots / "kpi_bars_synth.png", dpi=160)

    print(f"Charts saved to: {plots}")
    print(f"KPI table saved to: {reports / 'showcase_kpis.csv'}")
    print(f"Constituents table saved to: {reports / 'top_constituents_2024.csv'}")


if __name__ == "__main__":
    main()
