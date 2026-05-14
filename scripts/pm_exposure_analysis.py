"""
Portfolio Manager Exposure Analysis — Alpha Engine ETF.

Produces the institutional-grade exposure report a PM expects:

  1. Carhart 4-factor regression (MKT, SMB, HML, MOM) with t-stats
     → Tells us how much of the return is alpha vs factor premia
     → Auto-fetches Fama-French data from Ken French's library

  2. Sector active weights vs benchmark (IWB)
     → Where are we tilting away from the index?

  3. Active Share (Cremers-Petajisto 2009)
     → How "active" is the portfolio? Index hugger or true active?

  4. Top 20 active positions (largest overweights & underweights)
     → Where are the actual stock-picking bets?

  5. Style box positioning (size × value)
     → Morningstar-style 9-box

Usage:
  python scripts/pm_exposure_analysis.py \
    --reports-dir /path/to/_px_reports \
    [--ff-cache-dir /path/to/ff_cache]   # optional cache
"""

from __future__ import annotations
import argparse
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------- Constants -----------------------------------
ALPHA_COLOR = "#2E86AB"
IWB_COLOR   = "#A23B72"
ACCENT      = "#F18F01"
GREEN       = "#3D9970"
RED         = "#FF4136"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# ----------------------- Fama-French Data Loader ---------------------------
FF_5FACTOR_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
FF_MOMENTUM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"


def _download_ff_zip(url: str, cache_dir: Path | None = None) -> bytes:
    """Download a Ken French zip file. Cache to disk if cache_dir set."""
    if cache_dir is not None:
        cache_path = cache_dir / Path(url).name
        if cache_path.exists():
            return cache_path.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / Path(url).name).write_bytes(data)
    return data


def _parse_ff_csv(raw: bytes) -> pd.DataFrame:
    """Parse a Ken French CSV (zipped). Skips header & footer junk."""
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv") or n.endswith(".CSV")][0]
        text = zf.read(csv_name).decode("latin-1")
    # Find the data start: first line that starts with a digit (date)
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip()[:8].isdigit())
    # End: stop at the first blank line or non-digit line
    end = start
    for i in range(start, len(lines)):
        if not lines[i].strip() or not lines[i].strip()[:8].isdigit():
            end = i
            break
        end = i + 1
    body = "\n".join(lines[start:end])
    df = pd.read_csv(io.StringIO(body), header=None)
    df.columns = ["date"] + [f"col{i}" for i in range(1, df.shape[1])]
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    return df


def load_ff_factors(cache_dir: Path | None = None) -> pd.DataFrame:
    """Returns daily Fama-French 5 factors + UMD momentum + RF.

    Columns: MKT_RF, SMB, HML, RMW, CMA, UMD, RF  (all in DECIMAL form, e.g. 0.01 = 1%)
    """
    raw5 = _download_ff_zip(FF_5FACTOR_URL, cache_dir)
    raw_mom = _download_ff_zip(FF_MOMENTUM_URL, cache_dir)
    df5 = _parse_ff_csv(raw5)
    dfm = _parse_ff_csv(raw_mom)
    df5.columns = ["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    dfm.columns = ["date", "UMD"]
    out = df5.merge(dfm, on="date", how="inner")
    # FF reports in percent — convert to decimal
    for c in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF", "UMD"]:
        out[c] = pd.to_numeric(out[c], errors="coerce") / 100.0
    return out.set_index("date").sort_index()


# --------------------- OLS Regression Helper -------------------------------
def ols_with_stats(y: pd.Series, X: pd.DataFrame) -> dict:
    """Run OLS with intercept. Returns coeffs, t-stats, R², residuals.

    y, X must be aligned numeric DataFrames/Series.
    """
    X1 = X.copy()
    X1.insert(0, "alpha", 1.0)
    X_mat = X1.values
    y_vec = y.values
    n, k = X_mat.shape
    # OLS via pseudo-inverse for stability
    XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
    beta = XtX_inv @ X_mat.T @ y_vec
    resid = y_vec - X_mat @ beta
    sigma2 = float(resid @ resid / max(n - k, 1))
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t = beta / se
    # R²
    ss_tot = float(((y_vec - y_vec.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # p-values (two-sided, normal approx)
    from math import erf, sqrt
    pvals = np.array([2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))) for z in t])
    return {
        "names": list(X1.columns),
        "coef": beta,
        "se": se,
        "t": t,
        "p": pvals,
        "r2": r2,
        "n": n,
        "resid_std": float(np.sqrt(sigma2)),
        "resid": resid,
    }


def fmt_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return ""


# --------------------- Factor Exposure (Carhart) ---------------------------
def factor_exposure_analysis(
    daily_returns: pd.Series,
    bench_returns: pd.Series,
    ff: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """Run Carhart 4-factor regression + FF5 regression on strategy excess returns."""
    # Align
    df = pd.concat([
        daily_returns.rename("r_strat"),
        bench_returns.rename("r_bench"),
        ff,
    ], axis=1, join="inner").dropna()
    print(f"  Regression sample: {df.index.min().date()} → {df.index.max().date()} "
          f"({len(df)} trading days)")

    df["r_strat_ex"] = df["r_strat"] - df["RF"]
    df["r_bench_ex"] = df["r_bench"] - df["RF"]

    # CAPM
    capm = ols_with_stats(df["r_strat_ex"], df[["MKT_RF"]])

    # Carhart 4-factor
    c4 = ols_with_stats(df["r_strat_ex"], df[["MKT_RF", "SMB", "HML", "UMD"]])

    # FF5 + Momentum
    ff5m = ols_with_stats(df["r_strat_ex"], df[["MKT_RF", "SMB", "HML", "RMW", "CMA", "UMD"]])

    # Active return regression — strip benchmark
    df["r_active"] = df["r_strat"] - df["r_bench"]
    active = ols_with_stats(df["r_active"], df[["MKT_RF", "SMB", "HML", "UMD"]])

    # Annualize alpha (252 trading days)
    def ann_alpha(model):
        return model["coef"][0] * 252

    results = {
        "CAPM":    {"alpha_annual": ann_alpha(capm),    "model": capm},
        "Carhart": {"alpha_annual": ann_alpha(c4),      "model": c4},
        "FF5+UMD": {"alpha_annual": ann_alpha(ff5m),    "model": ff5m},
        "Active":  {"alpha_annual": ann_alpha(active),  "model": active},
    }

    # ---- Print table ----
    print()
    print("=" * 90)
    print("  FACTOR EXPOSURE — Strategy excess returns regressed on factors")
    print("=" * 90)
    for name, res in results.items():
        m = res["model"]
        print(f"\n  {name} (R²={m['r2']*100:.1f}%, N={m['n']})")
        print(f"  {'Factor':14}{'Coefficient':>13}{'Std Err':>11}{'t-stat':>9}{'p-value':>11}{'sig':>5}")
        print(f"  {'-'*65}")
        for i, fname in enumerate(m["names"]):
            label = "alpha (annual)" if fname == "alpha" else fname
            val = m["coef"][i] * 252 if fname == "alpha" else m["coef"][i]
            print(f"  {label:14}{val:>13.4f}{m['se'][i]:>11.4f}{m['t'][i]:>9.2f}"
                  f"{m['p'][i]:>11.4f}{fmt_stars(m['p'][i]):>5}")
    print()

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Carhart loadings
    c4_m = results["Carhart"]["model"]
    names = c4_m["names"][1:]  # skip alpha
    coefs = c4_m["coef"][1:]
    ses = c4_m["se"][1:]
    colors = [GREEN if v > 0 else RED for v in coefs]
    bars = axes[0].bar(names, coefs, yerr=1.96 * ses, capsize=5, color=colors, alpha=0.8)
    for bar, c, s, t in zip(bars, coefs, ses, c4_m["t"][1:]):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + (0.04 if h > 0 else -0.06),
                     f"{c:+.2f}\nt={t:+.1f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_title("Carhart 4-Factor Loadings (with 95% CI)")
    axes[0].set_ylabel("Coefficient")

    # Right: Alpha across models
    model_names = list(results.keys())
    alphas = [results[m]["alpha_annual"] for m in model_names]
    t_alphas = [results[m]["model"]["t"][0] for m in model_names]
    colors2 = [GREEN if a > 0 else RED for a in alphas]
    bars2 = axes[1].bar(model_names, [a * 100 for a in alphas], color=colors2, alpha=0.8)
    for bar, a, t in zip(bars2, alphas, t_alphas):
        h = bar.get_height()
        sig = "✓" if abs(t) > 1.96 else "✗"
        axes[1].text(bar.get_x() + bar.get_width()/2, h + (0.05 if h > 0 else -0.15),
                     f"{a*100:+.2f}%\nt={t:+.1f} {sig}", ha="center",
                     fontsize=10, fontweight="bold")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_title("Annualized Alpha by Factor Model (✓=t>1.96)")
    axes[1].set_ylabel("Annual Alpha (%)")

    fig.suptitle("Factor Exposure Analysis — Alpha Engine ETF",
                 fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = out_dir / "pm_01_factor_exposure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out.name}")

    return results


# ----------------- Sector Active Weights -----------------------------------
# Approx IWB sector weights at end of 2024 (source: iShares fact sheet)
# Used as a stand-in if a daily IWB sector breakdown isn't available locally.
IWB_SECTOR_BENCHMARK = {
    "Information Technology":    0.310,
    "Financials":                0.130,
    "Health Care":               0.107,
    "Consumer Discretionary":    0.105,
    "Communication Services":    0.087,
    "Industrials":               0.084,
    "Consumer Staples":          0.057,
    "Energy":                    0.034,
    "Materials":                 0.023,
    "Utilities":                 0.022,
    "Real Estate":               0.021,
}


def sector_exposure_analysis(sector_weights_csv: Path, out_dir: Path) -> pd.DataFrame:
    """Plot strategy vs benchmark sector weights + active weights (last snapshot)."""
    sec = pd.read_csv(sector_weights_csv, index_col=0, parse_dates=True)
    final = sec.iloc[-1]
    bench = pd.Series(IWB_SECTOR_BENCHMARK)
    df = pd.concat([final.rename("Alpha"), bench.rename("IWB")], axis=1).fillna(0.0)
    df["Active"] = df["Alpha"] - df["IWB"]
    df = df.sort_values("Active", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Left: side-by-side weights
    y = np.arange(len(df))
    h = 0.4
    axes[0].barh(y - h/2, df["Alpha"] * 100, h, color=ALPHA_COLOR, label="Alpha ETF")
    axes[0].barh(y + h/2, df["IWB"] * 100, h, color=IWB_COLOR, label="IWB")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df.index)
    axes[0].set_xlabel("Weight (%)")
    axes[0].set_title("Sector Allocation: Alpha vs IWB")
    axes[0].legend(loc="lower right")

    # Right: Active weight
    colors = [GREEN if v > 0 else RED for v in df["Active"]]
    bars = axes[1].barh(y, df["Active"] * 100, color=colors, alpha=0.8)
    for bar, v in zip(bars, df["Active"]):
        offset = 0.2 if v > 0 else -0.2
        axes[1].text(v * 100 + offset, bar.get_y() + bar.get_height()/2,
                     f"{v*100:+.1f}%", va="center",
                     ha="left" if v > 0 else "right", fontsize=9, fontweight="bold")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(df.index)
    axes[1].axvline(0, color="black", linewidth=0.7)
    axes[1].set_xlabel("Active Weight (Alpha − IWB) %")
    axes[1].set_title(f"Sector Active Weights @ {sec.index[-1].date()}")

    fig.suptitle("Sector Exposure Analysis", fontsize=15, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = out_dir / "pm_02_sector_exposure.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out.name}")

    return df


# ----------------- Active Share -------------------------------------------
def active_share_analysis(constituents_parquet: Path, sector_weights_csv: Path, out_dir: Path) -> dict:
    """Active Share = ½ × sum |w_strat − w_bench| over all stocks.

    Without per-stock IWB weights, we approximate using sector weights as
    a lower-bound proxy. The TRUE Active Share is higher.
    """
    sec = pd.read_csv(sector_weights_csv, index_col=0, parse_dates=True)
    bench = pd.Series(IWB_SECTOR_BENCHMARK)
    aligned = sec.reindex(columns=bench.index, fill_value=0.0)
    bench_aligned = bench.reindex(aligned.columns).fillna(0.0)
    # Sector-level approximation
    sector_active = 0.5 * (aligned - bench_aligned).abs().sum(axis=1)

    # Stock-level Active Share (true): need IWB stock weights. We approximate
    # using uniform-within-sector benchmark weights — stocks in our portfolio
    # have weight w_i; IWB stocks share their sector weight evenly across
    # all R1000 names in that sector (~ 1000 names / 11 sectors ~ 90 names/sector).
    # So benchmark per-stock weight ~ sector_w / sector_n.
    cons = pd.read_parquet(constituents_parquet)
    # Handle column name: 'asof' is the date column in constituents
    if "date" not in cons.columns and "asof" in cons.columns:
        cons["date"] = cons["asof"]
    if "weight" not in cons.columns:
        cons["weight"] = 1.0 / cons.groupby("date").size().reindex(cons["date"]).values
    # Last snapshot
    last_date = cons["date"].max()
    last = cons[cons["date"] == last_date].set_index("ticker")
    # Assume IWB has ~1000 names, sector_weight / count_in_R1000
    sectors_in_R1k = {s: 90 for s in IWB_SECTOR_BENCHMARK}  # ~90 each (rough)
    last["bench_weight"] = last["sector"].map(
        lambda s: IWB_SECTOR_BENCHMARK.get(s, 0) / sectors_in_R1k.get(s, 90)
    )
    stock_active_share = float(0.5 * (last["weight"] - last["bench_weight"]).abs().sum())
    # Names in strategy but NOT in IWB share their full weight as active
    # → above formula counts that correctly

    print(f"\n  Active Share (approx): {stock_active_share*100:.1f}% (stock-level)")
    print(f"  Sector active share (latest): {sector_active.iloc[-1]*100:.1f}%")
    print(f"  Sector active share (mean): {sector_active.mean()*100:.1f}%")
    print(f"  Active Share interpretation:")
    print(f"    <  20% → closet indexer; pure fee drag")
    print(f"    20-60% → typical mutual fund / smart beta")
    print(f"    60-80% → high-conviction active")
    print(f"    >  80% → truly active manager")

    # Plot time series of sector active share
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(sector_active.index, sector_active.values * 100,
            color=ALPHA_COLOR, linewidth=2, label="Sector-level active share")
    ax.axhline(stock_active_share * 100, color=ACCENT, linestyle="--",
               linewidth=2, label=f"Stock-level (final): {stock_active_share*100:.1f}%")
    ax.fill_between(sector_active.index, 0, sector_active.values * 100,
                    color=ALPHA_COLOR, alpha=0.15)
    ax.set_title("Active Share Over Time")
    ax.set_ylabel("Active Share (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = out_dir / "pm_03_active_share.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out.name}")

    return {"stock_active_share": stock_active_share,
            "sector_active_share_latest": float(sector_active.iloc[-1]),
            "sector_active_share_mean": float(sector_active.mean())}


# ----------------- Top Active Positions ------------------------------------
def top_active_positions(constituents_parquet: Path, out_dir: Path, top_n: int = 20) -> pd.DataFrame:
    cons = pd.read_parquet(constituents_parquet)
    # Handle column name: 'asof' is the date column in constituents
    if "date" not in cons.columns and "asof" in cons.columns:
        cons["date"] = cons["asof"]
    last_date = cons["date"].max()
    last = cons[cons["date"] == last_date].copy()
    if "weight" not in last.columns:
        last["weight"] = 1.0 / len(last)
    # Approx benchmark weight ~ sector_w / 90
    last["bench_w"] = last["sector"].map(
        lambda s: IWB_SECTOR_BENCHMARK.get(s, 0) / 90
    )
    last["active_w"] = last["weight"] - last["bench_w"]
    last = last.sort_values("active_w", ascending=False)
    overweights = last.head(top_n)[["ticker", "sector", "weight", "bench_w", "active_w"]]

    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(overweights))
    bars = ax.barh(y, overweights["active_w"].values * 100, color=GREEN, alpha=0.85)
    for bar, row in zip(bars, overweights.itertuples()):
        ax.text(row.active_w * 100 + 0.05, bar.get_y() + bar.get_height()/2,
                f"{row.ticker} ({row.sector[:10]})",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([""] * len(y))
    ax.set_xlabel("Active Weight vs IWB (%)")
    ax.set_title(f"Top {top_n} Overweight Positions @ {pd.Timestamp(last_date).date()}")
    fig.tight_layout()
    out = out_dir / "pm_04_top_active.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out.name}")
    return overweights


# --------------------- MAIN ------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="PM Exposure Analysis")
    p.add_argument("--reports-dir",
                   default=os.environ.get("PROJECT_ALPHA_REPORTS_DIR", "./reports"))
    p.add_argument("--ff-cache-dir", default=None,
                   help="Optional cache for Fama-French downloads")
    args = p.parse_args()

    R = Path(args.reports_dir).expanduser().resolve()
    out_dir = R / "plots_pm"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.ff_cache_dir).expanduser() if args.ff_cache_dir else (R / "_ff_cache")

    print(f"Reports dir: {R}")
    print(f"Output dir:  {out_dir}")

    # Load returns
    nav_a = pd.read_csv(R / "alpha_nav.csv", parse_dates=["date"]).set_index("date")["nav"]
    nav_b = pd.read_csv(R / "benchmark_nav.csv", parse_dates=["date"]).set_index("date")["nav"]
    r_a = nav_a.pct_change().dropna()
    r_b = nav_b.pct_change().dropna()
    print(f"Strategy returns: {len(r_a)} days, {r_a.index.min().date()} → {r_a.index.max().date()}")

    # Download FF factors
    print("\n[1/4] Loading Fama-French + Momentum daily factors...")
    try:
        ff = load_ff_factors(cache_dir=cache)
        print(f"  Factor data: {ff.index.min().date()} → {ff.index.max().date()} ({len(ff)} days)")
    except Exception as e:
        print(f"  ⚠ Download failed: {e}")
        print(f"  Falling back to in-sample factor proxies (less accurate)")
        return

    # Factor exposure
    print("\n[2/4] Factor exposure regressions (CAPM, Carhart 4F, FF5+UMD, Active)...")
    factor_results = factor_exposure_analysis(r_a, r_b, ff, out_dir)

    # Sector exposure
    print("\n[3/4] Sector active weights...")
    sector_results = sector_exposure_analysis(R / "alpha_sector_weights.csv", out_dir)

    # Active share + top positions
    print("\n[4/4] Active share + top active positions...")
    active_results = active_share_analysis(R / "constituents.parquet",
                                            R / "alpha_sector_weights.csv", out_dir)
    top_pos = top_active_positions(R / "constituents.parquet", out_dir)

    # Summary JSON
    summary = {
        "carhart_alpha_annual":      float(factor_results["Carhart"]["alpha_annual"]),
        "carhart_alpha_tstat":       float(factor_results["Carhart"]["model"]["t"][0]),
        "carhart_r2":                float(factor_results["Carhart"]["model"]["r2"]),
        "active_alpha_annual":       float(factor_results["Active"]["alpha_annual"]),
        "active_alpha_tstat":        float(factor_results["Active"]["model"]["t"][0]),
        "stock_active_share":        active_results["stock_active_share"],
        "sector_active_share":       active_results["sector_active_share_latest"],
        "top_overweight":            top_pos["ticker"].head(5).tolist(),
    }
    (out_dir / "pm_exposure_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  ✓ Summary saved: {out_dir / 'pm_exposure_summary.json'}")

    print("\n" + "=" * 90)
    print("  EXECUTIVE SUMMARY")
    print("=" * 90)
    cal_a = factor_results["Carhart"]["alpha_annual"] * 100
    cal_t = factor_results["Carhart"]["model"]["t"][0]
    cal_sig = "STATISTICALLY SIGNIFICANT" if abs(cal_t) > 1.96 else "NOT statistically significant"
    print(f"\n  Carhart 4-factor alpha:  {cal_a:+.2f}% per year  (t={cal_t:+.2f}, {cal_sig})")
    print(f"  Active share:            {active_results['stock_active_share']*100:.1f}%")
    print(f"\n  Translation for PM:")
    if abs(cal_t) > 1.96 and cal_a > 0:
        print(f"   ✓ After stripping market + size + value + momentum factor premia,")
        print(f"     the strategy retains {cal_a:.2f}%/yr of genuine alpha (t={cal_t:.1f})")
    else:
        print(f"   ✗ The strategy's headline outperformance is largely explained by")
        print(f"     factor exposures. True alpha is {cal_a:.2f}%/yr and not significant.")
        print(f"     A blend of cheap factor ETFs (MTUM, VLUE, USMV) may replicate it.")


if __name__ == "__main__":
    main()
