"""
Visualization suite for Iter-10 Alpha ETF vs IWB (10-year backtest).

Generates:
  1. NAV growth: $1M invested in each strategy
  2. Drawdown comparison
  3. Annual returns bar chart
  4. Sector tilt evolution
  5. Pillar IC over time
  6. Pillar weights over time (dynamic IC blending)
  7. Rolling 12m Sharpe & IR
  8. Final sector allocation comparison
  9. Strategy flow diagram
  10. Regime overlay
"""

from __future__ import annotations
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

# ---- Paths -----------------------------------------------------------------
# Reads reports dir from env var (preferred) or --reports-dir CLI arg.
# Falls back to ./reports relative to project root if neither is set.
import argparse
_parser = argparse.ArgumentParser(description="Iter-10 visualization suite")
_parser.add_argument("--reports-dir",
                     default=os.environ.get("PROJECT_ALPHA_REPORTS_DIR", "./reports"),
                     help="Directory containing kpis.json, alpha_nav.csv, etc. "
                          "(or set PROJECT_ALPHA_REPORTS_DIR env var)")
_args, _ = _parser.parse_known_args()
REPORTS = Path(_args.reports_dir).expanduser().resolve()
if not REPORTS.exists():
    raise SystemExit(f"Reports directory does not exist: {REPORTS}\n"
                     f"Pass --reports-dir or set PROJECT_ALPHA_REPORTS_DIR.")
PLOTS_DIR = REPORTS / "plots_iter10_10yr"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# ---- Styling --------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
ALPHA_COLOR = "#2E86AB"      # blue
IWB_COLOR = "#A23B72"        # magenta
ACCENT = "#F18F01"           # orange
GREEN = "#3D9970"
RED = "#FF4136"


def _load_data():
    """Load all output files."""
    data = {}
    nav_alpha = pd.read_csv(REPORTS / "alpha_nav.csv", parse_dates=["date"]).set_index("date")
    nav_iwb = pd.read_csv(REPORTS / "benchmark_nav.csv", parse_dates=["date"]).set_index("date")
    # Align both NAVs and rebase to $1M
    nav = pd.concat([nav_alpha["nav"].rename("alpha"),
                     nav_iwb["nav"].rename("iwb")], axis=1).dropna()
    nav["alpha"] = nav["alpha"] / nav["alpha"].iloc[0] * 1_000_000
    nav["iwb"] = nav["iwb"] / nav["iwb"].iloc[0] * 1_000_000
    data["nav"] = nav

    data["kpis"] = json.load(open(REPORTS / "kpis.json"))

    if (REPORTS / "alpha_sector_weights.csv").exists():
        sec = pd.read_csv(REPORTS / "alpha_sector_weights.csv", index_col=0, parse_dates=True)
        data["sector_weights"] = sec

    if (REPORTS / "pillar_ic_history.csv").exists():
        ic = pd.read_csv(REPORTS / "pillar_ic_history.csv", parse_dates=["asof"])
        data["pillar_ic"] = ic

    if (REPORTS / "pillar_weights_history.csv").exists():
        pw = pd.read_csv(REPORTS / "pillar_weights_history.csv", parse_dates=["date"])
        pw = pw.rename(columns={"date": "asof"})
        data["pillar_weights"] = pw

    if (REPORTS / "regime_log.csv").exists():
        rg = pd.read_csv(REPORTS / "regime_log.csv", parse_dates=["date"])
        data["regime"] = rg

    if (REPORTS / "alpha_rolling_ir_12m.csv").exists():
        ir = pd.read_csv(REPORTS / "alpha_rolling_ir_12m.csv", parse_dates=["date"]).set_index("date")
        data["rolling_ir_12m"] = ir

    return data


def _money_fmt(x, pos):
    if x >= 1e6:
        return f"${x/1e6:.2f}M"
    if x >= 1e3:
        return f"${x/1e3:.0f}k"
    return f"${x:.0f}"


def plot_nav_growth(data):
    """1. $1M Growth Curve."""
    nav = data["nav"]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(nav.index, nav["alpha"], label="Alpha ETF (Iter-10)", color=ALPHA_COLOR, linewidth=2.2)
    ax.plot(nav.index, nav["iwb"], label="IWB (Russell 1000)", color=IWB_COLOR, linewidth=2.2, linestyle="--")
    ax.fill_between(nav.index, nav["alpha"], nav["iwb"],
                    where=nav["alpha"] >= nav["iwb"], color=GREEN, alpha=0.15, label="Alpha leads")
    ax.fill_between(nav.index, nav["alpha"], nav["iwb"],
                    where=nav["alpha"] < nav["iwb"], color=RED, alpha=0.10, label="IWB leads")

    end_alpha = nav["alpha"].iloc[-1]
    end_iwb = nav["iwb"].iloc[-1]
    ax.annotate(f"${end_alpha/1e6:.3f}M", xy=(nav.index[-1], end_alpha),
                xytext=(15, 10), textcoords="offset points",
                fontsize=12, fontweight="bold", color=ALPHA_COLOR)
    ax.annotate(f"${end_iwb/1e6:.3f}M", xy=(nav.index[-1], end_iwb),
                xytext=(15, -18), textcoords="offset points",
                fontsize=12, fontweight="bold", color=IWB_COLOR)

    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr_alpha = (end_alpha / 1_000_000) ** (1 / n_years) - 1
    cagr_iwb = (end_iwb / 1_000_000) ** (1 / n_years) - 1
    ax.set_title(f"$1,000,000 Growth: {nav.index[0].date()} → {nav.index[-1].date()} ({n_years:.1f} years)\n"
                 f"Alpha CAGR: {cagr_alpha*100:.2f}%   |   IWB CAGR: {cagr_iwb*100:.2f}%   |   "
                 f"Outperformance: {(cagr_alpha-cagr_iwb)*100:+.2f}%/yr")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.axhline(1_000_000, color="gray", alpha=0.3, linestyle=":", linewidth=1)
    fig.tight_layout()
    out = PLOTS_DIR / "01_nav_growth_1M.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}  →  Alpha: ${end_alpha:,.0f}, IWB: ${end_iwb:,.0f}")
    return {"alpha_end": end_alpha, "iwb_end": end_iwb,
            "cagr_alpha": cagr_alpha, "cagr_iwb": cagr_iwb}


def plot_drawdown(data):
    """2. Drawdown Comparison."""
    nav = data["nav"]
    dd_alpha = nav["alpha"] / nav["alpha"].cummax() - 1
    dd_iwb = nav["iwb"] / nav["iwb"].cummax() - 1

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(dd_alpha.index, dd_alpha * 100, 0, color=ALPHA_COLOR, alpha=0.4, label=f"Alpha (max {dd_alpha.min()*100:.1f}%)")
    ax.fill_between(dd_iwb.index, dd_iwb * 100, 0, color=IWB_COLOR, alpha=0.3, label=f"IWB (max {dd_iwb.min()*100:.1f}%)")
    ax.plot(dd_alpha.index, dd_alpha * 100, color=ALPHA_COLOR, linewidth=1.5)
    ax.plot(dd_iwb.index, dd_iwb * 100, color=IWB_COLOR, linewidth=1.5, linestyle="--")
    ax.set_title("Drawdown: Alpha ETF vs IWB")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    out = PLOTS_DIR / "02_drawdown_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}  →  Alpha max DD: {dd_alpha.min()*100:.2f}%, IWB max DD: {dd_iwb.min()*100:.2f}%")


def plot_annual_returns(data):
    """3. Annual Returns Bar Chart."""
    nav = data["nav"]
    rets_a = nav["alpha"].resample("YE").last().pct_change()
    rets_i = nav["iwb"].resample("YE").last().pct_change()
    df = pd.concat([rets_a.rename("alpha"), rets_i.rename("iwb")], axis=1).dropna()
    df.index = df.index.year

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(df))
    w = 0.38
    bars_a = ax.bar(x - w/2, df["alpha"] * 100, w, label="Alpha ETF", color=ALPHA_COLOR)
    bars_i = ax.bar(x + w/2, df["iwb"] * 100, w, label="IWB", color=IWB_COLOR, alpha=0.85)
    for bar, val in zip(bars_a, df["alpha"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (0.4 if h >= 0 else -1.2),
                f"{val*100:.1f}%", ha="center", fontsize=9, color=ALPHA_COLOR, fontweight="bold")
    for bar, val in zip(bars_i, df["iwb"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (0.4 if h >= 0 else -1.2),
                f"{val*100:.1f}%", ha="center", fontsize=9, color=IWB_COLOR, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Annual Returns: Alpha ETF vs IWB")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Year")
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "03_annual_returns.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return df


def plot_sector_tilt(data):
    """4. Sector Tilt Over Time."""
    if "sector_weights" not in data:
        print("  ⚠ sector_weights not found — skipped")
        return
    sec = data["sector_weights"].copy()
    # Sort by mean weight (largest first)
    order = sec.mean().sort_values(ascending=False).index.tolist()
    sec = sec[order]

    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(1, len(order))) for i in range(len(order))]
    ax.stackplot(sec.index, sec.T.values * 100, labels=order, colors=colors, alpha=0.92)
    ax.set_title("Alpha ETF Sector Allocation Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    out = PLOTS_DIR / "04_sector_allocation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


def plot_pillar_ic(data):
    """5. Pillar IC Evolution."""
    if "pillar_ic" not in data:
        print("  ⚠ pillar_ic not found — skipped")
        return
    df = data["pillar_ic"].copy()
    df = df.sort_values("asof").set_index("asof")

    pillars = ["technical", "fundamental", "sentiment", "lowvol"]
    colors = {"technical": ALPHA_COLOR, "fundamental": RED,
              "sentiment": ACCENT, "lowvol": GREEN}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, p in zip(axes.flat, pillars):
        if p not in df.columns:
            continue
        s = df[p]
        rolling = s.rolling(6, min_periods=3).mean()
        bars = ax.bar(s.index, s.values, width=20, color=[colors[p] if v >= 0 else RED for v in s.values],
                      alpha=0.55, edgecolor="none")
        ax.plot(s.index, rolling, color="black", linewidth=2, label="6-event rolling mean")
        ax.axhline(0, color="black", linewidth=0.6)
        mean = s.dropna().mean()
        ax.axhline(mean, color=colors[p], linestyle="--", linewidth=1.2,
                   label=f"Mean: {mean:+.4f}")
        hit = (s.dropna() > 0).mean() * 100
        ax.set_title(f"{p.capitalize()} IC  (hit rate: {hit:.1f}%)")
        ax.set_ylabel("IC")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Pillar Information Coefficients Over Time", fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = PLOTS_DIR / "05_pillar_ic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


def plot_pillar_weights(data):
    """6. Dynamic Pillar Weights."""
    if "pillar_weights" not in data:
        print("  ⚠ pillar_weights not found — skipped")
        return
    df = data["pillar_weights"].copy().sort_values("asof").set_index("asof")
    pillars = [c for c in ["technical", "fundamental", "sentiment", "lowvol", "macro"] if c in df.columns]
    colors = {"technical": ALPHA_COLOR, "fundamental": RED, "sentiment": ACCENT,
              "lowvol": GREEN, "macro": "#7F4F8C"}

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.stackplot(df.index, *[df[p] * 100 for p in pillars], labels=pillars,
                 colors=[colors[p] for p in pillars], alpha=0.85)
    ax.set_title("Dynamic Pillar Weight Allocation (IC-blended)")
    ax.set_ylabel("Weight (%)")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 100)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)
    fig.tight_layout()
    out = PLOTS_DIR / "06_pillar_weights.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


def plot_rolling_metrics(data):
    """7. Rolling Sharpe + IR."""
    nav = data["nav"]
    daily_a = nav["alpha"].pct_change().dropna()
    daily_i = nav["iwb"].pct_change().dropna()
    excess = daily_a - daily_i
    win = 252  # 1-year rolling

    rolling_sharpe_a = daily_a.rolling(win).mean() / daily_a.rolling(win).std() * np.sqrt(252)
    rolling_sharpe_i = daily_i.rolling(win).mean() / daily_i.rolling(win).std() * np.sqrt(252)
    rolling_ir = excess.rolling(win).mean() / excess.rolling(win).std() * np.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(rolling_sharpe_a.index, rolling_sharpe_a, color=ALPHA_COLOR, linewidth=2, label="Alpha ETF")
    axes[0].plot(rolling_sharpe_i.index, rolling_sharpe_i, color=IWB_COLOR, linewidth=2, linestyle="--", label="IWB")
    axes[0].set_title("Rolling 12-Month Sharpe Ratio")
    axes[0].set_ylabel("Sharpe")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].legend(loc="upper left")

    axes[1].fill_between(rolling_ir.index, rolling_ir, 0, color=ALPHA_COLOR,
                         where=rolling_ir >= 0, alpha=0.45, label="IR > 0 (outperforming)")
    axes[1].fill_between(rolling_ir.index, rolling_ir, 0, color=RED,
                         where=rolling_ir < 0, alpha=0.30, label="IR < 0 (underperforming)")
    axes[1].plot(rolling_ir.index, rolling_ir, color="black", linewidth=1.2)
    axes[1].set_title("Rolling 12-Month Information Ratio (Alpha vs IWB)")
    axes[1].set_ylabel("Information Ratio")
    axes[1].set_xlabel("Date")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    out = PLOTS_DIR / "07_rolling_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


def plot_sector_compare(data):
    """8. Final Sector Allocation: Alpha vs IWB-implied (most recent)."""
    if "sector_weights" not in data:
        return
    sec = data["sector_weights"].copy()
    final = sec.iloc[-1].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(final.index, final.values * 100, color=ALPHA_COLOR, alpha=0.85)
    for bar, v in zip(bars, final.values):
        ax.text(v * 100 + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v*100:.1f}%", va="center", fontsize=9)
    ax.set_title(f"Alpha ETF Sector Allocation @ {sec.index[-1].date()}")
    ax.set_xlabel("Weight (%)")
    fig.tight_layout()
    out = PLOTS_DIR / "08_sector_final.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


def plot_strategy_diagram():
    """9. Strategy Flow Diagram (visual)."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    def box(x, y, w, h, text, color, fontsize=10, fontcolor="white"):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           edgecolor="black", facecolor=color, linewidth=1.5)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=fontcolor, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))

    # Title
    ax.text(7, 8.6, "Alpha ETF (Iter-10) Strategy Flow",
            ha="center", fontsize=16, fontweight="bold")
    ax.text(7, 8.15, "Russell 1000 Universe → Multi-Factor Composite → IC-Weighted Sleeve → Core+Satellite",
            ha="center", fontsize=11, color="#555", style="italic")

    # Layer 1: Universe
    box(5.5, 6.8, 3, 0.7, "Russell 1000 Universe\n(~1000 names, semi-annual reconstitution)", "#34495E")

    # Layer 2: Pillars
    pillars = [
        (0.3, 5.0, 2.6, 1.0, "TECHNICAL (45%)\nMomentum 12-1, 6-1\nSharpe-mom, Idio-mom", ALPHA_COLOR),
        (3.2, 5.0, 2.6, 1.0, "LOWVOL (40%)\n252-day vol\nQuality-stability tilt", GREEN),
        (6.1, 5.0, 2.6, 1.0, "SENTIMENT (10%)\n90d news polarity\nHalf-life decay", ACCENT),
        (9.0, 5.0, 2.6, 1.0, "MACRO (5%)\nVIX, slope, Fed\nRegime → sector tilt", "#7F4F8C"),
        (11.9, 5.0, 1.8, 1.0, "FUNDAMENTAL\n(DROPPED — IC<0)", "#999"),
    ]
    for (x, y, w, h, t, c) in pillars:
        box(x, y, w, h, t, c, fontsize=9)
        if c != "#999":
            arrow(x + w/2, y + h, x + w/2, 6.8)

    # Layer 3: Composite
    box(4, 3.5, 6, 0.7, "WEIGHTED COMPOSITE SCORE\nDynamic IC blending (12-event lookback) + Quality×Momentum gate (0.25)",
        "#2C3E50", fontsize=10)
    arrow(2, 5.0, 5, 4.2)
    arrow(4.5, 5.0, 6, 4.2)
    arrow(7.4, 5.0, 7, 4.2)
    arrow(9.9, 5.0, 8, 4.2)

    # Layer 4: Portfolio Construction
    box(0.5, 2.0, 4.5, 1.0, "PORTFOLIO OPTIMIZER\n• Top quintile filter (20%)\n• Max name 5%, beta 1.0±0.10\n• Sector active ±3%", "#1B4F72", fontsize=9)
    box(5.5, 2.0, 4.5, 1.0, "RISK CONSTRAINTS\n• λ_risk = 5.0\n• λ_turnover = 4.0\n• Score-band: 0.25 (no churn)", "#1B4F72", fontsize=9)
    box(10.5, 2.0, 3.0, 1.0, "TILT SLEEVE\nLong-only\n~190 names", "#1B4F72", fontsize=9)
    arrow(5, 3.5, 2.7, 3.0)
    arrow(7, 3.5, 7.7, 3.0)
    arrow(9, 3.5, 12, 3.0)

    # Layer 5: Final Portfolio
    box(2, 0.4, 4, 1.0, "CORE\n55% IWB (Russell 1000 ETF)\nPassive market exposure", IWB_COLOR, fontsize=10)
    box(8, 0.4, 4, 1.0, "SATELLITE TILT\n45% Active multi-factor\nFrom optimizer", ALPHA_COLOR, fontsize=10)
    arrow(4, 2.0, 4, 1.4)
    arrow(11, 2.0, 11, 1.4)

    # Plus sign between core and satellite
    ax.text(7, 0.9, "+", ha="center", va="center", fontsize=36, fontweight="bold", color="#888")

    fig.tight_layout()
    out = PLOTS_DIR / "09_strategy_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


def plot_regime_overlay(data):
    """10. Regime Overlay on NAV."""
    if "regime" not in data:
        return
    nav = data["nav"]
    rg = data["regime"].copy().set_index("date")
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(nav.index, nav["alpha"] / 1e6, color=ALPHA_COLOR, linewidth=2, label="Alpha ETF ($M)")
    ax.plot(nav.index, nav["iwb"] / 1e6, color=IWB_COLOR, linewidth=2, linestyle="--", label="IWB ($M)")

    # Shade regimes
    if "regime" in rg.columns:
        cur_regime = None
        start = None
        for dt, row in rg.iterrows():
            r = row["regime"]
            if r != cur_regime:
                if cur_regime is not None and start is not None:
                    if cur_regime == "risk_off":
                        ax.axvspan(start, dt, color=RED, alpha=0.10)
                    elif cur_regime == "risk_on":
                        ax.axvspan(start, dt, color=GREEN, alpha=0.08)
                cur_regime = r
                start = dt
        if cur_regime == "risk_off" and start is not None:
            ax.axvspan(start, rg.index[-1], color=RED, alpha=0.10)
        elif cur_regime == "risk_on" and start is not None:
            ax.axvspan(start, rg.index[-1], color=GREEN, alpha=0.08)

    ax.set_title("NAV with Macro Regime Overlay\n(green=risk-on, red=risk-off, white=neutral)")
    ax.set_ylabel("NAV ($M)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = PLOTS_DIR / "10_nav_regime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


def plot_summary_card(kpis, growth):
    """11. KPI Summary Card."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    ax.text(6, 6.5, "Alpha ETF (Iter-10) — 10-Year Backtest Summary",
            ha="center", fontsize=18, fontweight="bold")
    ax.text(6, 6.0, "$1,000,000 Invested 2015 → 2025",
            ha="center", fontsize=12, color="#666", style="italic")

    # Metrics grid
    metrics = [
        ("CAGR", f"{kpis['cagr']*100:.2f}%", f"{kpis['benchmark_cagr']*100:.2f}%",
         f"{(kpis['cagr']-kpis['benchmark_cagr'])*100:+.2f}%/yr"),
        ("Sharpe Ratio", f"{kpis['sharpe']:.3f}", "—", "—"),
        ("Max Drawdown", f"{kpis['max_drawdown']*100:.2f}%", "—", "—"),
        ("Annual Volatility", f"{kpis['ann_vol']*100:.2f}%", "—", "—"),
        ("Tracking Error", f"{kpis['tracking_error']*100:.2f}%", "—", "—"),
        ("Information Ratio", f"{kpis['information_ratio']:+.3f}", "—", "—"),
        ("Final Value", f"${growth['alpha_end']:,.0f}", f"${growth['iwb_end']:,.0f}",
         f"${growth['alpha_end']-growth['iwb_end']:+,.0f}"),
    ]

    headers = ["Metric", "Alpha ETF", "IWB", "Difference"]
    col_x = [0.5, 4.5, 7.5, 10.0]
    for i, h in enumerate(headers):
        ax.text(col_x[i], 5.3, h, fontsize=12, fontweight="bold", color="#333")
    ax.axhline(5.15, xmin=0.04, xmax=0.96, color="#333", linewidth=1)

    y = 4.6
    for label, alpha_v, iwb_v, diff in metrics:
        ax.text(col_x[0], y, label, fontsize=11, fontweight="bold")
        ax.text(col_x[1], y, alpha_v, fontsize=11, color=ALPHA_COLOR, fontweight="bold")
        ax.text(col_x[2], y, iwb_v, fontsize=11, color=IWB_COLOR)
        diff_color = GREEN if (diff.startswith("+") or diff.startswith("$") and not diff.startswith("$-"))             else RED if diff.startswith("-") or diff.startswith("$-") else "#333"
        ax.text(col_x[3], y, diff, fontsize=11, color=diff_color, fontweight="bold")
        y -= 0.55

    ax.text(6, 0.3, f"Strategy: 55% IWB core + 45% multi-factor tilt | Drop fundamental | Top-quintile filter | "
                    f"Dynamic IC pillar weights",
            ha="center", fontsize=10, color="#555", style="italic")

    fig.tight_layout()
    out = PLOTS_DIR / "00_summary_card.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


def main():
    print(f"Loading data from {REPORTS} ...")
    data = _load_data()
    print(f"Plots → {PLOTS_DIR}")
    print()

    growth = plot_nav_growth(data)
    plot_drawdown(data)
    annual = plot_annual_returns(data)
    plot_sector_tilt(data)
    plot_pillar_ic(data)
    plot_pillar_weights(data)
    plot_rolling_metrics(data)
    plot_sector_compare(data)
    plot_strategy_diagram()
    plot_regime_overlay(data)
    plot_summary_card(data["kpis"], growth)

    print()
    print("=" * 70)
    print("ALPHA ETF (ITER-10) — 10-YEAR PERFORMANCE SUMMARY")
    print("=" * 70)
    nav = data["nav"]
    n_yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    print(f"Backtest window:   {nav.index[0].date()} → {nav.index[-1].date()} ({n_yrs:.2f} years)")
    print(f"Initial capital:   $1,000,000")
    print(f"Final Alpha NAV:   ${growth['alpha_end']:>14,.0f}  (CAGR: {growth['cagr_alpha']*100:.2f}%)")
    print(f"Final IWB NAV:     ${growth['iwb_end']:>14,.0f}  (CAGR: {growth['cagr_iwb']*100:.2f}%)")
    print(f"Outperformance:    ${growth['alpha_end']-growth['iwb_end']:>+14,.0f}  "
          f"({(growth['cagr_alpha']-growth['cagr_iwb'])*100:+.2f}%/yr)")
    print()
    print(f"All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
