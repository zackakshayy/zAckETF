"""
Post-backtest coverage and compliance analysis.

Answers three questions:

  1. How many tickers had insufficient data to develop a meaningful score in
     each of the four pillars (technical / fundamental / sentiment / macro)
     at each rebalance, and across the entire backtest window?

  2. Are the ETF constituents a subset of the Russell 1000 universe at each
     rebalance? (Compliance check.)

  3. Visualisations:
        - bar:     no-data ticker count per pillar at the most recent rebalance
        - line:    universe size + per-pillar coverage over time
        - scatter: fundamental vs technical at the latest asof (sector-coloured)
        - line:    NAV (alpha) vs benchmark (IWB) over the full window
        - bar:     regime label distribution across rebalances
        - hist:    composite-score distribution at the latest asof

Outputs:

  reports_dir/coverage/
      coverage_summary.csv           per-pillar zero counts per asof
      coverage_pit_<asof>.csv        precise per-ticker NaN flags at 5 sample dates
      r1000_compliance.csv           per-(ticker, asof) flags showing R1000 membership
      r1000_compliance_summary.csv   summary stats of compliance check
      bar_coverage_latest.png
      line_coverage_over_time.png
      scatter_fund_vs_tech_latest.png
      line_nav_vs_bench.png
      bar_regime_distribution.png
      hist_composite_latest.png

The script re-runs the per-ticker feature computation at 5 representative
dates so we can count "raw NaN" inputs exactly (the saved scores.parquet
already has fillna(0) applied, which makes 0 a fuzzy proxy for missing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- repo on path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from project_alpha.config import get_cfg
from project_alpha.data import (
    load_russell_panel, universe_for_date,
    load_eod, point_in_time_features,
    load_news_window, aggregate_sentiment_window,
)
from project_alpha.scores.technical import compute_technical_features


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

cfg = get_cfg()
paths = cfg["paths"]
RPT = Path(paths["reports_dir"])
OUT = RPT / "coverage"
OUT.mkdir(parents=True, exist_ok=True)
print(f"Writing analysis to: {OUT}")

scores = pd.read_parquet(RPT / "scores.parquet")
constituents = pd.read_parquet(RPT / "constituents.parquet")
regime_log = pd.read_csv(RPT / "regime_log.csv", parse_dates=["date"])
alpha_nav = pd.read_csv(RPT / "alpha_nav.csv", parse_dates=[0], index_col=0)
bench_nav = pd.read_csv(RPT / "benchmark_nav.csv", parse_dates=[0], index_col=0)
russell = load_russell_panel(paths["russell_xlsx"])

scores["asof"] = pd.to_datetime(scores["asof"])
constituents["asof"] = pd.to_datetime(constituents["asof"])

# ---------------------------------------------------------------------------
# 1. Per-asof "score == 0" count per pillar (cheap proxy for missing data)
# ---------------------------------------------------------------------------

ZERO_TOL = 1e-9
pillars = ["technical_score", "fundamental_score", "sentiment_score", "macro_score"]

rows = []
for asof, sub in scores.groupby("asof"):
    rec = {"asof": asof, "n_universe": len(sub)}
    for col in pillars:
        rec[f"{col}_zero"] = int((sub[col].abs() < ZERO_TOL).sum())
    rows.append(rec)
coverage_summary = pd.DataFrame(rows).sort_values("asof").reset_index(drop=True)
coverage_summary.to_csv(OUT / "coverage_summary.csv", index=False)

print("\n=== Coverage summary across {n} rebalances ===".format(n=len(coverage_summary)))
print("Universe size:")
print(f"  min={coverage_summary['n_universe'].min()},  "
      f"median={int(coverage_summary['n_universe'].median())},  "
      f"max={coverage_summary['n_universe'].max()}")
print("\nPer-pillar 'no-data' counts (score == 0 ⇒ fillna fallback ⇒ no usable signal):")
for col in pillars:
    s = coverage_summary[f"{col}_zero"]
    print(f"  {col:20s}  median={int(s.median()):4d}  max={int(s.max()):4d}  total-events-with-≥50-no-data={(s>=50).sum()}")

# ---------------------------------------------------------------------------
# 2. Precise per-ticker NaN-input check at 5 sample dates
# ---------------------------------------------------------------------------

SAMPLE_DATES = [
    pd.Timestamp("2016-06-30"),
    pd.Timestamp("2018-12-31"),
    pd.Timestamp("2020-12-31"),
    pd.Timestamp("2022-12-30"),
    pd.Timestamp("2024-12-31"),
]
# Snap to the closest scored asof
distinct_asof = sorted(scores["asof"].unique())
def _snap(d):
    return min(distinct_asof, key=lambda x: abs(x - d))
SAMPLE_DATES = [_snap(d) for d in SAMPLE_DATES]
print("\n=== Precise NaN-input check at sample dates ===")
print("Sample dates (snapped):", [d.strftime("%Y-%m-%d") for d in SAMPLE_DATES])

pit_summary_rows = []
for asof in SAMPLE_DATES:
    uni = universe_for_date(russell, asof,
                            anchor_month_day=cfg["universe"]["reconstitution_anchor"],
                            tolerance_days=cfg["universe"]["snapshot_tolerance_days"])
    tickers = list(uni["ticker"])
    sector_map = dict(zip(uni["ticker"], uni["sector"]))

    rows = []
    for t in tickers:
        # Technical
        eod = load_eod(paths["prices_dir"], t)
        tech_nan = True
        if not eod.empty:
            f = compute_technical_features(eod, asof)
            # mom_12_1 requires 12m+skip-month of history
            tech_nan = not np.isfinite(f.get("mom_12_1", float("nan")))

        # Fundamental
        last_px = None
        if not eod.empty:
            sub = eod[eod["date"] <= asof]
            if not sub.empty:
                last_px = float(sub["adjusted_close"].iloc[-1])
        ff = point_in_time_features(paths["fundamentals_dir"], t, asof, last_price=last_px)
        # "no fundamental data" if revenue TTM couldn't be assembled
        fund_nan = not ff or not np.isfinite(ff.get("revenue_ttm", float("nan")))

        # Sentiment
        nf = load_news_window(paths["news_dir"], t, asof,
                              window_days=cfg["scoring"]["sentiment"]["window_days"])
        agg = aggregate_sentiment_window(nf, asof,
            half_life_days=cfg["scoring"]["sentiment"]["half_life_days"],
            relevance_weight=cfg["scoring"]["sentiment"]["relevance_weight"],
            require_ticker_in_symbols=True,
            min_articles=cfg["scoring"]["sentiment"]["min_articles"])
        sent_nan = not np.isfinite(agg.get("polarity_decay", float("nan")))

        rows.append({
            "ticker": t,
            "sector": sector_map.get(t, "Unknown"),
            "technical_nan": tech_nan,
            "fundamental_nan": fund_nan,
            "sentiment_nan": sent_nan,
            "all_pillars_missing": tech_nan and fund_nan and sent_nan,
            "any_pillar_missing": tech_nan or fund_nan or sent_nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"coverage_pit_{asof.strftime('%Y_%m_%d')}.csv", index=False)

    summary = {
        "asof": asof,
        "n_universe": len(df),
        "n_technical_nan": int(df["technical_nan"].sum()),
        "n_fundamental_nan": int(df["fundamental_nan"].sum()),
        "n_sentiment_nan": int(df["sentiment_nan"].sum()),
        "n_all_pillars_missing": int(df["all_pillars_missing"].sum()),
        "n_any_pillar_missing": int(df["any_pillar_missing"].sum()),
    }
    pit_summary_rows.append(summary)
    print(f"\n  {asof.strftime('%Y-%m-%d')}  N={summary['n_universe']}  "
          f"tech_nan={summary['n_technical_nan']}  "
          f"fund_nan={summary['n_fundamental_nan']}  "
          f"sent_nan={summary['n_sentiment_nan']}  "
          f"all_three_missing={summary['n_all_pillars_missing']}")

pit_summary = pd.DataFrame(pit_summary_rows)
pit_summary.to_csv(OUT / "coverage_pit_summary.csv", index=False)


# ---------------------------------------------------------------------------
# 3. Russell 1000 compliance: every constituent must be in the R1000 universe
#    at the rebalance date that selected it.
# ---------------------------------------------------------------------------

print("\n=== Russell 1000 compliance check ===")
flagged_rows = []
per_event_rows = []
total_constituents = 0
total_violations = 0

for asof, sub in constituents.groupby("asof"):
    uni = universe_for_date(russell, asof,
                            anchor_month_day=cfg["universe"]["reconstitution_anchor"],
                            tolerance_days=cfg["universe"]["snapshot_tolerance_days"])
    universe_set = set(uni["ticker"])
    selected = sub["ticker"].tolist()
    in_universe = sub["ticker"].isin(universe_set)
    n_total = len(selected)
    n_violators = int((~in_universe).sum())

    total_constituents += n_total
    total_violations += n_violators
    per_event_rows.append({
        "asof": asof,
        "n_constituents": n_total,
        "n_violations": n_violators,
        "violation_pct": (n_violators / n_total * 100) if n_total else 0.0,
        "snapshot_used": uni.attrs.get("snapshot_date"),
    })
    if n_violators:
        viol = sub.loc[~in_universe, ["ticker", "sector", "weight"]].copy()
        viol["asof"] = asof
        flagged_rows.append(viol)

per_event = pd.DataFrame(per_event_rows)
per_event.to_csv(OUT / "r1000_compliance.csv", index=False)
violations = pd.concat(flagged_rows, ignore_index=True) if flagged_rows else pd.DataFrame(
    columns=["ticker", "sector", "weight", "asof"])
violations.to_csv(OUT / "r1000_compliance_violations.csv", index=False)

summary_compliance = {
    "n_rebalances": len(per_event),
    "total_constituents_across_history": total_constituents,
    "total_violations_across_history": total_violations,
    "violation_rate_pct": (total_violations / total_constituents * 100) if total_constituents else 0.0,
    "n_rebalances_with_any_violation": int((per_event["n_violations"] > 0).sum()),
    "max_violations_in_one_rebalance": int(per_event["n_violations"].max()),
}
with open(OUT / "r1000_compliance_summary.json", "w") as fh:
    json.dump(summary_compliance, fh, indent=2)
print(json.dumps(summary_compliance, indent=2))

# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# 4.1 — Bar chart: per-pillar no-data counts at the most recent rebalance
latest_asof = scores["asof"].max()
latest = scores[scores["asof"] == latest_asof]
zero_counts = {
    "Technical": int((latest["technical_score"].abs() < ZERO_TOL).sum()),
    "Fundamental": int((latest["fundamental_score"].abs() < ZERO_TOL).sum()),
    "Sentiment": int((latest["sentiment_score"].abs() < ZERO_TOL).sum()),
}
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(zero_counts.keys(), zero_counts.values(),
              color=["#1f77b4", "#2ca02c", "#d62728"])
for b, v in zip(bars, zero_counts.values()):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, str(v), ha="center", fontsize=11, fontweight="bold")
ax.set_title(f"Tickers with NO usable score per pillar — {latest_asof.strftime('%Y-%m-%d')}\n"
             f"(universe size = {len(latest)})")
ax.set_ylabel("Number of tickers")
ax.set_ylim(0, max(zero_counts.values()) * 1.18 + 1)
fig.tight_layout()
fig.savefig(OUT / "bar_coverage_latest.png")
plt.close(fig)

# 4.2 — Line: universe size and per-pillar coverage over time
fig, ax = plt.subplots(figsize=(11, 5))
cs = coverage_summary
ax.plot(cs["asof"], cs["n_universe"], color="black", linewidth=1.6, label="Universe size")
ax.plot(cs["asof"], cs["n_universe"] - cs["technical_score_zero"],
        color="#1f77b4", linewidth=1.4, label="Technical-covered")
ax.plot(cs["asof"], cs["n_universe"] - cs["fundamental_score_zero"],
        color="#2ca02c", linewidth=1.4, label="Fundamental-covered")
ax.plot(cs["asof"], cs["n_universe"] - cs["sentiment_score_zero"],
        color="#d62728", linewidth=1.4, label="Sentiment-covered")
ax.set_title("Universe size and per-pillar coverage over time")
ax.set_ylabel("Tickers")
ax.legend(loc="lower right", frameon=False)
ax.grid(alpha=0.25)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "line_coverage_over_time.png")
plt.close(fig)

# 4.3 — Scatter: fundamental vs technical at latest, coloured by sector
fig, ax = plt.subplots(figsize=(10, 7))
sectors_latest = latest["sector"].unique()
cmap = plt.cm.get_cmap("tab20", len(sectors_latest))
for i, sec in enumerate(sorted(sectors_latest)):
    sub = latest[latest["sector"] == sec]
    ax.scatter(sub["fundamental_score"], sub["technical_score"],
               s=14, alpha=0.7, color=cmap(i), label=sec)
ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
ax.set_xlabel("Fundamental score")
ax.set_ylabel("Technical score")
ax.set_title(f"Fundamental vs technical scores — {latest_asof.strftime('%Y-%m-%d')}\n"
             "Each point = one ticker; sector colour-coded")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "scatter_fund_vs_tech_latest.png")
plt.close(fig)

# 4.4 — NAV: alpha vs benchmark
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(alpha_nav.index, alpha_nav.iloc[:, 0], label="Alpha ETF", color="#1f77b4", linewidth=1.4)
ax.plot(bench_nav.index, bench_nav.iloc[:, 0], label="IWB (Russell 1000)", color="#7f7f7f", linewidth=1.2, linestyle="--")
ax.set_title("NAV — Alpha ETF vs IWB")
ax.set_ylabel("NAV ($)")
ax.legend(loc="upper left", frameon=False)
ax.grid(alpha=0.25)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "line_nav_vs_bench.png")
plt.close(fig)

# 4.5 — Regime distribution bar
regime_counts = regime_log["regime"].value_counts().reindex(["risk_off", "neutral", "risk_on"]).fillna(0).astype(int)
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(regime_counts.index, regime_counts.values,
              color=["#d62728", "#7f7f7f", "#2ca02c"])
for b, v in zip(bars, regime_counts.values):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, str(int(v)),
            ha="center", fontsize=11, fontweight="bold")
ax.set_title(f"Macro-regime distribution across {len(regime_log)} rebalances")
ax.set_ylabel("Rebalance events")
fig.tight_layout()
fig.savefig(OUT / "bar_regime_distribution.png")
plt.close(fig)

# 4.6 — Histogram: composite score at the latest asof
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(latest["composite_score"], bins=40, color="#9467bd", edgecolor="white")
ax.set_xlabel("Composite score")
ax.set_ylabel("Tickers")
ax.set_title(f"Composite-score distribution — {latest_asof.strftime('%Y-%m-%d')}")
ax.axvline(latest["composite_score"].mean(), color="black", linestyle="--", linewidth=1, label=f"mean={latest['composite_score'].mean():.3f}")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "hist_composite_latest.png")
plt.close(fig)

print(f"\nWrote 6 plots and {len(list(OUT.glob('*.csv'))) + len(list(OUT.glob('*.json')))} CSV/JSON files to {OUT}")
print("Files:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name}  ({p.stat().st_size:,} bytes)")
