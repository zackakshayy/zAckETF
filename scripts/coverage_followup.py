"""Follow-up analysis on top of coverage_analysis.py.

Adds three artifacts:
  1. Sector breakdown of missing-data tickers at the latest rebalance
     (bar chart + CSV).
  2. Persistently-missing tickers — names that have NO fundamental TTM
     across all 5 sample dates (these are the genuinely uncoverable cases:
     thinly-followed listings or recent IPOs without filings).
  3. A precise-NaN line chart at the 5 sample dates, plotted on the same
     time axis as the score==0 proxy from coverage_analysis.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from project_alpha.config import get_cfg

cfg = get_cfg()
RPT = Path(cfg["paths"]["reports_dir"])
OUT = RPT / "coverage"

# ------------------------------------------------------------------
# 1. Sector breakdown at the latest rebalance
# ------------------------------------------------------------------
pit_paths = [p for p in sorted(OUT.glob("coverage_pit_*.csv")) if p.name != "coverage_pit_summary.csv"]
latest_pit_path = pit_paths[-1]
latest_pit = pd.read_csv(latest_pit_path)
print(f"Loaded latest PIT coverage: {latest_pit_path.name}")

sector_breakdown = latest_pit.groupby("sector").agg(
    n=("ticker", "count"),
    n_technical_nan=("technical_nan", "sum"),
    n_fundamental_nan=("fundamental_nan", "sum"),
    n_sentiment_nan=("sentiment_nan", "sum"),
    n_all_pillars_missing=("all_pillars_missing", "sum"),
).sort_values("n_all_pillars_missing", ascending=False)

sector_breakdown["pct_all_missing"] = (sector_breakdown["n_all_pillars_missing"] / sector_breakdown["n"] * 100).round(1)
sector_breakdown.to_csv(OUT / "sector_breakdown_latest.csv")
print("\nSector breakdown of missing tickers (latest sample date):")
print(sector_breakdown)

# Stacked bar: per-sector count of tickers missing each pillar
fig, ax = plt.subplots(figsize=(11, 6))
sectors = sector_breakdown.index
x = np.arange(len(sectors))
width = 0.27
ax.bar(x - width, sector_breakdown["n_technical_nan"],   width, label="Technical NaN", color="#1f77b4")
ax.bar(x,         sector_breakdown["n_fundamental_nan"], width, label="Fundamental NaN", color="#2ca02c")
ax.bar(x + width, sector_breakdown["n_sentiment_nan"],   width, label="Sentiment NaN", color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(sectors, rotation=30, ha="right")
ax.set_ylabel("Tickers with insufficient data")
ax.set_title(f"Missing-data tickers by sector — latest sample {latest_pit_path.stem.replace('coverage_pit_', '').replace('_', '-')}")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "bar_sector_breakdown_latest.png")
plt.close(fig)
print(f"Wrote sector breakdown chart: {OUT/'bar_sector_breakdown_latest.png'}")

# ------------------------------------------------------------------
# 2. Persistently missing tickers — never had usable fundamentals
# ------------------------------------------------------------------
pit_files = sorted(OUT.glob("coverage_pit_*.csv"))
pit_files = [p for p in pit_files if p.name != "coverage_pit_summary.csv"]

frames = []
for f in pit_files:
    d = pd.read_csv(f)
    d["asof_label"] = f.stem.replace("coverage_pit_", "").replace("_", "-")
    frames.append(d)
all_pit = pd.concat(frames, ignore_index=True)

# Tickers that had fundamentals NaN at EVERY sample date they appeared in
fund_status = all_pit.groupby("ticker")["fundamental_nan"].agg(["sum", "count"])
fund_status["all_dates_missing"] = fund_status["sum"] == fund_status["count"]
persistently_missing = fund_status[fund_status["all_dates_missing"] & (fund_status["count"] >= 3)].index.tolist()

# Subset metadata
meta = all_pit.drop_duplicates(subset=["ticker"])[["ticker", "sector"]].set_index("ticker")
persistent_df = (meta.loc[meta.index.intersection(persistently_missing)]
                 .reset_index()
                 .sort_values(["sector", "ticker"]))
persistent_df.to_csv(OUT / "persistently_missing_tickers.csv", index=False)
print(f"\nPersistently missing tickers (fundamental NaN across ≥3 sample dates): {len(persistent_df)}")
print(f"Wrote: {OUT/'persistently_missing_tickers.csv'}")

# Sector breakdown of persistently-missing tickers
if not persistent_df.empty:
    persist_by_sector = persistent_df.groupby("sector").size().sort_values(ascending=False)
    print("Top sectors among persistently-missing tickers:")
    print(persist_by_sector.head(10))

# ------------------------------------------------------------------
# 3. Precise NaN line chart from sample dates (more truthful than score==0)
# ------------------------------------------------------------------
pit_summary = pd.read_csv(OUT / "coverage_pit_summary.csv", parse_dates=["asof"])
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pit_summary["asof"], pit_summary["n_universe"], "k-", linewidth=1.6, marker="o", label="Universe size")
ax.plot(pit_summary["asof"], pit_summary["n_universe"] - pit_summary["n_technical_nan"],   "-", color="#1f77b4", marker="s", label="Tickers with usable technical")
ax.plot(pit_summary["asof"], pit_summary["n_universe"] - pit_summary["n_fundamental_nan"], "-", color="#2ca02c", marker="^", label="Tickers with usable fundamental")
ax.plot(pit_summary["asof"], pit_summary["n_universe"] - pit_summary["n_sentiment_nan"],   "-", color="#d62728", marker="D", label="Tickers with usable sentiment")
ax.set_title("Precise data coverage per pillar (sample dates)\nDots = explicit NaN-input check on the underlying features")
ax.set_ylabel("Number of tickers")
ax.set_ylim(0, pit_summary["n_universe"].max() * 1.05)
ax.legend(loc="lower right", frameon=False)
ax.grid(alpha=0.25)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "line_coverage_precise.png")
plt.close(fig)
print(f"Wrote precise coverage chart: {OUT/'line_coverage_precise.png'}")

# ------------------------------------------------------------------
# 4. Quick summary printout
# ------------------------------------------------------------------
summary = pit_summary.copy()
summary["pct_tech_missing"] = (summary["n_technical_nan"] / summary["n_universe"] * 100).round(1)
summary["pct_fund_missing"] = (summary["n_fundamental_nan"] / summary["n_universe"] * 100).round(1)
summary["pct_sent_missing"] = (summary["n_sentiment_nan"] / summary["n_universe"] * 100).round(1)
print("\n=== Final summary table ===")
print(summary[["asof", "n_universe", "pct_tech_missing", "pct_fund_missing", "pct_sent_missing", "n_all_pillars_missing"]].to_string(index=False))
