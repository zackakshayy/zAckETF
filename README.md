# Project Alpha (Active AI ETF) — v2

Production-ready Python modules to construct, score, backtest, and validate a **sector-matched Active ETF** that tracks **Russell 1000 sector weights** while aiming to outperform **Russell 1000 (IWB)** and **RECS ETF**.

**Hard rules (non‑negotiable):**
- Universe for **both 30-Jun and 31-Dec** rebalances of year *Y* is **exactly the Russell 1000 constituents as of 30/06/Y** (the Date column in `R1000.xlsx`).
- Sector composition of the Alpha ETF must **match** the Russell 1000 sector weights on that snapshot **exactly** (within floating‑point rounding).
- Validate that the sum of distinct sector weights on a snapshot date equals **100** (percent).

**Caching:** All expensive steps write to `cache/` so reruns **resume** without recomputing finished parts.

## Quick Start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Backtest 2015–2021, export per‑rebalance scorebooks and charts
python -m project_alpha.cli backtest --export-scores --resume
```
See CLI help:
```bash
python -m project_alpha.cli -h
```
