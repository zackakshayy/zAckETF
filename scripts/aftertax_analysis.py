"""
After-tax return analysis for Alpha ETF strategies.

Computes:
  - Annual portfolio turnover from rebalance data
  - Estimated short-term vs long-term capital gains split
  - Tax-adjusted CAGR using user's tax bracket
  - Comparison: monthly (Iter-10) vs semi-annual (Iter-12)
  - Net-of-tax $1M growth

Tax assumptions (US, top federal + state):
  - Short-term capital gains:  32% federal + 5% state = 37%
  - Long-term capital gains:   15% federal + 5% state = 20%
    (top bracket users would pay 20% federal + state, plus 3.8% NIIT)
  - We assume holdings sold within 365 days = STCG; > 365 days = LTCG
  - Trading cost: 8 bps per turnover (taken from config)

Usage:
  python scripts/aftertax_analysis.py \
    --reports-dir /path/to/_px_reports \
    --kpis-monthly /path/to/kpis_iter10.json \
    --kpis-semi    /path/to/kpis_iter12.json \
    --fed-rate 0.32 --state-rate 0.05
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np


def estimate_turnover_from_rebalances(rebalances_csv: Path) -> dict:
    """Estimate annual portfolio turnover from rebalance log."""
    if not rebalances_csv.exists():
        return {"annual_turnover": None, "n_rebalances": 0, "events_per_year": 0}

    df = pd.read_csv(rebalances_csv, parse_dates=["date"])
    df = df.sort_values(["date", "ticker"])

    # Group by date — each event is a snapshot
    snapshots = {dt: g.set_index("ticker")["weight"] for dt, g in df.groupby("date")}
    dates = sorted(snapshots.keys())
    if len(dates) < 2:
        return {"annual_turnover": None, "n_rebalances": len(dates), "events_per_year": 0}

    # Per-event one-way turnover = 0.5 * sum |w_new - w_old|
    one_way_turnovers = []
    for prev, cur in zip(dates[:-1], dates[1:]):
        w_prev = snapshots[prev]
        w_cur = snapshots[cur]
        all_tickers = w_prev.index.union(w_cur.index)
        diff = w_cur.reindex(all_tickers, fill_value=0.0) - w_prev.reindex(all_tickers, fill_value=0.0)
        one_way_turnovers.append(0.5 * diff.abs().sum())

    n_yrs = (dates[-1] - dates[0]).days / 365.25
    avg_one_way = np.mean(one_way_turnovers)
    events_per_yr = (len(dates) - 1) / max(n_yrs, 1e-9)
    annual_turnover = avg_one_way * events_per_yr * 2  # round-trip = 2x one-way

    return {
        "annual_turnover": float(annual_turnover),
        "avg_oneway_per_event": float(avg_one_way),
        "n_rebalances": len(dates),
        "events_per_year": float(events_per_yr),
        "n_years": float(n_yrs),
    }


def split_gains(annual_turnover: float, events_per_yr: float) -> tuple:
    """Heuristic: estimate the fraction of realized gains that are short-term.
    With monthly cadence, most positions held < 1 year → STCG.
    With semi-annual cadence, more positions held > 1 year → LTCG.
    """
    if events_per_yr <= 0:
        return 1.0, 0.0
    avg_holding_days = 365 / events_per_yr
    # Smooth interpolation: < 6 mo → 100% ST; > 18 mo → 100% LT; linear between
    if avg_holding_days <= 180:
        st_frac = 1.0
    elif avg_holding_days >= 540:
        st_frac = 0.0
    else:
        st_frac = (540 - avg_holding_days) / (540 - 180)
    return st_frac, 1.0 - st_frac


def aftertax_cagr(
    pretax_cagr: float,
    annual_turnover: float,
    st_frac: float,
    lt_frac: float,
    *,
    st_rate: float,
    lt_rate: float,
    trading_cost_bps: float = 8,
) -> dict:
    """Estimate after-tax CAGR.

    Model: each year, a fraction of portfolio = annual_turnover is sold &
    re-bought. Realized gain on the sold portion ≈ pretax_cagr * fraction held.
    We approximate annual realized gain rate as: turnover × pretax_return_rate.
    Tax drag = realized_gain_rate * blended_tax_rate.
    Plus trading cost drag = annual_turnover * trading_cost_bps.
    """
    blended_tax = st_frac * st_rate + lt_frac * lt_rate
    realized_gain_rate = max(0.0, pretax_cagr) * annual_turnover
    tax_drag = realized_gain_rate * blended_tax
    cost_drag = annual_turnover * (trading_cost_bps / 10_000.0)
    aftertax = pretax_cagr - tax_drag - cost_drag
    return {
        "blended_tax_rate": blended_tax,
        "annual_realized_gain_rate": realized_gain_rate,
        "tax_drag": tax_drag,
        "trading_cost_drag": cost_drag,
        "aftertax_cagr": aftertax,
    }


def main():
    p = argparse.ArgumentParser(description="After-tax return analysis")
    p.add_argument("--reports-dir", default=os.environ.get("PROJECT_ALPHA_REPORTS_DIR", "."))
    p.add_argument("--kpis-monthly", required=True, help="Path to monthly KPIs json (e.g. kpis_iter10.json)")
    p.add_argument("--kpis-semi",    required=True, help="Path to semi-annual KPIs json (e.g. kpis_iter12.json)")
    p.add_argument("--rebalances-monthly", default=None, help="Optional rebalance CSV for monthly run")
    p.add_argument("--rebalances-semi",    default=None, help="Optional rebalance CSV for semi run")
    p.add_argument("--fed-rate", type=float, default=0.32, help="Federal income tax rate")
    p.add_argument("--state-rate", type=float, default=0.05, help="State income tax rate")
    p.add_argument("--ltcg-fed",   type=float, default=0.15, help="Long-term cap gains federal rate")
    p.add_argument("--initial",    type=float, default=1_000_000)
    args = p.parse_args()

    reports = Path(args.reports_dir)
    monthly = json.load(open(args.kpis_monthly))
    semi    = json.load(open(args.kpis_semi))

    st_rate = args.fed_rate + args.state_rate            # short-term: ordinary income
    lt_rate = args.ltcg_fed  + args.state_rate           # long-term: 15% fed + state

    n_yrs_m = monthly.get("n_trading_days", 252*10) / 252
    n_yrs_s = semi.get("n_trading_days", 252*10) / 252

    # Heuristic event counts (we know from runs)
    events_monthly_per_yr = 12.0      # iter-10
    events_semi_per_yr    = 2.0       # iter-12

    # Estimate turnover (defaults if no rebalances csv given)
    turnover_monthly = 0.50           # ~50%/yr typical for monthly factor
    turnover_semi    = 0.20           # ~20%/yr for semi-annual (signal decay caps it)

    if args.rebalances_monthly:
        td = estimate_turnover_from_rebalances(Path(args.rebalances_monthly))
        if td["annual_turnover"] is not None:
            turnover_monthly = td["annual_turnover"]
            events_monthly_per_yr = td["events_per_year"]
    if args.rebalances_semi:
        td = estimate_turnover_from_rebalances(Path(args.rebalances_semi))
        if td["annual_turnover"] is not None:
            turnover_semi = td["annual_turnover"]
            events_semi_per_yr = td["events_per_year"]

    st_m, lt_m = split_gains(turnover_monthly, events_monthly_per_yr)
    st_s, lt_s = split_gains(turnover_semi, events_semi_per_yr)

    at_m = aftertax_cagr(monthly["cagr"], turnover_monthly, st_m, lt_m,
                         st_rate=st_rate, lt_rate=lt_rate)
    at_s = aftertax_cagr(semi["cagr"],    turnover_semi,    st_s, lt_s,
                         st_rate=st_rate, lt_rate=lt_rate)
    # Benchmark: assume ~3% turnover, all LT after first year
    at_b = aftertax_cagr(monthly["benchmark_cagr"], 0.03, 0.0, 1.0,
                         st_rate=st_rate, lt_rate=lt_rate)

    print("=" * 78)
    print(f"  AFTER-TAX RETURN ANALYSIS  (Fed {args.fed_rate*100:.0f}% + State {args.state_rate*100:.0f}%)")
    print("=" * 78)
    print(f"  Short-term cap gains rate: {st_rate*100:.1f}%")
    print(f"  Long-term  cap gains rate: {lt_rate*100:.1f}%")
    print()
    print(f"{'':32}{'Iter-10 monthly':>18}{'Iter-12 semi-annual':>22}{'IWB':>10}")
    print(f"{'-'*82}")
    print(f"{'Pre-tax CAGR':32}{monthly['cagr']*100:>17.2f}%{semi['cagr']*100:>21.2f}%{monthly['benchmark_cagr']*100:>9.2f}%")
    print(f"{'Annual turnover (round-trip)':32}{turnover_monthly*100:>17.0f}%{turnover_semi*100:>21.0f}%{3:>9.0f}%")
    print(f"{'Events per year':32}{events_monthly_per_yr:>18.1f}{events_semi_per_yr:>22.1f}{0:>10.1f}")
    print(f"{'Avg holding days':32}{365/max(events_monthly_per_yr,1e-9):>18.0f}{365/max(events_semi_per_yr,1e-9):>22.0f}{'>365':>10}")
    print(f"{'  → ST gain fraction':32}{st_m*100:>17.0f}%{st_s*100:>21.0f}%{0:>9.0f}%")
    print(f"{'  → LT gain fraction':32}{lt_m*100:>17.0f}%{lt_s*100:>21.0f}%{100:>9.0f}%")
    print(f"{'Blended tax rate':32}{at_m['blended_tax_rate']*100:>17.1f}%{at_s['blended_tax_rate']*100:>21.1f}%{at_b['blended_tax_rate']*100:>9.1f}%")
    print(f"{'Tax drag':32}{at_m['tax_drag']*100:>17.2f}%{at_s['tax_drag']*100:>21.2f}%{at_b['tax_drag']*100:>9.2f}%")
    print(f"{'Trading cost drag':32}{at_m['trading_cost_drag']*100:>17.2f}%{at_s['trading_cost_drag']*100:>21.2f}%{at_b['trading_cost_drag']*100:>9.2f}%")
    print(f"{'-'*82}")
    print(f"{'AFTER-TAX CAGR':32}{at_m['aftertax_cagr']*100:>17.2f}%{at_s['aftertax_cagr']*100:>21.2f}%{at_b['aftertax_cagr']*100:>9.2f}%")
    print()

    # $1M growth comparison
    fv = lambda c, y: args.initial * ((1 + c) ** y)
    n_yr = round((n_yrs_m + n_yrs_s) / 2, 2)
    print(f"  ${args.initial:,.0f} grown over {n_yr:.2f} years AFTER-TAX:")
    print(f"     Iter-10 monthly:       ${fv(at_m['aftertax_cagr'], n_yr):>14,.0f}")
    print(f"     Iter-12 semi-annual:   ${fv(at_s['aftertax_cagr'], n_yr):>14,.0f}")
    print(f"     IWB (passive):         ${fv(at_b['aftertax_cagr'], n_yr):>14,.0f}")
    print()
    print(f"  Semi-annual vs monthly (after-tax):  ${fv(at_s['aftertax_cagr'], n_yr) - fv(at_m['aftertax_cagr'], n_yr):>+14,.0f}")
    print(f"  Semi-annual vs IWB     (after-tax):  ${fv(at_s['aftertax_cagr'], n_yr) - fv(at_b['aftertax_cagr'], n_yr):>+14,.0f}")
    print()
    print("=" * 78)
    print("  Notes / caveats:")
    print("  - Holdings in tax-deferred accounts (401k/IRA) bypass these taxes entirely.")
    print("  - Top-bracket investors should also factor 3.8% NIIT (Net Investment Income Tax).")
    print("  - This model assumes annual realized gain ≈ pretax return × turnover. Real")
    print("    accounts may carry losses forward, harvest losses, and use long-term lots.")
    print("  - Trading costs assumed at 8 bps/turn (matches config). Spread + slippage")
    print("    on illiquid names could add another 5-15 bps.")
    print("=" * 78)


if __name__ == "__main__":
    main()
