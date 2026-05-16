#!/usr/bin/env python3
"""
Side-by-side comparison: Iter-10 (monthly) vs Iter-11 (semi-annual)
Focus on: pre-tax, after-tax, turnover, tax efficiency, IC metrics
"""

import json
import pandas as pd
from pathlib import Path

REPORTS_DIR = Path("/Users/zackakshay/Desktop/ProjectX_MasterData/_px_reports")


def compare_iterations():
    """Generate detailed comparison table"""

    # Load KPIs
    kpis_files = {
        "iter10": REPORTS_DIR / "kpis_iter10.json",
        "iter11": REPORTS_DIR / "kpis_iter11.json",
    }

    kpis = {}
    for name, fpath in kpis_files.items():
        if fpath.exists():
            with open(fpath) as f:
                kpis[name] = json.load(f)
        else:
            print(f"⚠️  Missing: {fpath}")

    if len(kpis) < 2:
        print("Cannot compare: missing one or both KPI files")
        return

    k10 = kpis["iter10"]
    k11 = kpis["iter11"]

    # Compute after-tax approximations
    def after_tax(pre_tax_cagr, short_term_pct, short_rate=0.37, long_rate=0.20):
        blended = short_term_pct * short_rate + (1 - short_term_pct) * long_rate
        return pre_tax_cagr * (1 - blended), blended

    at10, tr10 = after_tax(k10["cagr"], short_term_pct=0.85)
    at11, tr11 = after_tax(k11["cagr"], short_term_pct=0.30)

    # Create comparison DataFrame
    comparison = pd.DataFrame(
        {
            "Metric": [
                "CAGR (Pre-Tax)",
                "Benchmark CAGR",
                "Active Return",
                "Estimated After-Tax CAGR",
                "Blended Tax Rate",
                "Sharpe Ratio",
                "Information Ratio",
                "Tracking Error",
                "Max Drawdown",
                "Annualized Vol",
            ],
            "Iter-10 (Monthly)": [
                f"{k10['cagr']*100:.2f}%",
                f"{k10['benchmark_cagr']*100:.2f}%",
                f"{k10['active_return']*100:.2f}%",
                f"{at10*100:.2f}%",
                f"{tr10*100:.1f}%",
                f"{k10['sharpe']:.3f}",
                f"{k10['information_ratio']:.3f}",
                f"{k10['tracking_error']*100:.2f}%",
                f"{k10['max_drawdown']*100:.1f}%",
                f"{k10['ann_vol']*100:.2f}%",
            ],
            "Iter-11 (Semi-Annual)": [
                f"{k11['cagr']*100:.2f}%",
                f"{k11['benchmark_cagr']*100:.2f}%",
                f"{k11['active_return']*100:.2f}%",
                f"{at11*100:.2f}%",
                f"{tr11*100:.1f}%",
                f"{k11['sharpe']:.3f}",
                f"{k11['information_ratio']:.3f}",
                f"{k11['tracking_error']*100:.2f}%",
                f"{k11['max_drawdown']*100:.1f}%",
                f"{k11['ann_vol']*100:.2f}%",
            ],
        }
    )

    print("\n" + "=" * 100)
    print("ITER-10 vs ITER-11 COMPARISON")
    print("=" * 100)
    print(comparison.to_string(index=False))
    print("=" * 100)

    # Additional metrics
    print("\n📊 ADDITIONAL METRICS")
    print("-" * 100)
    print(f"Pre-tax advantage (Iter-10): {(k10['cagr'] - k11['cagr'])*100:+.2f}pp")
    print(f"After-tax advantage (Iter-11): {(at11 - at10)*100:+.2f}pp")
    print(f"Tax drag savings (Iter-11): {(tr10 - tr11)*100:.1f}pp")
    print(f"\nIter-11 after-tax reaches Iter-10 after-tax level? "
          f"{'✅ YES' if at11 >= at10 else '❌ Not yet'}")

    # Save comparison
    comp_file = REPORTS_DIR / "ITER10_VS_ITER11_COMPARISON.csv"
    comparison.to_csv(comp_file, index=False)
    print(f"\n✓ Comparison saved to: {comp_file}")


if __name__ == "__main__":
    compare_iterations()
