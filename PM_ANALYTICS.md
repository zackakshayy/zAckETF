# PM Analytics Suite — Alpha Engine ETF

Institutional-grade exposure, risk attribution, and consistency analysis for
the Alpha Engine ETF. Designed to address the questions a sophisticated
portfolio manager will ask in a strategy review.

## What's Included

| Module | Question Answered | Output |
|--------|---------------------|--------|
| `pm_exposure_analysis.py` | Where are we taking risk? What's true alpha vs factor premia? | Carhart 4-factor + FF5 regressions, sector active weights, active share, top positions |
| `pm_tracking_error.py` | Where does the 3% TE come from? Style? Sector? Stocks? | TE decomposition (factor / sector / stock-specific), predicted vs realized TE |
| `pm_investment_thesis.py` | WHY does this work and is it reproducible across regimes? | Hit rates, up/down capture, regime alphas, rolling IR, drawdowns |

## Quick Start

After running a backtest (`python -m project_alpha.backtest.run config/alpha.yaml`),
generate the full PM report:

```bash
export PROJECT_ALPHA_REPORTS_DIR=/path/to/_px_reports
bash scripts/pm_run_all.sh
```

Outputs land in `$PROJECT_ALPHA_REPORTS_DIR/plots_pm/`:

```
plots_pm/
├── pm_01_factor_exposure.png       Carhart 4F loadings + alpha-after-factors
├── pm_02_sector_exposure.png       Sector weights vs IWB + active weights
├── pm_03_active_share.png          Active share over time
├── pm_04_top_active.png            Top 20 overweight positions
├── pm_05_te_attribution.png        TE decomposition (factor/sector/idio)
├── pm_06_hit_rate.png              Hit rate by frequency + monthly distribution
├── pm_07_capture_ratios.png        Up/down capture ratios
├── pm_08_regime_performance.png    Alpha by Bull/Bear/Sideways
├── pm_09_rolling_ir.png            Rolling 12-month IR
├── pm_10_thesis_card.png           Executive summary card
├── pm_exposure_summary.json        Machine-readable metrics
├── pm_te_summary.json
└── pm_thesis_summary.json
```

## Methodology Notes

### Factor Exposure (Module 1)

- **Carhart 4-Factor Model** (1997): regresses strategy excess returns on
  MKT (market), SMB (size), HML (value), UMD (momentum) factors from
  Ken French's data library.
- **Fama-French 5+UMD**: adds RMW (profitability/quality) and CMA (investment).
  Industry standard since 2015.
- **Active Regression**: regresses (r_strategy - r_benchmark) on factors.
  The intercept is "true active alpha" after stripping factor premia.

**Interpretation:**
- If Carhart alpha t-stat > 1.96 (95% confidence): genuine skill
- If alpha disappears in Carhart but is positive in CAPM: strategy is just
  factor-betting, replicable with cheap factor ETFs (MTUM, VLUE, USMV)

### Tracking Error Attribution (Module 2)

Decomposes realized TE into:

```
TE² (total) = β'Σβ (factor) + σ²_residual (stock-specific)
```

- **Factor TE**: variance from factor exposures × factor covariance matrix
- **Idio TE**: residual variance after stripping factors → stock-selection skill
- **Predicted TE**: ex-ante TE using trailing 1-year factor model (forecast)

**Interpretation:**
- Idio % > 50%: most TE from stock picking — good sign of skill
- Factor % > 50%: most TE from style bets — replicable cheaply

### Investment Thesis (Module 3)

Six consistency metrics that prove (or disprove) reproducibility:

| Metric | Threshold | Interpretation |
|--------|-----------|-----------------|
| Monthly hit rate | > 55% | Beats IWB consistently |
| Up capture | > 100% | Captures more upside |
| Down capture | < 100% | Loses less in downside |
| Capture asymmetry | > 0 | Favorable risk profile |
| Rolling IR > 0 | > 60% of time | Persistent alpha |
| Regime alpha | > 0 in all 3 | Robust across markets |

The thesis card (`pm_10_thesis_card.png`) synthesizes all of this into a
single-page executive summary suitable for an investment committee.

## Data Dependencies

The PM modules require these files in `reports-dir`:

- `alpha_nav.csv`, `benchmark_nav.csv` — daily NAV curves
- `alpha_sector_weights.csv` — sector weights over time
- `constituents.parquet` — per-event holdings with sector tags

Fama-French factor data is auto-downloaded from
[Ken French's data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
on first run and cached in `reports-dir/_ff_cache/`.

## Limitations & Honest Caveats

1. **Sector benchmark weights are static** (end-2024 IWB approximation).
   For full rigor, fetch time-varying IWB sector weights from BlackRock fact sheets.

2. **Stock-level Active Share** uses uniform-within-sector benchmark weights
   (~ 90 stocks per sector). The true Active Share against IWB's actual
   stock-level weights would differ, typically by ±5pp.

3. **Sector TE attribution** uses approximate sector volatilities and
   assumes independence — this slightly overestimates sector contribution.
   For exact decomposition, use Barra or Axioma risk models.

4. **Predicted TE uses 1-year trailing window** — sensitive to regime change.
   Allocators typically use a covariance shrinkage estimator (Ledoit-Wolf 2003)
   for better stability.

## What a PM Will Notice

When you present this report, expect the following follow-up questions:

1. **"What's the Carhart alpha p-value?"** — Look at the t-stat. If < 1.96,
   reasonable PMs will discount the strategy. Be honest about it.

2. **"Can you replicate this with MTUM + USMV?"** — High factor loadings
   suggest yes. The fee differential matters.

3. **"What was performance in 2008?"** — Current backtest is 2015-2025 only.
   This is the biggest gap. Backfill if possible.

4. **"What's your capacity?"** — At Russell 1000 liquidity, ~$300-500M
   before market impact eats alpha. Have a number ready.

5. **"What's your edge over QuantStrat / AQR Multi-Strat?"** — Be ready to
   articulate what's NOVEL (not just well-implemented) in your design.
