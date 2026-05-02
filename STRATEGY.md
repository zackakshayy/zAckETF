# Project Alpha — How the ETF Is Built

This document walks through the end-to-end construction of the Alpha ETF as
implemented in `project_alpha_code_v2_1`. It covers the data, the four pillar
scores, the regime overlay, the rebalance cadence, the optimizer, and the
backtest. Every section links to the file and function where the logic lives
so you can read the code alongside the description.

---

## 1. Strategy in one paragraph

We construct an actively managed ETF whose universe is the Russell 1000.
Each holding gets a **composite score** that blends four pillars:
**fundamentals, technicals, sentiment, and a macro regime overlay.** The four
pillar weights themselves shift with the macro regime — risk-off raises
fundamentals + macro and damps technicals + sentiment, risk-on does the
reverse. We then run a sector-matched, beta-banded, ADV-capped optimizer
that tilts toward high-composite-score names while staying close to the
Russell 1000 sector profile. The benchmark is **IWB** (the Russell 1000 ETF).

---

## 2. Inputs and data layer

All paths are configured in [config/alpha.yaml](config/alpha.yaml) under `paths`.
The actual files live outside the repo at `<DATA_ROOT>/...` — see
[config/alpha.yaml.example](config/alpha.yaml.example) for the expected
directory layout.

| Source | Files | What we use it for |
|---|---|---|
| Universe | `R1000.xlsx` (sheet `R1K`) | Russell 1000 holdings, weight, sector ([data/universe.py](project_alpha/data/universe.py)) |
| Prices  | `*.US.eod.parquet`, `*.US.dividends.parquet`, `*.US.splits.parquet` | Returns, vol, momentum, ADV ([data/prices.py](project_alpha/data/prices.py)) |
| Fundamentals | `*.US.fundamentals.json` (EOD-Historical schema) | Quality / value / growth / leverage ([data/fundamentals.py](project_alpha/data/fundamentals.py)) |
| News | `TICKER.US_YYYY-MM.news.json` | Pre-scored polarity for sentiment ([data/news.py](project_alpha/data/news.py)) |
| Macro | FRED CSVs: `VIXCLS`, `T10Y2Y`, `FEDFUNDS`, `TB3MS`, `CPIAUCSL`, `UNRATE`, `GDPC1` | Regime classification ([data/macro.py](project_alpha/data/macro.py)) |
| Benchmark | `IWB.US.eod.parquet` | Daily returns for tracking error / IR |

### Look-ahead protection

The data layer is **point-in-time aware**:

- **Fundamentals**: every quarterly statement carries a `filing_date`. Our
  TTM rebuilder filters by `filing_date <= asof` (or `period_end + 45 days`
  if filing_date is missing) before summing, so we never see a 10-Q until
  it has been filed. See `_statement_frame` and `_ttm_sum` in
  [data/fundamentals.py:121](project_alpha/data/fundamentals.py).
- **News**: we keep only articles dated `<= asof` and within the
  `window_days` window. See [data/news.py:111](project_alpha/data/news.py).
- **Macro**: `asof_macro` forward-fills only up to `asof`, never beyond.
  Derived series (`CPI_YOY`, `FEDFUNDS_DIFF_3M`) are computed on the
  **native** monthly series before joining, so a 3-month diff is a true
  3-calendar-month diff, not 3 panel rows. See [data/macro.py:48](project_alpha/data/macro.py).
- **Prices**: returns and volatility are computed on `adjusted_close` (split-
  and dividend-adjusted), restricted to dates `<= asof`.

---

## 3. Universe — semi-annual reconstitution

Per the strategy spec, **both the June and December rebalances of year Y
use the Russell 1000 snapshot dated 30-Jun-Y**.
[`universe_for_date`](project_alpha/data/universe.py) implements this:

1. Compute the anchor target date `Y-06-30` for the year of `asof`.
2. Pick the latest snapshot in the panel `<=` that target.
3. If none found, fall back to the nearest within `tolerance_days` (default 14).
4. Drop the configured asset types (default `[Cash, Bond]`).
5. Renormalize the Port. Ending Weight column to sum to 1.0 after exclusions.
6. Sector tagging comes from R1000's `GICS Sector Extended` column.

> **Data limitation**: the workbook only contains 10 annual snapshots
> (2015-06-30 → 2024-06-28). For any rebalance after Jun 2024, the
> universe falls back to the 2024-06-28 snapshot. This is logged.

The orchestrator overrides each row's sector with the R1000 mapping before
sector-z-scoring downstream, because many fundamentals JSONs have
`General.GicSector = null` (only the non-GICS `Sector` field is populated).
See `_refresh_fundamental_panel` in [backtest/run.py:75](project_alpha/backtest/run.py).

---

## 4. The four pillar scores

Every pillar produces a **cross-sectional, sector-neutral, winsorized
z-score panel** (one row per ticker). The four panels are then combined.

A common helper, `_z_by_sector`, applies winsorization at ±3σ and z-scores
within each GICS sector when the sector has at least 3 names; smaller
sectors get the global z-score as fallback.

### 4.1 Fundamental score

Implemented in [scores/fundamental.py](project_alpha/scores/fundamental.py),
fed by [data/fundamentals.py:point_in_time_features](project_alpha/data/fundamentals.py:151).

Per ticker we rebuild TTM aggregates from quarterly statements
(`Income_Statement`, `Balance_Sheet`, `Cash_Flow`):

| Sub-score | Inputs | Direction |
|---|---|---|
| Quality   | `roe`, `roa`, `gross_margin`, `operating_margin`, `net_margin`, `fcf_margin` | higher is better |
| Value     | `1/pe`, `1/pb`, `1/ps`, `1/ev_ebitda`, `fcf_yield` | inverted multiples then higher is better |
| Growth    | `rev_growth_yoy`, `eps_growth_yoy` | higher is better |
| Leverage  | `-tanh(debt_to_equity)` | lower D/E is better |

Each input is sector-z-scored. Sub-scores are equally weighted within
their group, then blended:

```
fundamental_score = 0.40·Quality_z + 0.30·Value_z + 0.20·Growth_z + 0.10·Leverage_z
```

The result is sector-z-scored once more so the pillar output is on a
unit-z scale.

**Why we recompute TTM instead of using the JSON's pre-baked Highlights**:
the Highlights block (e.g. `ProfitMargin`, `OperatingMarginTTM`) is the
**current** snapshot — using it in a 2018 backtest would inject look-ahead.
We prefer the slower-but-correct quarterly rebuild for backtests; the
`load_fundamental_features` helper exists for live (current-day) scoring
where look-ahead is not a concern.

### 4.2 Technical score

Implemented in [scores/technical.py](project_alpha/scores/technical.py).
Inputs come from each ticker's EOD parquet (`adjusted_close`, `volume`).

Per-ticker features as of `asof`:

| Feature | Definition |
|---|---|
| `mom_12_1`   | 252-day return ending 21 days before `asof` (skip-month) |
| `mom_6_1`    | 126-day return ending 21 days before `asof` |
| `rev_21d`    | negative of last 21-day return (short-term reversal) |
| `vol_252d`   | 1y realized vol of daily returns × √252 |
| `high52_gap` | (price − 52w high) / 52w high  (≤ 0) |
| `ma_200_gap` | (price − 200d MA) / 200d MA |
| `liq_trend`  | EMA20(price·volume) / EMA60(price·volume) − 1 |

Each feature is sector-z-scored. Combined as:

```
technical_score = 0.50 · MomentumZ
                – 0.20 · VolZ
                + 0.10 · ReversalZ
                + 0.10 · High52Z
                + 0.10 · LiquidityZ
```

where `MomentumZ = 0.7·z(mom_12_1) + 0.3·z(mom_6_1)`. Final result is
again sector-z-scored.

The 12-1 (skip-month) construction is the standard Jegadeesh-Titman
momentum that excludes the reversal-prone last month.

### 4.3 Sentiment score

Implemented in [scores/sentiment.py](project_alpha/scores/sentiment.py).
Inputs come from [data/news.py](project_alpha/data/news.py).

For each ticker, over a 90-day window ending at `asof`:

1. Load every monthly news JSON file that intersects the window.
2. Keep articles whose `date` is in `[asof − 90d, asof]` and whose
   pre-computed `sentiment.polarity` is finite.
3. Filter to articles where the ticker actually appears in `symbols`
   (vendor's stringified list parsed defensively).
4. Compute weighted statistics:
   - `polarity_decay = Σ wᵢ·polarityᵢ / Σ wᵢ` where
     `wᵢ = 2^(−ageᵢ / half_life) · 1/n_symbols(i)`
   - `polarity_slope = mean(polarity | age ≤ 30d) − mean(polarity | age ≤ 90d)`
   - `article_count_log = log1p(N)`

Half-life is **30 days** by default, so a 30-day-old article gets ~half the
weight of today's. Inverse `n_symbols` weighting penalizes broad
"market commentary" stories tagged with dozens of tickers vs. ticker-specific
news.

We **never re-score the text** with VADER/FinBERT — the vendor already did
that, and re-running it would only add noise.

Tickers with fewer than `min_articles = 3` matched stories get a NaN
`polarity_decay` and end up with a sentiment score of 0 (neutral) after
sector z-scoring fills in the global fallback.

Cross-sectional combination:

```
sentiment_score = 0.65 · z(polarity_decay) + 0.25 · z(polarity_slope) + 0.10 · z(article_count_log)
```

### 4.4 Macro pillar — regime classifier + sector tilt

Implemented in [scores/regime.py](project_alpha/scores/regime.py) (regime
classifier) and [scores/composite.py:macro_sector_tilt](project_alpha/scores/composite.py)
(per-sector tilt mapping).

The macro pillar is **not a per-stock score** — it is a per-sector tilt that
captures the prevailing macro regime. We classify the regime at each
rebalance using a small basket of FRED indicators:

```
regime_score = 0
+ 1.0    if VIX percentile ≤ 0.20            (calm tape)
- 1.0    if VIX percentile ≥ 0.80            (panic)
- 0.5    if T10Y2Y < 0                       (inverted curve)
+ 0.5    if T10Y2Y > 0.5                     (steep curve)
+ 0.5    if T10Y2Y diff(63d) > 0.25          (steepening)
- 0.5    if T10Y2Y diff(63d) < -0.25         (flattening)
+ 0.5    if FEDFUNDS Δ3M < -0.25             (cutting cycle)
- 0.5    if FEDFUNDS Δ3M > 0.25              (hiking cycle)
+ 0.25   if UNRATE Δ3M < -0.10               (improving labor market)
- 0.5    if UNRATE Δ3M > 0.20                (deteriorating labor market)

regime = "risk_off"  if score ≤ -1.5
       = "risk_on"   if score ≥ +1.5
       = "neutral"   otherwise
```

Once a regime is classified, two things happen:

**1. Pillar weight tilt** — the four-pillar weights are tilted from the
neutral base config:

```yaml
base_weights:    { technical: 0.35, fundamental: 0.30, sentiment: 0.20, macro: 0.15 }
risk_off_tilt:   { technical: -0.10, fundamental: +0.05, sentiment: -0.05, macro: +0.10 }
risk_on_tilt:    { technical: +0.10, fundamental: -0.05, sentiment: +0.05, macro: -0.10 }
```

After applying the tilt, weights are re-normalized to sum to 1.0
([scores/regime.py:pillar_weights_for_regime](project_alpha/scores/regime.py)).

**2. Per-sector tilt** — `macro_sector_tilt` returns a per-sector value
that becomes each ticker's `macro_score`:

| Regime | Cyclicals (IT, Financials, Industrials, Discretionary, Materials, Energy, Comm Svcs) | Defensives (Staples, Utilities, Health Care, Real Estate) |
|---|---|---|
| `risk_on`  | +1 (z-scaled) | −1 (z-scaled) |
| `risk_off` | −1 (z-scaled) | +1 (z-scaled) |
| `neutral`  | 0 | 0 |

### 4.5 Composite combination

Implemented in [scores/composite.py:combine_composite](project_alpha/scores/composite.py).

```
composite_score(t) = w_T · technical_score(t)
                   + w_F · fundamental_score(t)
                   + w_S · sentiment_score(t)
                   + w_M · macro_score(sector(t))
```

with `(w_T, w_F, w_S, w_M)` from the regime tilt above. The composite is
then **sector-neutralized again** (winsorized z-score within sector) so
that a top-decile name is always strong relative to its peers, not because
its sector got a free macro bump.

---

## 5. Cadence — three rebalance frequencies

Implemented in [backtest/schedule.py:build_schedule](project_alpha/backtest/schedule.py).

Every event lands on the **last trading day of the month**. Each event
carries a flag-set telling the orchestrator what to refresh:

| Event flag | When it fires | What gets refreshed |
|---|---|---|
| `reconstitute`        | `month ∈ {6, 12}`         | Universe pulled from R1000 snapshot |
| `refresh_fundamental` | `month ∈ {2, 5, 8, 11}` ∪ recon months | Fundamental panel rebuilt |
| `refresh_technical`   | every event              | Technical panel rebuilt |
| `refresh_sentiment`   | every event              | Sentiment panel rebuilt |
| `refresh_macro`       | every event              | Regime classified, pillar weights computed |

Why this layered cadence:

- **Universe** rarely changes — semi-annual matches Russell's reconstitution
  schedule and avoids constant churn.
- **Fundamentals** are filing-driven; recomputing more often than 10-Q
  cadence is wasted compute.
- **Technicals & sentiment** decay fast and are cheap to refresh — monthly.
- **Macro** can shift fast (e.g. an FOMC surprise); monthly is the
  practical cap given the data we have.

---

## 6. Portfolio construction — picking the constituents

Implemented in [portfolio/construct.py:optimize_sector_matched](project_alpha/portfolio/construct.py).

The orchestrator hands the optimizer:

- `scores` — the composite z-score per ticker
- `sector` — sector tag per ticker
- `sector_targets` — Russell 1000 sector weights (sum to 1.0)
- `prev_w` — last rebalance's portfolio weights (or None)
- `stock_rets`, `bench_rets` — 252-day return histories for risk model
- `price`, `adv20` — last close and 60d dollar ADV
- `max_name`, `adv_consume_fraction`, `beta_target`, `beta_tol`, `lambda_risk`, `lambda_turnover`

The optimizer is a long-only convex program (cvxpy + ECOS, with SCS fallback):

```
maximize    score · w  −  λ_risk · ||diag(σ_252) · w||₂  −  λ_turnover · ||w − w_prev||₁

subject to  w ≥ 0
            Σ w = 1
            (sector mask · w) = sector_target,                  for every sector
            β_target − β_tol  ≤  β·w  ≤  β_target + β_tol
            w ≤ cap                                             per name
```

Where:

- `σ_252` is the annualized stdev of daily returns over the trailing 252 days,
  smoothed 50/50 with the sector mean.
- `β` is each name's CAPM beta vs. IWB over the same window.
- `cap` per name is `min(adv_consume_fraction · ADV_dollars / NAV, max_name)`.
  The `/ NAV` conversion turns a dollar liquidity cap into a portfolio
  weight cap so it can be compared with `max_name` consistently.
- The sector match is an **exact equality** so the portfolio's GICS sector
  mix mirrors the Russell 1000.
- The L1 turnover penalty makes the optimizer prefer keeping names with
  small score drift rather than churning every month.

### No-trade band

After the optimizer returns, we apply a **no-trade band** on top of names
that survive into the new portfolio:

```python
common = new_w.index ∩ prev_w.index
if |new_w[i] − prev_w[i]| < 0.5%:   # turnover_no_trade_band
    new_w[i] := prev_w[i]
renormalize(new_w)
```

This prevents the optimizer from generating tiny, costly trades when
scores wiggle around the same level. Names that **left the universe at
reconstitution** are never frozen — they must be sold.
([backtest/run.py:380](project_alpha/backtest/run.py))

### Fallback allocator

If cvxpy is unavailable or the SOCP fails to converge, the orchestrator
falls back to a proportional-by-sector allocator that distributes each
sector's target weight across its members in proportion to their
clipped-positive scores. Beta and ADV constraints are not strictly enforced
in fallback mode — this is a degraded path, not the production path.

---

## 7. Backtest mechanics

Implemented in [backtest/run.py:run](project_alpha/backtest/run.py).

For each event in the schedule:

1. **Refresh universe** if `reconstitute` flag is set; otherwise reuse.
2. **Warm price cache** for any new tickers (lazy load of `*.eod.parquet`).
3. **Refresh pillar panels** (fundamental quarterly, others monthly).
4. **Classify regime** and compute pillar weights.
5. **Combine into composite** (sector-neutralized).
6. **Run optimizer** to produce target weights.
7. **Apply no-trade band**.
8. **Apply transaction cost** as a one-day haircut at the start of the
   period: `cost = turnover × bps_per_side / 10_000`. Default `bps = 8`.
9. **Compute daily returns** from the next trading day to the day before
   the next event:
   ```
   r_t = Σ_i w_i · r_{i,t}
   ```
   These returns roll forward without re-balancing intra-period.
10. **Compound NAV** and log everything.

After all events, daily returns are concatenated into a single time series
and KPIs are computed against `IWB`:

| KPI | Formula |
|---|---|
| CAGR | `(1+r).prod()^(252/N) − 1` |
| Ann vol | `r.std(ddof=0) · √252` |
| Sharpe | `r.mean() / r.std() · √252` |
| Max DD | `min((1+r).cumprod() / cummax − 1)` |
| Tracking error | `(r − r_bench).std() · √252` |
| Information Ratio | `(r − r_bench).mean() / TE · √252` |
| Active return | `Alpha CAGR − Benchmark CAGR` |

---

## 8. Outputs

After a backtest run, the configured `reports_dir` will contain:

| File | Contents |
|---|---|
| `alpha_daily_returns.csv` | Daily simple returns of the strategy |
| `alpha_nav.csv`           | Cumulative NAV of the strategy (starts at `initial_nav`) |
| `benchmark_nav.csv`       | Same for IWB |
| `rebalance_nav_snapshots.csv` | NAV, regime, holding count at every event |
| `regime_log.csv`          | Regime label + classifier signals at every event |
| `constituents.parquet`    | Per-event holdings with weights and pillar scores |
| `scores.parquet`          | Per-event full score panel (every ticker, every pillar) |
| `kpis.json`               | Final summary KPIs |

---

## 9. Configuration

Single source of truth: [config/alpha.yaml](config/alpha.yaml). The loader
([project_alpha/config.py](project_alpha/config.py)) validates against a
JSON-schema and a small set of cross-field invariants — most importantly
that `scoring.composite.base_weights` sums to 1.0 and that
`backtest.start < backtest.end`. A bad config raises at startup; it does
**not** silently fall back to in-code defaults.

Key parameters and where they bite:

| Setting | Default | Effect |
|---|---|---|
| `universe.reconstitution_anchor` | `06-30` | The R1000 snapshot used for both June and December rebalances |
| `cadence.quarterly_fundamental_months` | `[2, 5, 8, 11]` | When 10-Qs typically settle |
| `scoring.composite.base_weights` | tech 0.35, fund 0.30, sent 0.20, macro 0.15 | Neutral-regime mix |
| `portfolio.max_name_weight` | 0.05 | Hard cap on any single position |
| `portfolio.adv_consume_fraction` | 0.10 | We won't take more than 10% of 60d ADV in a name |
| `portfolio.beta_target` / `beta_tol` | 1.00 / 0.05 | Beta is constrained to the band |
| `portfolio.lambda_turnover` | 4.0 | L1 penalty on weight changes vs. previous portfolio |
| `portfolio.trade_cost_bps` | 8 | Per-side trading cost subtracted from NAV at each rebalance |
| `portfolio.turnover_no_trade_band` | 0.005 | Freeze any name whose target weight changes by less than 50 bps |

---

## 10. Known limitations (Phase 1+2)

These are real, deliberate, and on the Phase 3 punch list:

1. **No SEC compliance enforcement** — 1940 Act 75-5-10 diversification,
   RIC quarterly tests, and Rule 22e-4 liquidity buckets are not yet
   wired in as hard constraints.
2. **No tax-lot accounting** — realized gains/losses, wash-sale detection,
   and HIFO lot selection are absent. The current backtest is pre-tax.
3. **No dividend reinvestment in NAV** — daily returns use
   `adjusted_close.pct_change()`, which already incorporates dividend
   reinvestment for *single-name* return calculation. A separate cash
   dividend leg is not maintained.
4. **No expense-ratio drag** — we don't apply a daily TER haircut to NAV.
5. **Universe limited to 2024-06-28 snapshot for any rebalance after that
   date** (the data only contains 10 annual snapshots).
6. **MSFT** has no `.eod.parquet` on disk — it is silently dropped. The
   data layer already logs missing tickers; fix the fundamental data
   delivery to close this gap.
7. **AnalystRatings** is a single current snapshot in the JSON, so we do
   not use it in the backtest (would inject look-ahead). It is available
   for the live-scoring path.

---

## 11. How to run

```bash
cd project_alpha_code_v2_1

# Show the effective, validated config
python -m project_alpha.cli print-config

# Run the full backtest (window + paths come from alpha.yaml)
python -m project_alpha.cli backtest

# Override the config file location
PROJECT_ALPHA_CONFIG=/path/to/alpha.yaml python -m project_alpha.cli backtest

# Test suite — 35 tests, ~20 s
python -m pytest tests/ -q
```

A full run over the 2015-07 → 2025-04 window produces:

- ~118 monthly rebalance events
- ~10 universe reconstitutions (one per Jun and Dec)
- ~40 fundamental refreshes (Feb / May / Aug / Nov + the recon months)
- A daily NAV series of ~2,470 trading days

---

## 12. Quick mental model

```
                ┌─────────────────────────────────────────┐
                │   alpha.yaml  (single config, validated)│
                └───────────────┬─────────────────────────┘
                                │
                        ┌───────▼────────┐
                        │  Universe      │  semi-annual recon from R1000
                        │  Sectors       │  override via R1000 GICS
                        └────────────────┘
                                │
            ┌───────────────────┼────────────────┬────────────┐
            ▼                   ▼                ▼            ▼
     ┌────────────┐   ┌──────────────┐   ┌────────────┐  ┌────────────┐
     │ Technical  │   │ Fundamental  │   │ Sentiment  │  │  Macro     │
     │ 12-1 mom,  │   │ Quality/     │   │ pre-scored │  │  regime    │
     │ vol, 52w,  │   │ Value/Growth │   │ polarity   │  │  classifier│
     │ liquidity  │   │ /Leverage    │   │ + decay    │  │            │
     └─────┬──────┘   └──────┬───────┘   └─────┬──────┘  └──────┬─────┘
           └──────────────┬──┴──────────────┬──┘                │
                          │  per-ticker z   │                   │
                          ▼                 ▼                   ▼
                    ┌────────────────────────────┐    ┌────────────────┐
                    │ composite_score (sector-z) │◀───│ pillar weights │
                    │   = wT·T + wF·F + wS·S +   │    │  (regime tilt) │
                    │     wM·sector_tilt(macro)  │    └────────────────┘
                    └──────────────┬─────────────┘
                                   ▼
                    ┌────────────────────────────────┐
                    │ Sector-matched, β-banded,      │
                    │ ADV-capped optimizer (cvxpy)   │
                    └──────────────┬─────────────────┘
                                   ▼
                          ┌────────────────────┐
                          │  Holdings + weights │
                          └────────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │  Daily NAV vs IWB  │
                          └────────────────────┘
```
