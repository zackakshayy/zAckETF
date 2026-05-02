# Project Alpha — RUNBOOK

How to configure and run the backtest engine on a local Mac/Linux box from VS
Code or the terminal. For the full strategy explanation, see
[STRATEGY.md](STRATEGY.md).

---

## 1. One-time setup

### 1.1 Prerequisites

- **Python 3.10+** (3.11 recommended)
- **VS Code** with the Python extension installed
- ~5 GB free disk space for the off-repo data directory

### 1.2 Clone & install

```bash
# From the directory where you want the repo
cd /path/to/your/projects
# (your repo is already cloned at AlphaEngine_v60/project_alpha_code_v2_1)

cd AlphaEngine_v60/project_alpha_code_v2_1

# Create a venv and install dependencies
python3 -m venv ../zack_ai
source ../zack_ai/bin/activate
pip install --upgrade pip
pip install -r requirements.txt   # if a requirements file exists

# If no requirements.txt, install the runtime deps directly:
pip install pandas numpy pyyaml jsonschema tqdm cvxpy ecos scs pyarrow
```

### 1.3 Data layout

The engine expects the **off-repo data root** to be laid out exactly as
`alpha.yaml` declares. Substitute your absolute path for `<DATA_ROOT>`
(e.g. `~/Desktop/ProjectX_MasterData`):

```
ProjectX_MasterData/
├── R1000.xlsx                          # Russell 1000 universe (sheet "R1K")
├── categorized/
│   ├── prices/{TICKER}.US.eod.parquet  # daily prices/volume
│   ├── fundamentals/{TICKER}.US.fundamentals.json
│   ├── news/{TICKER}.US_YYYY-MM.news.json
│   └── macro/{SERIES_ID}.csv           # FRED CSVs (VIXCLS, T10Y2Y, …)
├── _px_reports/                        # backtest output (auto-created)
└── _px_cache/                          # optional cache (auto-created)
```

If your data lives elsewhere, edit the `paths:` block in
[`config/alpha.yaml`](config/alpha.yaml) — every path is configurable and
nothing is hard-coded in code.

---

## 2. Configuration (the only file that matters)

Single source of truth: [`config/alpha.yaml`](config/alpha.yaml). The loader
([`project_alpha/config.py`](project_alpha/config.py)) validates the file
against a JSON-schema and raises if anything is malformed. **There are no
in-code default fallbacks.**

### 2.1 The blocks you'll touch most often

| Block | Purpose | Common edits |
|---|---|---|
| `paths` | Filesystem locations | Point to your local data root |
| `backtest.start` / `.end` | Window | Change to e.g. `"2020-05-01"` → `"2025-04-30"` for a 5-year run |
| `backtest.benchmark_ticker` | Benchmark | Default `IWB` (Russell 1000 ETF). Change if you want a different index |
| `scoring.composite.base_weights` | Pillar weights at neutral regime | Must sum to 1.0 (validator enforces) |
| `scoring.composite.dynamic_pillar_weights` | Turn IC-blender on/off | `true` (default) uses trailing-IC weights; `false` uses static `base_weights` |
| `scoring.composite.quality_momentum_gate_w` | Quality × Momentum interaction strength | `0.0` disables, `0.20` is current default |
| `portfolio.core_iwb_weight` | Core-satellite IWB anchor weight | `0.60` default; set `0.0` for pure tilt |
| `portfolio.top_quintile_filter` | Concentration filter | `true` keeps top X% per sector; set `false` to use full universe (much slower) |
| `portfolio.beta_tol` | Beta band half-width | Tighter = closer to IWB beta |
| `portfolio.score_no_trade_band` | Freeze names whose composite score didn't move | Default `0.25` (¼-σ); `0.0` disables |
| `portfolio.drawdown_overlay` | De-risk on stress | `false` recommended (was firing 71% of the time in iter-1) |

### 2.2 Switching between strategy variants

Three knobs cover the main "modes":

```yaml
# Pure smart-beta sleeve (no IWB anchor, expect higher TE)
portfolio:
  core_iwb_weight: 0.0
  top_quintile_filter: true
  quintile_keep: 0.20

# Core-satellite (recommended) — bounded TE, modest active risk
portfolio:
  core_iwb_weight: 0.60
  top_quintile_filter: true
  quintile_keep: 0.20

# Closet indexer — maximal IWB tracking, tilt is cosmetic
portfolio:
  core_iwb_weight: 0.85
  top_quintile_filter: true
  quintile_keep: 0.30
  sector_active_band: 0.0    # exact sector match
  beta_tol: 0.05
```

### 2.3 Validation

After editing `alpha.yaml`, **always run the print-config command first** —
it loads the YAML, validates against the schema, and prints the effective
config. If anything is wrong, you'll see the JSON-schema error before you
waste a backtest run:

```bash
python -m project_alpha.cli print-config | head -50
```

---

## 3. Run from VS Code

### 3.1 Open the workspace

```bash
code AlphaEngine_v60/project_alpha_code_v2_1
```

### 3.2 Select the right interpreter

`Cmd+Shift+P` → "Python: Select Interpreter" → pick the venv at
`AlphaEngine_v60/zack_ai/bin/python`. VS Code persists this choice in
`.vscode/settings.json`.

### 3.3 Recommended `.vscode/launch.json`

Save this as `.vscode/launch.json` inside the repo root for one-click runs:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run backtest",
      "type": "debugpy",
      "request": "launch",
      "module": "project_alpha.cli",
      "args": ["backtest"],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Print effective config",
      "type": "debugpy",
      "request": "launch",
      "module": "project_alpha.cli",
      "args": ["print-config"],
      "console": "integratedTerminal"
    },
    {
      "name": "Pytest (full suite)",
      "type": "debugpy",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-q"],
      "console": "integratedTerminal"
    }
  ]
}
```

Then `F5` runs the backtest. Use the **Run and Debug** panel
(`Cmd+Shift+D`) to switch between configurations.

### 3.4 Or run from the integrated terminal

```bash
# Activate the venv (only needed once per terminal)
source ../zack_ai/bin/activate

# Validate config
python -m project_alpha.cli print-config

# Run the full backtest (window comes from alpha.yaml)
python -m project_alpha.cli backtest

# Override the config file location for a one-off run
PROJECT_ALPHA_CONFIG=/path/to/custom_alpha.yaml python -m project_alpha.cli backtest

# Run the test suite (~25 seconds, 67 tests)
python -m pytest tests/ -q
```

### 3.5 Where the output lands

Everything goes to `paths.reports_dir` (e.g. `<DATA_ROOT>/_px_reports`):

| File | Contents |
|---|---|
| `kpis.json` | CAGR, Sharpe, MaxDD, TE, IR, active return |
| `alpha_daily_returns.csv` | Daily strategy returns |
| `alpha_nav.csv` / `benchmark_nav.csv` | NAV curves |
| `rebalance_nav_snapshots.csv` | NAV + regime + holdings count + gross at each rebalance |
| `regime_log.csv` | Regime label + signal breakdown each rebalance |
| `pillar_weights_history.csv` | Pillar weights chosen each rebalance (after IC blending) |
| `pillar_ic_history.csv` | Trailing IC of each pillar (when dynamic blending is on) |
| `constituents.parquet` | Per-event holdings × scores |
| `scores.parquet` | Per-event full score panel |

A quick KPI comparison vs IWB:

```bash
python -c "
import json, pandas as pd, numpy as np
k = json.load(open('<DATA_ROOT>/_px_reports/kpis.json'))
print(f\"CAGR    Alpha {k['cagr']:7.2%}   IWB {k['benchmark_cagr']:7.2%}\")
print(f\"Sharpe          {k['sharpe']:6.3f}\")
print(f\"MaxDD   Alpha {k['max_drawdown']:7.2%}\")
print(f\"TE / IR        {k['tracking_error']:7.2%} / {k['information_ratio']:6.3f}\")
print(f\"Active return  {k['active_return']:+7.2%}\")
"
```

---

## 4. Common workflows

### 4.1 "Try a different backtest window"

Edit `backtest.start` / `backtest.end` in `alpha.yaml`. Re-run. No code
changes needed.

### 4.2 "Disable a pillar"

Set its weight in `scoring.composite.base_weights` to 0 and redistribute
across the remaining pillars (must still sum to 1.0). With
`dynamic_pillar_weights: true`, the IC blender will down-weight any pillar
with poor trailing IC anyway — but the floor (`ic_floor: 0.05`) keeps it
alive at a small weight.

### 4.3 "Run a faster backtest (sanity-check a config change)"

```yaml
backtest:
  start: "2024-01-01"
  end:   "2024-12-31"

portfolio:
  top_quintile_filter: true
  quintile_keep: 0.10        # tighter → fewer optimizer variables → faster
  min_names_per_sector: 3
```

A 1-year window with tight concentration runs in ~3 minutes. Good for
config-change validation before the full 5-year run.

### 4.4 "Add a new pillar"

1. Create a module in `project_alpha/scores/<newpillar>.py` mirroring
   `lowvol.py`'s structure: `compute_<x>_features` (per-ticker dict) and
   `compute_<x>_panel` (cross-sectional DataFrame with `<newpillar>_score`).
2. Add a `_refresh_<x>_panel` helper in `backtest/run.py`.
3. Wire into `combine_composite` (`composite.py`) and add the pillar name to
   `dynamic_weights.PILLARS` so the IC blender sees it.
4. Add an entry in `scoring.composite.base_weights` (re-normalize others).
5. Add a test file `tests/test_<newpillar>.py`.

The codebase is structured so that adding a pillar touches ~5 files and
takes <100 lines.

---

## 5. Performance & memory expectations

On a 2024-era M1/M2 Mac, a full 5-year run:

| Phase | Time | Memory |
|---|---|---|
| Config load + universe load | ~1 sec | ~50 MB |
| EOD cache warmup (~1500 tickers) | ~30 sec | ~500 MB peak |
| Per rebalance (steady state) | 25–40 sec | +50 MB transient |
| Total 60-event run | 25–40 min | ~1 GB peak |

The optimizer (cvxpy + ECOS) dominates per-event cost. Tightening
`quintile_keep` from 0.30 to 0.20 cuts solver time roughly in half because
the SOCP problem size shrinks.

Memory tip: the `eod_cache` evicts tickers when they leave the universe at
reconstitution, so it doesn't grow unboundedly across long backtests.

---

## 6. Debugging tips

| Symptom | Likely cause | Fix |
|---|---|---|
| `alpha.yaml failed schema validation` | Typo / wrong type in YAML | The error names the field — check indentation and value types |
| `composite base_weights must sum to 1.0` | Edited weights without re-normalizing | Sum your weights in a calculator first |
| Backtest stops after 0 events | Benchmark EOD missing | Verify `IWB.US.eod.parquet` exists in `prices_dir` |
| Tilt CAGR very low (single-digit) | Pillar ICs are near-zero this period | Inspect `pillar_ic_history.csv` — if all IC ~0, signals are not predictive |
| Optimizer falling back to proportional | cvxpy unavailable or solver failing | Check `pip install cvxpy ecos scs` succeeded |
| MSFT or other mega-cap missing | No EOD parquet for that ticker | Known data gap; engine logs and continues |

Set `logging.level: DEBUG` in `alpha.yaml` for verbose output during a run.
Logs go to stdout (visible in the VS Code terminal).

---

## 7. Project layout (orientation)

```
project_alpha_code_v2_1/
├── config/
│   └── alpha.yaml                  # the single config file
├── project_alpha/
│   ├── cli.py                      # `python -m project_alpha.cli ...`
│   ├── config.py                   # YAML loader + JSON-schema validator
│   ├── data/                       # universe, prices, fundamentals, news, macro loaders
│   ├── scores/
│   │   ├── technical.py            # momentum + idio momentum + vol + …
│   │   ├── fundamental.py          # quality / value / growth / leverage
│   │   ├── sentiment.py            # news polarity decay
│   │   ├── lowvol.py               # vol + beta + vol-of-vol pillar
│   │   ├── regime.py               # macro regime classifier (3-state)
│   │   ├── composite.py            # pillar combiner + Q×M gate
│   │   └── dynamic_weights.py      # IC-weighted blender (PillarICTracker)
│   ├── portfolio/
│   │   ├── construct.py            # cvxpy SOCP optimizer + sector / beta / ADV constraints
│   │   └── nav.py                  # daily P&L stitching helpers
│   └── backtest/
│       ├── schedule.py             # rebalance event generator
│       └── run.py                  # main orchestrator
├── tests/                          # pytest suite (67 tests)
├── STRATEGY.md                     # strategy in detail
└── RUNBOOK.md                      # this file
```

---

## 8. Production checklist

Before treating any output as live-grade:

- [ ] `python -m pytest tests/ -q` shows all green
- [ ] `python -m project_alpha.cli print-config` round-trips your edits
- [ ] `kpis.json` shows non-zero `n_trading_days` (matches expected window)
- [ ] `pillar_ic_history.csv` shows finite ICs (not all NaN)
- [ ] `rebalance_nav_snapshots.csv` shows sensible holdings counts (>20)
- [ ] Manual spot-check: a few rebalance events have plausible top-10 holdings

If any of these fail, **don't ship the run.** Re-validate config and re-run.
