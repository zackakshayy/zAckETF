"""
Backtest orchestrator (Phase 1 + Phase 2).

End-to-end pipeline:
  1. Load data (universe, prices, IWB benchmark, macro panel).
  2. Generate rebalance schedule (semi-annual recon / quarterly fundamental
     / monthly technical+macro).
  3. At each event, refresh score panels with a `last_*` cache so we only
     do quarterly fundamental work on quarterly events, etc.
  4. Combine pillars into composite (regime-tilted weights).
  5. Optimize portfolio (sector-matched, beta-banded, ADV-capped).
  6. Stitch daily returns and write reports.

This module replaces the previous run.py that wired only technical features.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from project_alpha.config import get_cfg
from project_alpha.data import (
    load_russell_panel, universe_for_date, sector_targets_for_date,
    load_eod, build_returns_panel, adv20_panel, last_price_series,
    point_in_time_features, point_in_time_earnings,
    list_monthly_files, load_news_window, aggregate_sentiment_window,
    load_macro_panel,
)
from project_alpha.scores.fundamental import compute_fundamental_panel
from project_alpha.scores.technical import compute_technical_features, compute_technical_panel
from project_alpha.scores.lowvol import compute_lowvol_features, compute_lowvol_panel
from project_alpha.scores.sentiment import compute_sentiment_panel
from project_alpha.scores.regime import classify_regime, pillar_weights_for_regime
from project_alpha.scores.composite import combine_composite
from project_alpha.scores.dynamic_weights import PillarICTracker
from project_alpha.backtest.schedule import build_schedule, ScheduleEvent
from project_alpha.portfolio.construct import optimize_sector_matched, filter_top_quintile_by_sector
from project_alpha.portfolio.nav import stitch_pnl

LOG = logging.getLogger("project_alpha.backtest")


# ---------------------------------------------------------------------------
# Pillar refresh helpers
# ---------------------------------------------------------------------------

def _refresh_technical_panel(
    eod_cache: Dict[str, pd.DataFrame],
    tickers: List[str],
    sector_map: Dict[str, str],
    asof: pd.Timestamp,
    cfg_tech: Dict[str, Any],
    bench_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        eod = eod_cache.get(t)
        if eod is None or eod.empty:
            continue
        feats = compute_technical_features(
            eod, asof,
            momentum_12_1_days=cfg_tech.get("momentum_12_1_days", 252),
            momentum_12_1_skip_days=cfg_tech.get("momentum_12_1_skip_days", 21),
            momentum_6_1_days=cfg_tech.get("momentum_6_1_days", 126),
            vol_lookback_days=cfg_tech.get("vol_lookback_days", 252),
            high52_lookback_days=cfg_tech.get("high52_lookback_days", 252),
            short_term_reversal_days=cfg_tech.get("short_term_reversal_days", 21),
            bench_returns=bench_returns,
        )
        if not np.isfinite(feats.get("price", float("nan"))):
            continue
        feats["ticker"] = t
        rows.append(feats)
    return compute_technical_panel(rows, sector_map=sector_map)


def _refresh_lowvol_panel(
    eod_cache: Dict[str, pd.DataFrame],
    tickers: List[str],
    sector_map: Dict[str, str],
    asof: pd.Timestamp,
    bench_returns: Optional[pd.Series] = None,
    vol_lookback_days: int = 252,
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        eod = eod_cache.get(t)
        if eod is None or eod.empty:
            continue
        feats = compute_lowvol_features(eod, asof, bench_returns=bench_returns,
                                        vol_lookback_days=vol_lookback_days)
        if not np.isfinite(feats.get("vol_252d", float("nan"))):
            continue
        feats["ticker"] = t
        rows.append(feats)
    return compute_lowvol_panel(rows, sector_map=sector_map)


def _refresh_fundamental_panel(
    fund_dir: str,
    tickers: List[str],
    asof: pd.Timestamp,
    last_price_map: Dict[str, float],
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """Build the fundamental panel.

    The fundamentals JSON's `General.GicSector` is missing for many issuers.
    We override every row's sector with the R1000 sector_map (the canonical
    universe-level GICS sector) before sector z-scoring.
    """
    rows = []
    for t in tickers:
        feats = point_in_time_features(fund_dir, t, asof, last_price=last_price_map.get(t))
        if not feats:
            continue
        # Iter-13: merge point-in-time earnings-surprise (PEAD) features so the
        # rebuilt fundamental pillar can score the catalyst sub-pillar.
        earn = point_in_time_earnings(fund_dir, t, asof)
        for k in ("surprise_last", "surprise_avg", "surprise_streak", "eps_ttm_growth"):
            feats[k] = earn.get(k)
        feats["sector"] = sector_map.get(t) or feats.get("sector") or "Unknown"
        rows.append(feats)
    return compute_fundamental_panel(rows)


def _refresh_sentiment_panel(
    news_dir: str,
    tickers: List[str],
    sector_map: Dict[str, str],
    asof: pd.Timestamp,
    cfg_sent: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for t in tickers:
        df = load_news_window(news_dir, t, asof, window_days=cfg_sent.get("window_days", 90))
        agg = aggregate_sentiment_window(
            df, asof,
            half_life_days=cfg_sent.get("half_life_days", 30),
            relevance_weight=cfg_sent.get("relevance_weight", "inverse_symbol_count"),
            require_ticker_in_symbols=True,
            min_articles=cfg_sent.get("min_articles", 3),
        )
        agg["ticker"] = t
        rows.append(agg)
    return compute_sentiment_panel(rows, sector_map=sector_map)


# ---------------------------------------------------------------------------
# Optimizer-input prep
# ---------------------------------------------------------------------------

def _prepare_optimizer_inputs(
    universe: pd.DataFrame,
    composite: pd.DataFrame,
    asof: pd.Timestamp,
    eod_cache: Dict[str, pd.DataFrame],
    bench_returns: pd.Series,
    *,
    adv_lookback_days: int,
    history_lookback_days: int = 252,
):
    """Build aligned pandas inputs for the cvxpy optimizer."""
    tickers = sorted(set(universe["ticker"]) & set(composite.index))
    if not tickers:
        return None

    # Stock returns history (daily simple) up to asof
    returns: Dict[str, pd.Series] = {}
    last_px: Dict[str, float] = {}
    last_advd: Dict[str, float] = {}
    for t in tickers:
        eod = eod_cache.get(t)
        if eod is None or eod.empty:
            continue
        s = eod[eod["date"] <= asof].set_index("date")
        if "adjusted_close" not in s.columns:
            continue
        r = s["adjusted_close"].astype(float).pct_change().dropna()
        if len(r) < 60:
            continue
        returns[t] = r.tail(history_lookback_days)
        last_px[t] = float(s["adjusted_close"].iloc[-1])
        if "volume" in s.columns and "close" in s.columns:
            dv = (s["close"].astype(float) * s["volume"].astype(float)).rolling(adv_lookback_days, min_periods=10).mean()
            last_advd[t] = float(dv.iloc[-1]) if len(dv) > 0 else float("nan")

    if not returns:
        return None

    # Use a dict→DataFrame so column names are the ticker keys (concat of
    # unnamed Series produces integer columns and trips reindex).
    R = pd.DataFrame(returns).sort_index()
    bench = bench_returns.reindex(R.index).ffill().fillna(0.0)
    R = R.fillna(0.0)

    score = composite.loc[[t for t in R.columns if t in composite.index], "composite_score"].astype(float)
    sector = composite.loc[score.index, "sector"].astype(str)
    price = pd.Series({t: last_px.get(t, float("nan")) for t in score.index})
    advd = pd.Series({t: last_advd.get(t, float("nan")) for t in score.index})

    sector_targets = sector_targets_for_date(universe.set_index("ticker").loc[
        [t for t in score.index if t in set(universe["ticker"])]
    ].reset_index())

    return dict(
        scores=score,
        sector=sector,
        sector_targets=sector_targets,
        stock_rets=R[score.index],
        bench_rets=bench,
        price=price,
        adv20=advd,
    )


def _liquidity_cap_to_weight(adv_dollars: pd.Series, *, nav: float, fraction: float) -> pd.Series:
    """Convert dollar liquidity caps to portfolio-weight caps."""
    if nav <= 0:
        return pd.Series(np.nan, index=adv_dollars.index)
    cap = (adv_dollars * float(fraction)) / float(nav)
    return cap.replace([np.inf, -np.inf], np.nan).clip(lower=0.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> Dict[str, Any]:
    cfg = get_cfg()
    paths = cfg["paths"]
    out_dir = Path(paths["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info(f"Loaded config from {cfg.get('_loaded_from')}")
    LOG.info(f"Backtest window: {cfg['backtest']['start']} → {cfg['backtest']['end']}")

    # ---- Universe + benchmark + macro ----
    russell = load_russell_panel(paths["russell_xlsx"])
    LOG.info(f"R1000 panel: {russell['snapshot_date'].nunique()} snapshots, "
             f"{russell['ticker'].nunique()} tickers")

    bench_eod = load_eod(paths["prices_dir"], cfg["backtest"]["benchmark_ticker"])
    if bench_eod.empty:
        raise RuntimeError(f"Benchmark EOD missing for {cfg['backtest']['benchmark_ticker']}")
    bench_returns = bench_eod.set_index("date")["adjusted_close"].astype(float).pct_change().dropna()

    macro_panel = load_macro_panel(paths["macro_dir"])
    LOG.info(f"Macro panel columns: {list(macro_panel.columns)}")

    # Trading calendar from benchmark
    start = pd.Timestamp(cfg["backtest"]["start"])
    end = pd.Timestamp(cfg["backtest"]["end"])
    trading_dates = bench_returns.index[(bench_returns.index >= start) & (bench_returns.index <= end)]

    # ---- Schedule ----
    cad = cfg["cadence"]
    # rebalance_months: None → monthly; [6,12] → semi-annual; [3,6,9,12] → quarterly.
    # When monthly_rebalance is true the filter is dropped (monthly cadence).
    rebal_months = None if cad.get("monthly_rebalance", True) else cad.get("rebalance_months")
    schedule = build_schedule(
        trading_dates,
        start=start, end=end,
        quarterly_fundamental_months=cad["quarterly_fundamental_months"],
        semiannual_reconstitution_months=cad["semiannual_reconstitution_months"],
        rebalance_months=rebal_months,
    )
    LOG.info(f"Generated {len(schedule)} rebalance events "
             f"(cadence: {'monthly' if rebal_months is None else sorted(rebal_months)}).")

    # ---- Caches (refreshed on cadence) ----
    eod_cache: Dict[str, pd.DataFrame] = {}
    universe_df: Optional[pd.DataFrame] = None
    fundamental_panel: Optional[pd.DataFrame] = None
    technical_panel: Optional[pd.DataFrame] = None
    sentiment_panel: Optional[pd.DataFrame] = None
    lowvol_panel: Optional[pd.DataFrame] = None
    sector_map: Dict[str, str] = {}

    # Indexed bench returns for IC / idio momentum / beta
    bench_idx_returns = bench_returns.copy()
    bench_idx_returns.index = pd.DatetimeIndex(bench_idx_returns.index)

    # Output stores
    daily_returns: List[pd.Series] = []
    nav_snapshots: List[Dict[str, Any]] = []
    constituent_log: List[pd.DataFrame] = []
    score_log: List[pd.DataFrame] = []
    regime_log: List[Dict[str, Any]] = []

    last_weights: Optional[pd.Series] = None
    last_scores: Optional[pd.Series] = None
    nav = float(cfg["backtest"]["initial_nav"])
    nav_high = nav

    portfolio_cfg = cfg["portfolio"]
    composite_cfg = cfg["scoring"]["composite"]

    # A1 — IC-weighted pillar blending (warmup uses regime-tilted base weights)
    use_dynamic_weights = bool(composite_cfg.get("dynamic_pillar_weights", False))
    ic_tracker = PillarICTracker(
        lookback_events=int(composite_cfg.get("ic_lookback_events", 12)),
        floor=float(composite_cfg.get("ic_floor", 0.05)),
        cap=float(composite_cfg.get("ic_cap", 0.60)),
        exclude_pillars=composite_cfg.get("exclude_pillars", []),
    ) if use_dynamic_weights else None
    pillar_weight_log: List[Dict[str, Any]] = []

    # Track which tickers are still referenced. We retain previously-held names
    # for one extra reconstitution (so the no-trade-band and prev-weight logic
    # can still see them) before evicting from the cache to bound memory.
    eod_cache_keep_extra: set = set()

    for i, ev in enumerate(tqdm(schedule, desc="Rebalances", unit="evt")):
        asof = ev.date
        # ---- Universe ----
        if ev.reconstitute or universe_df is None:
            uni = universe_for_date(
                russell, asof,
                anchor_month_day=cfg["universe"].get("reconstitution_anchor", "06-30"),
                tolerance_days=cfg["universe"].get("snapshot_tolerance_days", 14),
                exclude_asset_types=cfg["universe"].get("exclude_asset_types"),
            )
            universe_df = uni
            sector_map = dict(zip(uni["ticker"], uni["sector"]))

            # Memory: evict EOD frames for tickers not in the new universe and
            # not held in the previous portfolio. On a 1000-name universe with
            # ~20% annual churn this saves ~200 frames over a 5y backtest.
            new_set = set(uni["ticker"])
            held_prev = set(last_weights.index) if last_weights is not None else set()
            keep = new_set | held_prev | eod_cache_keep_extra
            for t in list(eod_cache.keys()):
                if t not in keep:
                    del eod_cache[t]
            eod_cache_keep_extra = held_prev  # carried forward one cycle

            # Warm up EOD cache for the new universe
            for t in uni["ticker"]:
                if t not in eod_cache:
                    df = load_eod(paths["prices_dir"], t)
                    if not df.empty:
                        eod_cache[t] = df

        if universe_df is None or universe_df.empty:
            LOG.warning(f"No universe at {asof.date()}; skipping.")
            continue

        tickers = list(universe_df["ticker"])

        # Last close as-of (for fundamentals price-derived ratios)
        last_px_map = {}
        for t in tickers:
            eod = eod_cache.get(t)
            if eod is not None and not eod.empty:
                sub = eod[eod["date"] <= asof]
                if not sub.empty:
                    last_px_map[t] = float(sub["adjusted_close"].iloc[-1])

        # ---- Pillar refresh ----
        if ev.refresh_fundamental or fundamental_panel is None:
            fundamental_panel = _refresh_fundamental_panel(
                paths["fundamentals_dir"], tickers, asof, last_px_map, sector_map,
            )

        # Bench returns up to asof (inclusive) — used by idio momentum and lowvol beta
        bench_to_asof = bench_idx_returns.loc[bench_idx_returns.index <= asof]

        if ev.refresh_technical or technical_panel is None:
            technical_panel = _refresh_technical_panel(
                eod_cache, tickers, sector_map, asof, cfg["scoring"]["technical"],
                bench_returns=bench_to_asof,
            )
            lowvol_panel = _refresh_lowvol_panel(
                eod_cache, tickers, sector_map, asof,
                bench_returns=bench_to_asof,
                vol_lookback_days=int(cfg["scoring"]["technical"].get("vol_lookback_days", 252)),
            )

        if ev.refresh_sentiment or sentiment_panel is None:
            sentiment_panel = _refresh_sentiment_panel(
                paths["news_dir"], tickers, sector_map, asof, cfg["scoring"]["sentiment"],
            )

        # ---- Macro regime → pillar weights ----
        regime, signals = classify_regime(
            macro_panel, asof,
            lookback_years=cfg["scoring"]["macro"].get("regime_lookback_years", 5),
        )
        weights = pillar_weights_for_regime(
            composite_cfg["base_weights"], regime,
            risk_off_tilt=composite_cfg.get("risk_off_tilt"),
            risk_on_tilt=composite_cfg.get("risk_on_tilt"),
        )

        # A1 — IC-weighted blending. We need the un-blended pillar panel to compute IC,
        # which we get from a first pass of `combine_composite` with whatever weights
        # are currently in hand (they don't affect the per-pillar columns we read).
        gate_w = float(composite_cfg.get("quality_momentum_gate_w", 0.0))
        composite_pre = combine_composite(
            technical_panel=technical_panel,
            fundamental_panel=fundamental_panel,
            sentiment_panel=sentiment_panel,
            pillar_weights=weights,
            regime=regime,
            sector_neutralize=composite_cfg.get("sector_neutralize", False),
            lowvol_panel=lowvol_panel,
            quality_momentum_gate_w=gate_w,
        )
        if ic_tracker is not None and not composite_pre.empty:
            prices_at_t = pd.Series(last_px_map, dtype=float).dropna()
            ic_cols = [c for c in
                       ("technical_score", "fundamental_score", "sentiment_score", "lowvol_score")
                       if c in composite_pre.columns]
            weights = ic_tracker.update_and_weights(
                asof=asof,
                score_panel=composite_pre[ic_cols],
                prices_at_t=prices_at_t,
                fixed_weights=weights,
            )

        # ---- Tier-2 item 5 — Volatility regime gating ----
        # When VIX percentile is elevated (panic / late-cycle), shift weight
        # from technical → lowvol. The shift scales with how far above the
        # threshold we are, capped at `vol_gate_shift_max`. This is a meta-
        # adjustment on top of the IC blender — captures the empirical fact
        # that momentum decays during high-vol regimes while low-vol earns
        # its premium most reliably exactly then.
        gate_threshold = float(composite_cfg.get("vol_regime_gate_threshold", 0.75))
        gate_shift_max = float(composite_cfg.get("vol_regime_gate_shift_max", 0.10))
        vix_pctile = float(signals.get("vix_pctile", float("nan"))) if signals else float("nan")
        if (gate_shift_max > 0 and np.isfinite(vix_pctile)
                and vix_pctile >= gate_threshold and 1.0 - gate_threshold > 1e-9):
            shift_strength = (vix_pctile - gate_threshold) / (1.0 - gate_threshold)
            shift_strength = float(np.clip(shift_strength, 0.0, 1.0))
            tech_w   = float(weights.get("technical", 0.0))
            # Don't drain more than half of technical's current weight.
            shift = min(gate_shift_max * shift_strength, 0.5 * tech_w)
            if shift > 0:
                weights = dict(weights)
                weights["technical"] = tech_w - shift
                weights["lowvol"]    = float(weights.get("lowvol", 0.0)) + shift
                # Re-normalize defensively (should already sum to 1).
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}

        regime_log.append({"date": asof.strftime("%Y-%m-%d"), "regime": regime,
                           **weights, **signals})

        # ---- Composite (final, with possibly IC-blended weights) ----
        composite = combine_composite(
            technical_panel=technical_panel,
            fundamental_panel=fundamental_panel,
            sentiment_panel=sentiment_panel,
            pillar_weights=weights,
            regime=regime,
            sector_neutralize=composite_cfg.get("sector_neutralize", False),
            lowvol_panel=lowvol_panel,
            quality_momentum_gate_w=gate_w,
        )
        pillar_weight_log.append({"date": asof.strftime("%Y-%m-%d"), **weights})
        if composite.empty:
            LOG.warning(f"Composite empty at {asof.date()}; skipping rebalance.")
            continue

        # ---- Optimizer ----
        opt = _prepare_optimizer_inputs(
            universe_df, composite, asof, eod_cache, bench_returns,
            adv_lookback_days=portfolio_cfg.get("adv_lookback_days", 60),
        )
        if opt is None:
            LOG.warning(f"No optimizer inputs at {asof.date()}; skipping.")
            continue

        # A3 — Top-quintile within-sector concentration filter
        if portfolio_cfg.get("top_quintile_filter", False):
            keep_idx = filter_top_quintile_by_sector(
                opt["scores"], opt["sector"],
                quintile=float(portfolio_cfg.get("quintile_keep", 0.20)),
                min_per_sector=int(portfolio_cfg.get("min_names_per_sector", 3)),
            )
            if len(keep_idx) > 0:
                opt["scores"]     = opt["scores"].reindex(keep_idx)
                opt["sector"]     = opt["sector"].reindex(keep_idx)
                opt["price"]      = opt["price"].reindex(keep_idx)
                opt["adv20"]      = opt["adv20"].reindex(keep_idx)
                opt["stock_rets"] = opt["stock_rets"][keep_idx]

        cap_w = _liquidity_cap_to_weight(
            opt["adv20"], nav=nav,
            fraction=portfolio_cfg.get("adv_consume_fraction", 0.10),
        )
        per_name_max = float(portfolio_cfg["max_name_weight"])
        # Apply liquidity floor (cap ≤ per_name_max)
        per_name_caps = pd.concat([cap_w, pd.Series(per_name_max, index=cap_w.index)], axis=1).min(axis=1)

        weights_w = optimize_sector_matched(
            scores=opt["scores"],
            sector=opt["sector"],
            sector_targets=opt["sector_targets"],
            prev_w=last_weights,
            stock_rets=opt["stock_rets"],
            bench_rets=opt["bench_rets"],
            price=opt["price"],
            adv20=per_name_caps * float(nav),  # convert weight cap back to dollars for the optimizer
            max_name=per_name_max,
            adv_frac=1.0,                       # already converted
            beta_target=float(portfolio_cfg.get("beta_target", 1.0)),
            beta_tol=float(portfolio_cfg.get("beta_tol", 0.05)),
            lambda_risk=float(portfolio_cfg.get("lambda_risk", 5.0)),
            lambda_turnover=float(portfolio_cfg.get("lambda_turnover", 4.0)),
            sector_band=float(portfolio_cfg.get("sector_active_band", 0.0)),
        )

        if weights_w is None or weights_w.empty:
            LOG.warning(f"Optimizer returned empty at {asof.date()}; reusing previous weights.")
            if last_weights is None:
                continue
            weights_w = last_weights.copy()

        # A4 — Score-aware no-trade band: freeze only names whose composite
        # score barely moved. Falls back to weight-band when score band is 0.
        score_band = float(portfolio_cfg.get("score_no_trade_band", 0.0))
        nt_band    = float(portfolio_cfg.get("turnover_no_trade_band", 0.0))
        if score_band > 0 and last_weights is not None and last_scores is not None:
            common = weights_w.index.intersection(last_weights.index).intersection(last_scores.index)
            if len(common) > 0 and "composite_score" in composite.columns:
                cur_s  = composite.loc[common, "composite_score"].astype(float)
                prev_s = last_scores.reindex(common).astype(float)
                d_score = (cur_s - prev_s).abs()
                freeze = common[d_score < score_band]
                if len(freeze) > 0:
                    weights_w = weights_w.copy()
                    weights_w.loc[freeze] = last_weights.loc[freeze]
                    total = float(weights_w.sum())
                    if total > 0:
                        weights_w = weights_w / total
        elif last_weights is not None and nt_band > 0:
            common = weights_w.index.intersection(last_weights.index)
            if len(common) > 0:
                diff = (weights_w.loc[common] - last_weights.loc[common]).abs()
                freeze = common[diff < nt_band]
                if len(freeze) > 0:
                    weights_w = weights_w.copy()
                    weights_w.loc[freeze] = last_weights.loc[freeze]
                    total = float(weights_w.sum())
                    if total > 0:
                        weights_w = weights_w / total

        # Drawdown overlay — scale gross exposure when VIX is panicky or
        # rolling NAV drawdown breaches the threshold. Remainder is held in cash
        # (returns 0 over the period).
        dd_scale = 1.0
        if portfolio_cfg.get("drawdown_overlay", False):
            cur_dd = (nav / nav_high) - 1.0 if nav_high > 0 else 0.0
            vix_pctile = float(signals.get("vix_pctile", float("nan"))) if signals else float("nan")
            dd_threshold = float(portfolio_cfg.get("drawdown_threshold", -0.08))
            vix_threshold = float(portfolio_cfg.get("drawdown_vix_pctile", 0.80))
            risk_off = (cur_dd <= dd_threshold) or (np.isfinite(vix_pctile) and vix_pctile >= vix_threshold)
            if risk_off:
                dd_scale = float(portfolio_cfg.get("drawdown_scale", 0.60))
        if dd_scale != 1.0:
            weights_w = weights_w * dd_scale

        # ---- Compute returns over [asof+1 .. next_event] ----
        if i + 1 < len(schedule):
            next_d = schedule[i + 1].date
        else:
            next_d = trading_dates[-1]
        period_dates = trading_dates[(trading_dates > asof) & (trading_dates <= next_d)]
        if len(period_dates) > 0:
            R_period = build_returns_panel(
                paths["prices_dir"], list(weights_w.index),
                start=period_dates[0], end=period_dates[-1],
            ).reindex(period_dates).fillna(0.0)

            # Apply turnover cost as a single haircut on day 1
            tc_bps = float(portfolio_cfg.get("trade_cost_bps", 0.0))
            cost = 0.0
            if last_weights is not None and tc_bps > 0:
                turnover = float((weights_w.reindex(last_weights.index, fill_value=0.0) - last_weights).abs().sum() / 2.0)
                cost = turnover * (tc_bps / 1e4)
            elif last_weights is None and tc_bps > 0:
                cost = float(weights_w.sum()) * (tc_bps / 1e4)

            tilt_daily = (R_period * weights_w.reindex(R_period.columns).fillna(0.0)).sum(axis=1)
            # Core-satellite blend: realized strategy daily = core·IWB + (1-core)·tilt.
            # Trade cost is charged on the tilt portion only.
            core_w = float(portfolio_cfg.get("core_iwb_weight", 0.0))
            core_w = float(np.clip(core_w, 0.0, 1.0))
            if core_w > 0:
                bench_period = bench_returns.reindex(R_period.index).fillna(0.0)
                daily = core_w * bench_period + (1.0 - core_w) * tilt_daily
                if cost > 0 and len(daily) > 0:
                    daily.iloc[0] = daily.iloc[0] - cost * (1.0 - core_w)
            else:
                daily = tilt_daily
                if cost > 0 and len(daily) > 0:
                    daily.iloc[0] = daily.iloc[0] - cost
            daily_returns.append(daily)
            nav = float(nav * (1.0 + daily).prod())
            nav_high = max(nav_high, nav)
            nav_snapshots.append({"date": asof.strftime("%Y-%m-%d"), "nav": nav,
                                  "regime": regime, "n_holdings": int((weights_w > 1e-6).sum()),
                                  "gross": float(weights_w.sum())})

        # Logging
        cdf = composite.reindex(weights_w.index).copy()
        cdf["weight"] = weights_w.values
        cdf["asof"] = asof
        constituent_log.append(cdf.reset_index().rename(columns={"index": "ticker"}))
        score_log.append(composite.assign(asof=asof).reset_index().rename(columns={"index": "ticker"}))

        last_weights = weights_w
        if "composite_score" in composite.columns:
            last_scores = composite["composite_score"].astype(float).copy()

    # ---- Final outputs ----
    daily = pd.concat(daily_returns).sort_index() if daily_returns else pd.Series(dtype=float)
    daily = daily[~daily.index.duplicated(keep="last")]
    nav_curve = (1.0 + daily).cumprod() * float(cfg["backtest"]["initial_nav"])

    bench_period = bench_returns.reindex(daily.index).fillna(0.0)
    bench_curve = (1.0 + bench_period).cumprod() * float(cfg["backtest"]["initial_nav"])

    # Persist
    daily.to_csv(out_dir / "alpha_daily_returns.csv", header=["return"])
    nav_curve.to_csv(out_dir / "alpha_nav.csv", header=["nav"])
    bench_curve.to_csv(out_dir / "benchmark_nav.csv", header=["nav"])
    pd.DataFrame(nav_snapshots).to_csv(out_dir / "rebalance_nav_snapshots.csv", index=False)
    pd.DataFrame(regime_log).to_csv(out_dir / "regime_log.csv", index=False)
    if pillar_weight_log:
        pd.DataFrame(pillar_weight_log).to_csv(out_dir / "pillar_weights_history.csv", index=False)
    if ic_tracker is not None:
        ic_tracker.to_dataframe().to_csv(out_dir / "pillar_ic_history.csv", index=False)
    if cfg["reporting"].get("write_constituents"):
        pd.concat(constituent_log, ignore_index=True).to_parquet(out_dir / "constituents.parquet")
    if cfg["reporting"].get("write_scores"):
        pd.concat(score_log, ignore_index=True).to_parquet(out_dir / "scores.parquet")

    # KPIs
    kpis = _summary_kpis(daily, bench_period)
    with open(out_dir / "kpis.json", "w") as fh:
        json.dump(kpis, fh, indent=2)

    LOG.info("Backtest complete.")
    LOG.info(json.dumps(kpis, indent=2))
    return {
        "kpis": kpis,
        "daily_returns": daily,
        "nav": nav_curve,
        "benchmark_nav": bench_curve,
    }


def _summary_kpis(daily: pd.Series, bench_daily: pd.Series) -> Dict[str, float]:
    if daily.empty:
        return {}
    ann = 252.0
    cagr = float((1.0 + daily).prod() ** (ann / len(daily)) - 1.0)
    vol = float(daily.std(ddof=0) * np.sqrt(ann))
    sharpe = float(daily.mean() / daily.std(ddof=0) * np.sqrt(ann)) if daily.std(ddof=0) > 0 else 0.0
    dd = float(((1.0 + daily).cumprod() / (1.0 + daily).cumprod().cummax() - 1.0).min())

    excess = (daily - bench_daily.reindex(daily.index).fillna(0.0))
    te = float(excess.std(ddof=0) * np.sqrt(ann))
    ir = float(excess.mean() / excess.std(ddof=0) * np.sqrt(ann)) if excess.std(ddof=0) > 0 else 0.0

    bench_cagr = float((1.0 + bench_daily.reindex(daily.index).fillna(0.0)).prod() ** (ann / len(daily)) - 1.0)

    return {
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "tracking_error": te,
        "information_ratio": ir,
        "benchmark_cagr": bench_cagr,
        "active_return": cagr - bench_cagr,
        "n_trading_days": int(len(daily)),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run()
