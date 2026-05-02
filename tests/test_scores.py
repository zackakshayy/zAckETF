"""Score-module tests: invariants on inputs you can construct synthetically."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Fundamental panel
# -----------------------------------------------------------------------------

def _make_fund_rows(n_per_sector: int = 30):
    """Build a deterministic fundamental panel across two sectors."""
    rng = np.random.default_rng(7)
    rows = []
    for sec in ["Tech", "Health Care"]:
        for i in range(n_per_sector):
            rev = float(rng.uniform(1e9, 1e11))
            ni = float(rev * rng.uniform(-0.05, 0.20))
            equity = float(rng.uniform(1e9, 5e10))
            rows.append({
                "ticker":          f"{sec[:3].upper()}{i:03d}",
                "sector":          sec,
                "revenue_ttm":     rev,
                "net_income_ttm":  ni,
                "operating_income_ttm": ni * 1.2,
                "gross_profit_ttm": rev * float(rng.uniform(0.2, 0.7)),
                "equity":          equity,
                "equity_avg4":     equity,
                "fcf_ttm":         ni * 0.8,
                "market_cap":      rev * float(rng.uniform(1.0, 8.0)),
                "gross_margin":    float(rng.uniform(0.1, 0.7)),
                "operating_margin": float(rng.uniform(-0.1, 0.4)),
                "net_margin":      ni / rev,
                "fcf_margin":      0.7 * (ni / rev),
                "roe":             ni / equity,
                "roa":             ni / (equity * 1.5),
                "debt_to_equity":  float(rng.uniform(0.0, 3.0)),
                "pe":              float(rng.uniform(5, 60)),
                "pb":              float(rng.uniform(0.5, 12)),
                "ps":              float(rng.uniform(0.5, 25)),
                "fcf_yield":       float(rng.uniform(-0.02, 0.10)),
                "rev_growth_yoy":  float(rng.normal(0.05, 0.10)),
                "eps_growth_yoy":  float(rng.normal(0.05, 0.30)),
                "ev_ebitda":       float(rng.uniform(5, 40)),
            })
    return rows


def test_fundamental_panel_sector_neutral():
    """Within each sector, the score column must have ≈ zero mean."""
    from project_alpha.scores.fundamental import compute_fundamental_panel
    panel = compute_fundamental_panel(_make_fund_rows(n_per_sector=40))
    for sec, sub in panel.groupby("sector"):
        assert abs(float(sub["fundamental_score"].mean())) < 0.05


def test_fundamental_panel_value_inversion():
    """A name with low PE should score higher in value than one with high PE."""
    from project_alpha.scores.fundamental import compute_fundamental_panel
    rows = _make_fund_rows(n_per_sector=30)
    rows[0]["pe"] = 5.0
    rows[1]["pe"] = 100.0
    rows[0]["ticker"] = "CHEAP"
    rows[1]["ticker"] = "RICH"
    panel = compute_fundamental_panel(rows)
    assert panel.loc["CHEAP", "value_z"] > panel.loc["RICH", "value_z"]


# -----------------------------------------------------------------------------
# Technical features
# -----------------------------------------------------------------------------

def _make_eod(price_path: np.ndarray, *, start="2018-01-02") -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=len(price_path))
    return pd.DataFrame({
        "date": dates,
        "open": price_path,
        "high": price_path * 1.01,
        "low":  price_path * 0.99,
        "close": price_path,
        "adjusted_close": price_path,
        "volume": np.full(len(price_path), 1_000_000.0),
    })


def test_technical_no_lookahead():
    """Features must use only data up to and including asof."""
    from project_alpha.scores.technical import compute_technical_features
    px = np.linspace(100, 200, 800)  # gentle uptrend
    eod = _make_eod(px)
    asof_early = eod["date"].iloc[300]
    asof_late = eod["date"].iloc[700]
    f_early = compute_technical_features(eod, asof_early)
    f_late = compute_technical_features(eod, asof_late)
    assert f_early["price"] < f_late["price"]
    # Adding future rows must not change f_early
    eod_future = pd.concat([eod, eod.assign(date=eod["date"] + pd.Timedelta(days=10*365))], ignore_index=True)
    f_early_2 = compute_technical_features(eod_future, asof_early)
    assert abs(f_early["price"] - f_early_2["price"]) < 1e-9
    if pd.notna(f_early.get("mom_12_1")) and pd.notna(f_early_2.get("mom_12_1")):
        assert abs(f_early["mom_12_1"] - f_early_2["mom_12_1"]) < 1e-9


def test_technical_panel_sector_neutral():
    from project_alpha.scores.technical import compute_technical_features, compute_technical_panel
    rng = np.random.default_rng(11)
    rows = []
    sector_map = {}
    for sec in ["Tech", "Health Care"]:
        for i in range(30):
            ticker = f"{sec[:3].upper()}{i:03d}"
            sector_map[ticker] = sec
            px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
            eod = _make_eod(px)
            f = compute_technical_features(eod, eod["date"].iloc[-1])
            f["ticker"] = ticker
            rows.append(f)
    panel = compute_technical_panel(rows, sector_map=sector_map)
    for sec, sub in panel.groupby("sector"):
        assert abs(float(sub["technical_score"].mean())) < 0.10


# -----------------------------------------------------------------------------
# Sentiment aggregator
# -----------------------------------------------------------------------------

def test_sentiment_decay_recency_bias():
    """A recent positive article should outweigh an old negative one with same |polarity|."""
    from project_alpha.data.news import aggregate_sentiment_window
    asof = pd.Timestamp("2023-12-31")
    df = pd.DataFrame([
        {"date": asof - pd.Timedelta(days=1), "polarity": +0.8, "n_symbols": 5, "ticker_in_symbols": True},
        {"date": asof - pd.Timedelta(days=85), "polarity": -0.8, "n_symbols": 5, "ticker_in_symbols": True},
        {"date": asof - pd.Timedelta(days=30), "polarity": 0.0,  "n_symbols": 5, "ticker_in_symbols": True},
    ])
    agg = aggregate_sentiment_window(df, asof, half_life_days=30, min_articles=1)
    assert agg["polarity_decay"] > 0


def test_sentiment_relevance_inverse_symbol_count():
    """Articles tagged with many symbols should weigh less than ticker-specific ones."""
    from project_alpha.data.news import aggregate_sentiment_window
    asof = pd.Timestamp("2023-12-31")
    df = pd.DataFrame([
        {"date": asof - pd.Timedelta(days=1), "polarity": +0.5, "n_symbols": 1,  "ticker_in_symbols": True},
        {"date": asof - pd.Timedelta(days=1), "polarity": -0.5, "n_symbols": 50, "ticker_in_symbols": True},
    ])
    agg = aggregate_sentiment_window(df, asof, relevance_weight="inverse_symbol_count", min_articles=1)
    assert agg["polarity_decay"] > 0


# -----------------------------------------------------------------------------
# Regime classifier
# -----------------------------------------------------------------------------

def test_regime_extreme_vix_is_risk_off():
    """Crank VIX up to a top-percentile level → expect risk_off."""
    import numpy as np
    from project_alpha.scores.regime import classify_regime
    dates = pd.bdate_range("2010-01-01", periods=4000)
    vix = pd.Series(np.linspace(10, 40, len(dates)), index=dates, name="VIXCLS")
    # T10Y2Y inverted
    slope = pd.Series(-0.5, index=dates, name="T10Y2Y")
    # Hike cycle
    ff_diff = pd.Series(np.linspace(-0.5, 1.0, len(dates)), index=dates, name="FEDFUNDS_DIFF_3M")
    # Rising unemployment
    un_diff = pd.Series(np.linspace(-0.1, 0.5, len(dates)), index=dates, name="UNRATE_DIFF_3M")
    panel = pd.concat([vix, slope, ff_diff, un_diff], axis=1)
    label, sig = classify_regime(panel, dates[-1])
    assert label == "risk_off"


def test_regime_calm_market_is_risk_on():
    import numpy as np
    from project_alpha.scores.regime import classify_regime
    dates = pd.bdate_range("2010-01-01", periods=4000)
    vix = pd.Series(np.linspace(40, 12, len(dates)), index=dates, name="VIXCLS")  # ends low
    slope = pd.Series(np.linspace(0.0, 1.5, len(dates)), index=dates, name="T10Y2Y")
    ff_diff = pd.Series(np.linspace(0.5, -0.6, len(dates)), index=dates, name="FEDFUNDS_DIFF_3M")
    un_diff = pd.Series(np.linspace(0.4, -0.2, len(dates)), index=dates, name="UNRATE_DIFF_3M")
    panel = pd.concat([vix, slope, ff_diff, un_diff], axis=1)
    label, sig = classify_regime(panel, dates[-1])
    assert label == "risk_on"


def test_pillar_weights_sum_to_one_after_tilt():
    from project_alpha.scores.regime import pillar_weights_for_regime
    base = {"technical": 0.35, "fundamental": 0.30, "sentiment": 0.20, "macro": 0.15}
    tilt_off = {"technical": -0.10, "fundamental": +0.05, "sentiment": -0.05, "macro": +0.10}
    w = pillar_weights_for_regime(base, "risk_off", risk_off_tilt=tilt_off)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["macro"] > base["macro"]
    assert w["technical"] < base["technical"]
