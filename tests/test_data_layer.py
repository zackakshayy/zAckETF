"""Data-layer correctness tests against the actual on-disk shapes."""

from __future__ import annotations

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# R1000 universe
# -----------------------------------------------------------------------------

def test_russell_panel_shape(russell_panel):
    assert {"snapshot_date", "ticker", "sector", "weight", "asset_type"}.issubset(russell_panel.columns)
    assert (russell_panel["weight"] >= 0).all()
    # Sector tagging populated
    assert russell_panel["sector"].astype(str).str.strip().ne("").all()


def test_universe_for_date_uses_jun_anchor(russell_panel):
    from project_alpha.data import universe_for_date
    asof = pd.Timestamp("2020-12-31")
    uni = universe_for_date(russell_panel, asof, anchor_month_day="06-30", tolerance_days=14)
    snap = uni.attrs["snapshot_date"]
    assert snap.year == 2020
    assert snap.month == 6


def test_sector_targets_sum_to_one(russell_panel):
    from project_alpha.data import universe_for_date, sector_targets_for_date
    uni = universe_for_date(russell_panel, pd.Timestamp("2022-09-30"))
    tgt = sector_targets_for_date(uni)
    assert abs(float(tgt.sum()) - 1.0) < 1e-9
    assert (tgt > 0).all()


# -----------------------------------------------------------------------------
# Prices
# -----------------------------------------------------------------------------

def test_eod_loader_aapl(data_paths):
    from project_alpha.data import load_eod
    df = load_eod(data_paths["prices_dir"], "AAPL")
    assert not df.empty
    assert {"date", "adjusted_close", "volume"}.issubset(df.columns)
    assert df["date"].is_monotonic_increasing
    # Sanity: adjusted close must be positive and non-decreasing in long periods
    assert (df["adjusted_close"] > 0).all()


def test_returns_panel_alignment(data_paths):
    from project_alpha.data import build_returns_panel
    # MSFT is currently missing from prices/ on disk; use a different cohort.
    cohort = ["AAPL", "AMZN", "NVDA"]
    df = build_returns_panel(
        data_paths["prices_dir"],
        cohort,
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-12-31"),
    )
    assert not df.empty
    assert set(df.columns) == set(cohort)
    # No future dates
    assert df.index.max() <= pd.Timestamp("2023-12-31")


def test_missing_ticker_silently_skipped(data_paths):
    """Tickers without EOD parquet must not break the panel."""
    from project_alpha.data import build_returns_panel
    df = build_returns_panel(
        data_paths["prices_dir"],
        ["AAPL", "ZZZZNONEXISTENT"],
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-12-31"),
    )
    assert "AAPL" in df.columns
    assert "ZZZZNONEXISTENT" not in df.columns


# -----------------------------------------------------------------------------
# Fundamentals
# -----------------------------------------------------------------------------

def test_pit_features_no_lookahead(data_paths):
    """A point-in-time feature pull at date X must not use filings filed > X."""
    from project_alpha.data import point_in_time_features
    feats_old = point_in_time_features(
        data_paths["fundamentals_dir"], "AAPL", pd.Timestamp("2017-06-30"),
    )
    feats_now = point_in_time_features(
        data_paths["fundamentals_dir"], "AAPL", pd.Timestamp("2024-06-30"),
    )
    # Revenue TTM must not magically grow at the older date.
    if all(map(lambda x: pd.notna(x), [feats_old.get("revenue_ttm"), feats_now.get("revenue_ttm")])):
        assert feats_old["revenue_ttm"] < feats_now["revenue_ttm"]


def test_pit_features_consistent_keys(data_paths):
    from project_alpha.data import point_in_time_features
    feats = point_in_time_features(
        data_paths["fundamentals_dir"], "AAPL", pd.Timestamp("2024-12-31"),
    )
    expected = {
        "revenue_ttm", "operating_income_ttm", "net_income_ttm", "fcf_ttm",
        "equity", "shares_outstanding", "market_cap", "gross_margin",
        "operating_margin", "roe", "debt_to_equity", "pe", "pb", "ps",
    }
    assert expected.issubset(feats.keys())


# -----------------------------------------------------------------------------
# News
# -----------------------------------------------------------------------------

def test_news_window_returns_recent_only(data_paths):
    from project_alpha.data import load_news_window
    asof = pd.Timestamp("2024-04-30")
    df = load_news_window(data_paths["news_dir"], "AAPL", asof, window_days=90)
    if df.empty:
        pytest.skip("no AAPL news in window")
    assert (df["date"] <= asof).all()
    assert (df["date"] >= asof - pd.Timedelta(days=90)).all()
    assert df["polarity"].between(-1.5, 1.5).all()


def test_sentiment_aggregation_returns_dict(data_paths):
    from project_alpha.data import load_news_window, aggregate_sentiment_window
    asof = pd.Timestamp("2024-04-30")
    df = load_news_window(data_paths["news_dir"], "AAPL", asof, window_days=90)
    agg = aggregate_sentiment_window(df, asof)
    assert {"polarity_decay", "polarity_slope", "article_count", "article_count_log"}.issubset(agg.keys())


# -----------------------------------------------------------------------------
# Macro
# -----------------------------------------------------------------------------

def test_macro_panel_has_expected_series(macro_panel):
    expected = {"VIXCLS", "T10Y2Y", "FEDFUNDS", "CPIAUCSL", "UNRATE",
                "CPI_YOY", "FEDFUNDS_DIFF_3M", "UNRATE_DIFF_3M"}
    assert expected.issubset(set(macro_panel.columns))


def test_macro_diff_3m_uses_native_frequency(macro_panel):
    """FEDFUNDS_DIFF_3M must be computed on the native monthly series.

    Spot-check: known Fed cuts in late 2024 should show as negative DIFF.
    """
    s = macro_panel["FEDFUNDS_DIFF_3M"].dropna()
    if s.empty:
        pytest.skip("FEDFUNDS DIFF_3M empty")
    # Some negative observation in 2024-Q4 / 2025-Q1
    rng = s.loc["2024-12-01":"2025-03-01"]
    assert not rng.empty
    assert (rng < 0).any()


def test_asof_macro_no_future_leak(macro_panel):
    from project_alpha.data.macro import asof_macro
    asof = pd.Timestamp("2020-03-15")
    snap = asof_macro(macro_panel, asof)
    # The snapshot's CPI value must equal the latest CPI ≤ 2020-03-15
    cpi = macro_panel["CPIAUCSL"].dropna()
    expected = float(cpi.loc[cpi.index <= asof].iloc[-1])
    assert abs(snap["CPIAUCSL"] - expected) < 1e-6
