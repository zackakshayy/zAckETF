"""Regression test for the two fundamentals on-disk schemas.

Schema A (full history): Financials.X.quarterly = {date: {fields}, ...}
Schema B (recent slice): Financials.X has top-level quarterly_last_0..N keys,
each mapping to a single period record.

Both must be flattened into the same as-of-correct frame and produce a
non-NaN revenue_ttm for a recent rebalance date.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_fund_json(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(blob, fh)


def _quarter(period_end: str, *, total_revenue: float, net_income: float,
             operating_income: float, gross_profit: float,
             ebitda: float = 0.0, filing_date: str | None = None) -> dict:
    return {
        "date": period_end,
        "filing_date": filing_date or (pd.Timestamp(period_end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        "currency_symbol": "USD",
        "totalRevenue": total_revenue,
        "netIncome": net_income,
        "operatingIncome": operating_income,
        "grossProfit": gross_profit,
        "ebitda": ebitda,
    }


def _balance(period_end: str, *, equity: float, debt: float, cash: float,
             filing_date: str | None = None) -> dict:
    return {
        "date": period_end,
        "filing_date": filing_date or (pd.Timestamp(period_end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        "currency_symbol": "USD",
        "totalStockholderEquity": equity,
        "totalDebt": debt,
        "cashAndCashEquivalents": cash,
    }


def _cashflow(period_end: str, *, cfo: float, capex: float,
              filing_date: str | None = None) -> dict:
    return {
        "date": period_end,
        "filing_date": filing_date or (pd.Timestamp(period_end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
        "currency_symbol": "USD",
        "totalCashFromOperatingActivities": cfo,
        "capitalExpenditures": capex,
    }


@pytest.fixture(scope="module")
def fund_dir():
    tmp = tempfile.mkdtemp(prefix="fund_test_")
    base = Path(tmp)

    # ---- Schema A: full quarterly history dict-of-dicts ----
    schema_a = {
        "General": {"GicSector": "Information Technology", "Sector": "Technology"},
        "Highlights": {"MarketCapitalization": 1_000_000_000_000.0},
        "Financials": {
            "Income_Statement": {
                "currency_symbol": "USD",
                "quarterly": {
                    "2024-12-31": _quarter("2024-12-31", total_revenue=120e9, net_income=30e9, operating_income=35e9, gross_profit=60e9, ebitda=40e9),
                    "2024-09-30": _quarter("2024-09-30", total_revenue=110e9, net_income=27e9, operating_income=33e9, gross_profit=55e9, ebitda=38e9),
                    "2024-06-30": _quarter("2024-06-30", total_revenue=100e9, net_income=24e9, operating_income=30e9, gross_profit=50e9, ebitda=34e9),
                    "2024-03-31": _quarter("2024-03-31", total_revenue= 95e9, net_income=22e9, operating_income=28e9, gross_profit=48e9, ebitda=32e9),
                    "2023-12-31": _quarter("2023-12-31", total_revenue= 90e9, net_income=20e9, operating_income=26e9, gross_profit=45e9, ebitda=30e9),
                },
            },
            "Balance_Sheet": {
                "currency_symbol": "USD",
                "quarterly": {
                    "2024-12-31": _balance("2024-12-31", equity=200e9, debt=50e9, cash=70e9),
                    "2024-09-30": _balance("2024-09-30", equity=190e9, debt=50e9, cash=68e9),
                    "2024-06-30": _balance("2024-06-30", equity=185e9, debt=52e9, cash=64e9),
                    "2024-03-31": _balance("2024-03-31", equity=180e9, debt=53e9, cash=60e9),
                },
            },
            "Cash_Flow": {
                "currency_symbol": "USD",
                "quarterly": {
                    "2024-12-31": _cashflow("2024-12-31", cfo=33e9, capex=-10e9),
                    "2024-09-30": _cashflow("2024-09-30", cfo=30e9, capex=-9e9),
                    "2024-06-30": _cashflow("2024-06-30", cfo=28e9, capex=-9e9),
                    "2024-03-31": _cashflow("2024-03-31", cfo=27e9, capex=-8e9),
                },
            },
        },
        "outstandingShares": {
            "quarterly": {
                "0": {"date": "2024-Q4", "dateFormatted": "2024-12-31", "shares": 16e9},
            },
        },
    }
    _write_fund_json(base / "AAAA.US.fundamentals.json", schema_a)

    # ---- Schema B: top-level quarterly_last_* / yearly_last_* records ----
    schema_b = {
        "General": {"GicSector": None, "Sector": "Technology"},
        "Highlights": {"MarketCapitalization": 500_000_000_000.0},
        "Financials": {
            "Income_Statement": {
                "currency_symbol": "USD",
                "quarterly_last_0": _quarter("2024-12-31", total_revenue=40e9, net_income=22e9, operating_income=24e9, gross_profit=30e9, ebitda=27e9),
                "quarterly_last_1": _quarter("2024-09-30", total_revenue=35e9, net_income=19e9, operating_income=21e9, gross_profit=26e9, ebitda=24e9),
                "quarterly_last_2": _quarter("2024-06-30", total_revenue=30e9, net_income=15e9, operating_income=18e9, gross_profit=22e9, ebitda=20e9),
                "quarterly_last_3": _quarter("2024-03-31", total_revenue=26e9, net_income=14e9, operating_income=15e9, gross_profit=19e9, ebitda=17e9),
                "yearly_last_0":  _quarter("2024-12-31", total_revenue=131e9, net_income=70e9, operating_income=78e9, gross_profit=97e9, ebitda=88e9),
                "yearly_last_1":  _quarter("2023-12-31", total_revenue=80e9,  net_income=42e9, operating_income=48e9, gross_profit=58e9, ebitda=53e9),
            },
            "Balance_Sheet": {
                "currency_symbol": "USD",
                "quarterly_last_0": _balance("2024-12-31", equity=66e9, debt=8e9, cash=9e9),
                "quarterly_last_1": _balance("2024-09-30", equity=60e9, debt=8e9, cash=8e9),
                "quarterly_last_2": _balance("2024-06-30", equity=55e9, debt=7e9, cash=8e9),
                "quarterly_last_3": _balance("2024-03-31", equity=50e9, debt=7e9, cash=7e9),
                "yearly_last_0":   _balance("2024-12-31", equity=66e9, debt=8e9, cash=9e9),
                "yearly_last_1":   _balance("2023-12-31", equity=46e9, debt=6e9, cash=7e9),
            },
            "Cash_Flow": {
                "currency_symbol": "USD",
                "quarterly_last_0": _cashflow("2024-12-31", cfo=20e9, capex=-2e9),
                "quarterly_last_1": _cashflow("2024-09-30", cfo=18e9, capex=-2e9),
                "quarterly_last_2": _cashflow("2024-06-30", cfo=16e9, capex=-2e9),
                "quarterly_last_3": _cashflow("2024-03-31", cfo=14e9, capex=-1e9),
            },
        },
        # Schema-B files often lack outstandingShares — must still produce sensible features.
    }
    _write_fund_json(base / "BBBB.US.fundamentals.json", schema_b)

    yield base


def test_schema_a_full_history(fund_dir):
    from project_alpha.data import point_in_time_features
    f = point_in_time_features(fund_dir, "AAAA", pd.Timestamp("2025-03-31"), last_price=200.0)
    assert np.isfinite(f["revenue_ttm"])
    # Sum of 2024 four quarters: 95 + 100 + 110 + 120 = 425e9
    assert abs(f["revenue_ttm"] - 425e9) < 1
    assert np.isfinite(f["fcf_ttm"])  # CFO + CapEx (CapEx is negative): 33+30+28+27 + (-10-9-9-8) = 82e9
    assert abs(f["fcf_ttm"] - 82e9) < 1


def test_schema_b_quarterly_last(fund_dir):
    from project_alpha.data import point_in_time_features
    f = point_in_time_features(fund_dir, "BBBB", pd.Timestamp("2025-03-31"), last_price=400.0)
    assert np.isfinite(f["revenue_ttm"])
    # 26 + 30 + 35 + 40 = 131e9
    assert abs(f["revenue_ttm"] - 131e9) < 1
    assert np.isfinite(f["operating_income_ttm"])
    assert np.isfinite(f["net_income_ttm"])
    assert np.isfinite(f["equity"])
    assert f["equity"] == 66e9
    # P/E should be a positive finite ratio
    assert np.isfinite(f["pe"]) and f["pe"] > 0
    # FCF: cfo+capex over 4 q = (20+18+16+14) + (-2-2-2-1) = 61e9
    assert abs(f["fcf_ttm"] - 61e9) < 1


def test_schema_b_yearly_fallback(fund_dir):
    """For an asof BEFORE the most recent 4 quarterly periods are filed,
    the loader should fall back to yearly_last_*."""
    from project_alpha.data import point_in_time_features
    # Two years before the file's data — quarterlies are too recent (their
    # available_date > 2023-06-30), but yearly_last_1 (filed 2024-01-30) is also too recent.
    # Pick a date where yearly_last_1 (2023-12-31, filed 2024-01-30) is too recent
    # but the file is otherwise valid: 2023-12-31's filing was 2024-01-30, so asof < that excludes it.
    # Use asof = 2024-02-15 → all quarterly_last_* are filed AFTER it (2024-04-30 etc.) but
    # yearly_last_1 (filed 2024-01-30) IS available.
    f = point_in_time_features(fund_dir, "BBBB", pd.Timestamp("2024-02-15"), last_price=350.0)
    # Quarterly TTM unavailable; yearly fallback supplies revenue_ttm.
    assert np.isfinite(f["revenue_ttm"])
    assert abs(f["revenue_ttm"] - 80e9) < 1   # yearly_last_1 totalRevenue = 80e9


def test_no_lookahead_schema_b(fund_dir):
    """Asof in 2020 must produce NaN for the all-2024-data Schema B file."""
    from project_alpha.data import point_in_time_features
    f = point_in_time_features(fund_dir, "BBBB", pd.Timestamp("2020-12-31"), last_price=400.0)
    assert not np.isfinite(f.get("revenue_ttm", float("nan")))
