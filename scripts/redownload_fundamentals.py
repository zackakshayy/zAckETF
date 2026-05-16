#!/usr/bin/env python3
"""
Targeted fundamentals re-download from Yahoo Finance (trusted public source).

Patches gaps in the EOD-Historical fundamentals dataset by pulling the same
financial statements from Yahoo Finance via the `yfinance` library and writing
them in the on-disk EOD schema that `data/fundamentals.py` already consumes.

Usage:
    python scripts/redownload_fundamentals.py MSFT
    python scripts/redownload_fundamentals.py MSFT BRK.B BF.B   # multiple

The output JSON matches Schema A (Financials.<stmt>.quarterly = {date: {...}}),
so `point_in_time_features` reads it with no code change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

FUND_DIR = Path("/Users/zackakshay/Desktop/ProjectX_MasterData/categorized/fundamentals")

# Yahoo sector names -> GICS sector names used elsewhere in the project
_YH_SECTOR_TO_GICS = {
    "Technology": "Information Technology",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}

# EOD field name -> yfinance row label
_INCOME_MAP = {
    "totalRevenue": "Total Revenue",
    "grossProfit": "Gross Profit",
    "operatingIncome": "Operating Income",
    "netIncome": "Net Income",
    "ebitda": "EBITDA",
}
_BALANCE_MAP = {
    "totalStockholderEquity": "Stockholders Equity",
    "totalDebt": "Total Debt",
    "cashAndCashEquivalents": "Cash And Cash Equivalents",
}
_CASHFLOW_MAP = {
    "totalCashFromOperatingActivities": "Operating Cash Flow",
    "capitalExpenditures": "Capital Expenditure",
}


def _num(x: Any) -> Optional[float]:
    """Convert to a JSON-safe float (None for NaN/inf/missing)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _statement_block(df: pd.DataFrame, field_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Convert a yfinance statement frame (rows=line items, cols=dates) into
    the EOD `{date_str: {field: value, ...}}` dict-of-dicts shape."""
    block: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return block
    for col in df.columns:
        date_str = pd.Timestamp(col).strftime("%Y-%m-%d")
        rec: Dict[str, Any] = {"date": date_str, "filing_date": None}
        for eod_field, yh_label in field_map.items():
            rec[eod_field] = _num(df.loc[yh_label, col]) if yh_label in df.index else None
        block[date_str] = rec
    return block


def _yahoo_ticker(ticker: str) -> str:
    """EOD ticker -> Yahoo ticker (share classes use '-', e.g. BRK.B -> BRK-B)."""
    return ticker.upper().replace(".", "-")


def build_fundamentals_json(ticker: str) -> Dict[str, Any]:
    yh = _yahoo_ticker(ticker)
    t = yf.Ticker(yh)
    info = t.info or {}

    # --- Financials: quarterly + yearly ---
    financials = {}
    for stmt_name, getter_q, getter_y, fmap in (
        ("Income_Statement", "quarterly_income_stmt", "income_stmt", _INCOME_MAP),
        ("Balance_Sheet", "quarterly_balance_sheet", "balance_sheet", _BALANCE_MAP),
        ("Cash_Flow", "quarterly_cashflow", "cashflow", _CASHFLOW_MAP),
    ):
        financials[stmt_name] = {
            "quarterly": _statement_block(getattr(t, getter_q), fmap),
            "yearly": _statement_block(getattr(t, getter_y), fmap),
        }

    # --- outstandingShares (from balance sheet 'Ordinary Shares Number') ---
    out_shares: Dict[str, Any] = {}
    qb = t.quarterly_balance_sheet
    if qb is not None and not qb.empty and "Ordinary Shares Number" in qb.index:
        for i, col in enumerate(qb.columns):
            d = pd.Timestamp(col).strftime("%Y-%m-%d")
            out_shares[str(i)] = {"date": d, "dateFormatted": d,
                                  "shares": _num(qb.loc["Ordinary Shares Number", col])}

    # --- Earnings.History (epsActual / epsEstimate / surprisePercent) ---
    earnings_history: Dict[str, Any] = {}
    try:
        eh = t.get_earnings_history()
        if eh is not None and not eh.empty:
            for q, row in eh.iterrows():
                d = pd.Timestamp(q).strftime("%Y-%m-%d")
                sp = _num(row.get("surprisePercent"))
                earnings_history[d] = {
                    "reportDate": d, "date": d,
                    "epsActual": _num(row.get("epsActual")),
                    "epsEstimate": _num(row.get("epsEstimate")),
                    "epsDifference": _num(row.get("epsDifference")),
                    # EOD stores surprisePercent as a percent (e.g. 8.09), yfinance as 0.0809
                    "surprisePercent": (sp * 100.0) if sp is not None else None,
                }
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] earnings history unavailable for {ticker}: {exc}")

    sector = _YH_SECTOR_TO_GICS.get(info.get("sector", ""), info.get("sector", ""))

    return {
        "General": {
            "Code": ticker.upper(),
            "Name": info.get("longName") or info.get("shortName"),
            "GicSector": sector,
            "Sector": sector,
            "Industry": info.get("industry"),
            "CountryISO": "US",
            "IsDelisted": False,
        },
        "Highlights": {
            "MarketCapitalization": _num(info.get("marketCap")),
            "PERatio": _num(info.get("trailingPE")),
            "PEGRatio": _num(info.get("trailingPegRatio")),
            "BookValue": _num(info.get("bookValue")),
            "DividendYield": _num(info.get("dividendYield")),
            "EarningsShare": _num(info.get("trailingEps")),
            "ProfitMargin": _num(info.get("profitMargins")),
            "OperatingMarginTTM": _num(info.get("operatingMargins")),
            "ReturnOnAssetsTTM": _num(info.get("returnOnAssets")),
            "ReturnOnEquityTTM": _num(info.get("returnOnEquity")),
            "RevenueTTM": _num(info.get("totalRevenue")),
            "GrossProfitTTM": _num(info.get("grossProfits")),
            "DilutedEpsTTM": _num(info.get("trailingEps")),
            "QuarterlyRevenueGrowthYOY": _num(info.get("revenueGrowth")),
            "QuarterlyEarningsGrowthYOY": _num(info.get("earningsGrowth")),
        },
        "Valuation": {
            "TrailingPE": _num(info.get("trailingPE")),
            "ForwardPE": _num(info.get("forwardPE")),
            "PriceSalesTTM": _num(info.get("priceToSalesTrailing12Months")),
            "PriceBookMRQ": _num(info.get("priceToBook")),
            "EnterpriseValue": _num(info.get("enterpriseValue")),
            "EnterpriseValueRevenue": _num(info.get("enterpriseToRevenue")),
            "EnterpriseValueEbitda": _num(info.get("enterpriseToEbitda")),
        },
        "SharesStats": {"SharesOutstanding": _num(info.get("sharesOutstanding"))},
        "Technicals": {"Beta": _num(info.get("beta"))},
        "Earnings": {"History": earnings_history, "Trend": {}, "Annual": {}},
        "Financials": financials,
        "outstandingShares": {"quarterly": out_shares, "annual": {}},
        "DataSource": "yfinance (Yahoo Finance) targeted re-download",
    }


def main(tickers: list[str]) -> None:
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    for tk in tickers:
        print(f"Re-downloading {tk} (Yahoo: {_yahoo_ticker(tk)}) ...")
        try:
            blob = build_fundamentals_json(tk)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {tk}: {exc}")
            continue
        out_path = FUND_DIR / f"{tk.upper()}.US.fundamentals.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2)
        q = len(blob["Financials"]["Income_Statement"]["quarterly"])
        y = len(blob["Financials"]["Income_Statement"]["yearly"])
        e = len(blob["Earnings"]["History"])
        print(f"  [OK] wrote {out_path.name}  ({q} qtr, {y} yr, {e} earnings recs)")


if __name__ == "__main__":
    args = sys.argv[1:] or ["MSFT"]
    main(args)
