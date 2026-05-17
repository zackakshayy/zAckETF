#!/usr/bin/env python3
"""
SEC EDGAR fundamentals fetcher — deep, point-in-time, trusted-source data.

The EOD-Historical dataset carries only a 4-quarter slice for ~89% of the
Russell 1000, so historical backtests fall back to coarse yearly data and the
fundamental signal can't be tested fairly. This script pulls the full XBRL
filing history from the SEC's official `companyfacts` API — ~15 years deep,
and genuinely point-in-time: every fact carries the date it was `filed`, so
the backtest can use only data that was public as of each rebalance.

Output is written in the same on-disk schema `data/fundamentals.py` already
reads (`Financials.<stmt>.quarterly = {date: {field: value, filing_date}}`),
into a separate `fundamentals_edgar/` directory so the original data is never
destroyed. R1000 names that EDGAR can't serve are back-filled from the
existing EOD files, so the output directory has full R1000 coverage.

Usage:
    python scripts/fetch_edgar_fundamentals.py            # full R1000
    python scripts/fetch_edgar_fundamentals.py AAPL MSFT  # specific tickers
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

MASTER = Path("/Users/zackakshay/Desktop/ProjectX_MasterData")
EOD_DIR = MASTER / "categorized" / "fundamentals"
OUT_DIR = MASTER / "categorized" / "fundamentals_edgar"
R1000_XLSX = MASTER / "R1000.xlsx"

# SEC requires a descriptive User-Agent with contact info.
HEADERS = {"User-Agent": "AlphaEngine Research alpha-research@example.com"}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Candidate us-gaap tags per concept (first present wins). XBRL tag usage
# drifts across companies/years, so each concept lists fallbacks.
CONCEPT_TAGS: Dict[str, List[str]] = {
    "totalRevenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                     "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "grossProfit": ["GrossProfit"],
    "operatingIncome": ["OperatingIncomeLoss"],
    "netIncome": ["NetIncomeLoss", "ProfitLoss"],
    "depAmort": ["DepreciationDepletionAndAmortization",
                 "DepreciationAmortizationAndAccretionNet", "DepreciationAndAmortization"],
}
BALANCE_TAGS: Dict[str, List[str]] = {
    "totalStockholderEquity": ["StockholdersEquity",
                               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "longTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "currentDebt": ["LongTermDebtCurrent", "DebtCurrent"],
    "cashAndCashEquivalents": ["CashAndCashEquivalentsAtCarryingValue",
                               "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}
CASHFLOW_TAGS: Dict[str, List[str]] = {
    "totalCashFromOperatingActivities": ["NetCashProvidedByUsedInOperatingActivities",
                                         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capitalExpenditures": ["PaymentsToAcquirePropertyPlantAndEquipment",
                            "PaymentsForCapitalImprovements", "PaymentsToAcquireProductiveAssets"],
}


def _get_json(url: str, retries: int = 3) -> Optional[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return None


def load_cik_map() -> Dict[str, str]:
    data = _get_json(SEC_TICKERS_URL) or {}
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}


def _facts_for(gaap: dict, tags: List[str]) -> List[dict]:
    """Merge USD unit-facts across ALL candidate tags for a concept.

    Companies switch XBRL tags over time (e.g. AAPL moved revenue from
    `Revenues`/`SalesRevenueNet` to `RevenueFromContractWithCustomerExcluding-
    AssessedTax` in 2018). Returning only the first tag would silently drop
    the older history — so we merge every candidate tag and let
    `_earliest_by_period` dedupe by period-end.
    """
    merged: List[dict] = []
    for tag in tags:
        node = gaap.get(tag)
        if node and "units" in node:
            for unit_key in ("USD", "USD/shares"):
                if unit_key in node["units"]:
                    merged.extend(node["units"][unit_key])
    return merged


def _is_quarter(rec: dict) -> bool:
    """Flow fact spanning ~one fiscal quarter (80-100 days)."""
    s, e = rec.get("start"), rec.get("end")
    if not s or not e:
        return False
    days = (pd.Timestamp(e) - pd.Timestamp(s)).days
    return 80 <= days <= 100


def _is_year(rec: dict) -> bool:
    s, e = rec.get("start"), rec.get("end")
    if not s or not e:
        return False
    days = (pd.Timestamp(e) - pd.Timestamp(s)).days
    return 350 <= days <= 380


def _earliest_by_period(recs: List[dict]) -> Dict[str, dict]:
    """Key facts by period-end, keeping the EARLIEST-filed value (as first
    reported — the genuine point-in-time figure, pre-restatement)."""
    out: Dict[str, dict] = {}
    for r in recs:
        end = r.get("end")
        filed = r.get("filed")
        if not end or not filed or r.get("val") is None:
            continue
        if end not in out or filed < out[end].get("filed", "9999"):
            out[end] = r
    return out


def _flow_quarterly(gaap: dict, tags: List[str]) -> Dict[str, dict]:
    """Quarterly series for a flow concept. Q4 (missing from 10-Qs) is derived
    as annual − (Q1+Q2+Q3) of the same fiscal year when possible."""
    facts = _facts_for(gaap, tags)
    q = _earliest_by_period([r for r in facts if _is_quarter(r)])
    y = _earliest_by_period([r for r in facts if _is_year(r)])
    series: Dict[str, dict] = {
        end: {"val": float(r["val"]), "filed": r["filed"]} for end, r in q.items()
    }
    # Derive the missing fiscal-year-end quarter from the annual figure.
    for yend, yr in y.items():
        if yend in series:
            continue
        y_end_ts = pd.Timestamp(yend)
        prior = [series[e] for e in series
                 if 250 <= (y_end_ts - pd.Timestamp(e)).days <= 290 or
                    0 < (y_end_ts - pd.Timestamp(e)).days <= 290]
        # take the 3 quarters within ~9 months before the year-end
        within = [(e, series[e]) for e in series
                  if 0 < (y_end_ts - pd.Timestamp(e)).days <= 290]
        if len(within) >= 3:
            within.sort(key=lambda x: x[0], reverse=True)
            q3 = within[:3]
            q4_val = float(yr["val"]) - sum(v["val"] for _, v in q3)
            series[yend] = {"val": q4_val, "filed": yr["filed"]}
    return series


def _balance_quarterly(gaap: dict, tags: List[str]) -> Dict[str, dict]:
    """Instant (balance-sheet) series, keyed by period-end."""
    facts = _facts_for(gaap, tags)
    inst = _earliest_by_period([r for r in facts if r.get("end")])
    return {end: {"val": float(r["val"]), "filed": r["filed"]} for end, r in inst.items()}


def build_from_edgar(ticker: str, cik: str) -> Optional[Dict[str, Any]]:
    cf = _get_json(COMPANYFACTS_URL.format(cik=cik))
    if not cf or "facts" not in cf:
        return None
    gaap = cf["facts"].get("us-gaap", {})
    dei = cf["facts"].get("dei", {})
    if not gaap:
        return None

    # Flow concepts
    rev = _flow_quarterly(gaap, CONCEPT_TAGS["totalRevenue"])
    gp = _flow_quarterly(gaap, CONCEPT_TAGS["grossProfit"])
    opi = _flow_quarterly(gaap, CONCEPT_TAGS["operatingIncome"])
    ni = _flow_quarterly(gaap, CONCEPT_TAGS["netIncome"])
    da = _flow_quarterly(gaap, CONCEPT_TAGS["depAmort"])
    ocf = _flow_quarterly(gaap, CASHFLOW_TAGS["totalCashFromOperatingActivities"])
    capex = _flow_quarterly(gaap, CASHFLOW_TAGS["capitalExpenditures"])
    # Balance concepts
    eq = _balance_quarterly(gaap, BALANCE_TAGS["totalStockholderEquity"])
    ltd = _balance_quarterly(gaap, BALANCE_TAGS["longTermDebt"])
    cud = _balance_quarterly(gaap, BALANCE_TAGS["currentDebt"])
    cash = _balance_quarterly(gaap, BALANCE_TAGS["cashAndCashEquivalents"])

    def _income_block() -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for end in sorted(set(rev) | set(ni)):
            ebitda = None
            if end in opi and end in da:
                ebitda = opi[end]["val"] + da[end]["val"]
            filed = (rev.get(end) or ni.get(end) or opi.get(end) or {}).get("filed")
            out[end] = {
                "date": end, "filing_date": filed,
                "totalRevenue": rev.get(end, {}).get("val"),
                "grossProfit": gp.get(end, {}).get("val"),
                "operatingIncome": opi.get(end, {}).get("val"),
                "netIncome": ni.get(end, {}).get("val"),
                "ebitda": ebitda,
            }
        return out

    def _balance_block() -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for end in sorted(set(eq) | set(cash)):
            debt = None
            if end in ltd or end in cud:
                debt = (ltd.get(end, {}).get("val") or 0.0) + (cud.get(end, {}).get("val") or 0.0)
            out[end] = {
                "date": end, "filing_date": eq.get(end, {}).get("filed"),
                "totalStockholderEquity": eq.get(end, {}).get("val"),
                "totalDebt": debt,
                "cashAndCashEquivalents": cash.get(end, {}).get("val"),
            }
        return out

    def _cashflow_block() -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for end in sorted(set(ocf) | set(capex)):
            out[end] = {
                "date": end, "filing_date": ocf.get(end, {}).get("filed"),
                "totalCashFromOperatingActivities": ocf.get(end, {}).get("val"),
                # EDGAR reports capex as a positive payment; EOD/loader expects
                # it negative (FCF = CFO + capex), so negate.
                "capitalExpenditures": (-capex[end]["val"]) if end in capex else None,
            }
        return out

    # outstanding shares (dei)
    shares: Dict[str, dict] = {}
    sh_facts = []
    for tag in ("EntityCommonStockSharesOutstanding",):
        node = dei.get(tag)
        if node and "units" in node:
            for u in node["units"].get("shares", []):
                sh_facts.append(u)
    for i, r in enumerate(_earliest_by_period(sh_facts).values()):
        shares[str(i)] = {"date": r["end"], "dateFormatted": r["end"], "shares": float(r["val"])}

    income = _income_block()
    if len(income) < 8:   # too thin to be useful — signal a fallback
        return None

    return {
        "General": {"Code": ticker.upper(), "Name": cf.get("entityName"),
                    "GicSector": "", "Sector": "", "CountryISO": "US", "IsDelisted": False},
        "Highlights": {}, "Valuation": {}, "SharesStats": {},
        "Technicals": {}, "Earnings": {"History": {}, "Trend": {}, "Annual": {}},
        "Financials": {
            "Income_Statement": {"quarterly": income, "yearly": {}},
            "Balance_Sheet": {"quarterly": _balance_block(), "yearly": {}},
            "Cash_Flow": {"quarterly": _cashflow_block(), "yearly": {}},
        },
        "outstandingShares": {"quarterly": shares, "annual": {}},
        "DataSource": "SEC EDGAR companyfacts (point-in-time XBRL)",
    }


def r1000_tickers() -> List[str]:
    xl = pd.read_excel(R1000_XLSX, sheet_name="R1K")
    xl = xl[xl["Asset Type (Client Definition/ FactSet)"].astype(str).str.contains("Equity", na=False)]
    return sorted(xl["Ticker"].dropna().astype(str).str.strip().unique())


def main(tickers: List[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cik_map = load_cik_map()
    print(f"SEC ticker->CIK map: {len(cik_map)} companies")
    print(f"Fetching {len(tickers)} tickers -> {OUT_DIR}\n")

    ok = fallback = failed = 0
    for i, tk in enumerate(tickers, 1):
        out_path = OUT_DIR / f"{tk.upper()}.US.fundamentals.json"
        cik = cik_map.get(tk.upper().replace(".", "-")) or cik_map.get(tk.upper())
        blob = build_from_edgar(tk, cik) if cik else None
        if blob is not None:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(blob, fh, indent=1)
            ok += 1
            tag = "EDGAR"
        else:
            # Back-fill from the existing EOD file so coverage stays complete.
            eod = EOD_DIR / f"{tk.upper()}.US.fundamentals.json"
            if eod.is_file():
                shutil.copy2(eod, out_path)
                fallback += 1
                tag = "EOD-fallback"
            else:
                failed += 1
                tag = "MISSING"
        time.sleep(0.12)   # ~8 req/s, under SEC's 10 req/s limit
        if i % 50 == 0 or tag == "MISSING":
            print(f"  [{i}/{len(tickers)}] {tk}: {tag}  "
                  f"(edgar={ok} fallback={fallback} missing={failed})")

    print(f"\nDone. EDGAR={ok}  EOD-fallback={fallback}  missing={failed}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else r1000_tickers())
