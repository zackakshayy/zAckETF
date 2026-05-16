"""
Fundamentals loader for EOD-Historical-style JSON.

Two modes:
  1. `load_fundamental_features` returns the **current** Highlights/Valuation
     ratios from the JSON. This is the cheap path used for live ETF scoring.

  2. `point_in_time_features` rebuilds TTM ratios from the quarterly
     Income/Balance/Cash-Flow blocks **using only data with filing_date or
     period-end <= asof**. This is what the backtest uses to avoid look-ahead.

Field naming on disk (keys we actually rely on):
  General.GicSector, General.Sector
  Highlights.{MarketCapitalization, MarketCapitalizationMln, EBITDA, PERatio,
              PEGRatio, BookValue, DividendShare, DividendYield, EarningsShare,
              ProfitMargin, OperatingMarginTTM, ReturnOnAssetsTTM,
              ReturnOnEquityTTM, RevenueTTM, GrossProfitTTM, DilutedEpsTTM,
              QuarterlyRevenueGrowthYOY, QuarterlyEarningsGrowthYOY}
  Valuation.{TrailingPE, ForwardPE, PriceSalesTTM, PriceBookMRQ,
             EnterpriseValue, EnterpriseValueRevenue, EnterpriseValueEbitda}
  SharesStats.SharesOutstanding
  Technicals.Beta
  Financials.{Income_Statement,Balance_Sheet,Cash_Flow}.{quarterly,yearly}
  outstandingShares.quarterly[*]  (date, dateFormatted, shares)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _path_for(fund_dir: Path, ticker: str) -> Optional[Path]:
    t = ticker.upper()
    for p in (
        fund_dir / f"{t}.US.fundamentals.json",
        fund_dir / f"{t}.fundamentals.json",
        fund_dir / f"{t}.json",
    ):
        if p.is_file():
            return p
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _to_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except (TypeError, ValueError):
        return float("nan")


def _is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


# ---------------------------------------------------------------------------
# Live (current) features — for live scoring, NOT backtest
# ---------------------------------------------------------------------------

def load_fundamental_features(fund_dir: str | Path, ticker: str) -> Dict[str, Any]:
    """Return the latest Highlights/Valuation ratios as a flat dict.

    No look-ahead protection — only safe for *live* scoring on the system
    date. For historical backtests use `point_in_time_features`.
    """
    p = _path_for(Path(fund_dir), ticker)
    if p is None:
        return {}
    blob = _load_json(p)
    h = blob.get("Highlights") or {}
    v = blob.get("Valuation") or {}
    s = blob.get("SharesStats") or {}
    g = blob.get("General") or {}
    t = blob.get("Technicals") or {}

    out: Dict[str, Any] = {
        "ticker":             ticker.upper(),
        "sector":             g.get("GicSector") or g.get("Sector") or "",
        "industry":           g.get("Industry"),
        "country":            g.get("CountryISO") or g.get("CountryName"),
        "is_delisted":        bool(g.get("IsDelisted") or False),

        "market_cap":         _to_float(h.get("MarketCapitalization")),
        "ebitda":             _to_float(h.get("EBITDA")),
        "pe_ratio":           _to_float(h.get("PERatio")),
        "peg_ratio":          _to_float(h.get("PEGRatio")),
        "book_value":         _to_float(h.get("BookValue")),
        "dividend_yield":     _to_float(h.get("DividendYield")),
        "eps":                _to_float(h.get("EarningsShare")),
        "profit_margin":      _to_float(h.get("ProfitMargin")),
        "operating_margin":   _to_float(h.get("OperatingMarginTTM")),
        "roa":                _to_float(h.get("ReturnOnAssetsTTM")),
        "roe":                _to_float(h.get("ReturnOnEquityTTM")),
        "revenue_ttm":        _to_float(h.get("RevenueTTM")),
        "gross_profit_ttm":   _to_float(h.get("GrossProfitTTM")),
        "diluted_eps_ttm":    _to_float(h.get("DilutedEpsTTM")),
        "rev_growth_yoy":     _to_float(h.get("QuarterlyRevenueGrowthYOY")),
        "eps_growth_yoy":     _to_float(h.get("QuarterlyEarningsGrowthYOY")),

        "trailing_pe":        _to_float(v.get("TrailingPE")),
        "forward_pe":         _to_float(v.get("ForwardPE")),
        "ps_ttm":             _to_float(v.get("PriceSalesTTM")),
        "pb_mrq":             _to_float(v.get("PriceBookMRQ")),
        "enterprise_value":   _to_float(v.get("EnterpriseValue")),
        "ev_revenue":         _to_float(v.get("EnterpriseValueRevenue")),
        "ev_ebitda":          _to_float(v.get("EnterpriseValueEbitda")),

        "shares_outstanding": _to_float(s.get("SharesOutstanding")),
        "beta":               _to_float(t.get("Beta")),
    }
    return out


# ---------------------------------------------------------------------------
# Point-in-time TTM rebuild
# ---------------------------------------------------------------------------

def _flatten_statement(stmt: Optional[Dict[str, Any]], frequency: str = "quarterly") -> pd.DataFrame:
    """Convert any of the on-disk Financials statement shapes into a tidy frame.

    Two schemas are supported:

      Schema A (full history). The statement dict has a `quarterly` (or
        `yearly`) key whose value is `{date_str: {field: value, ...}, ...}`.
        Used by ~10% of the R1000 fundamentals files.

      Schema B (recent slice). The statement dict has flat top-level keys
        `quarterly_last_0`, `quarterly_last_1`, ... and `yearly_last_0..N`,
        each mapping to a SINGLE period's record. Used by ~84% of the R1000
        fundamentals files.

    A `currency_symbol` key may sit beside the records and is ignored. The
    function returns the four available periods (or fewer if the file is
    short) sorted by period_end with an `available_date` column equal to
    `filing_date` or `period_end + 45 days` as a conservative fallback.
    """
    if not stmt:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    def _push_record(rec: Optional[Dict[str, Any]], fallback_period_end: Any = None):
        if not isinstance(rec, dict):
            return
        period_end = pd.to_datetime(rec.get("date") or fallback_period_end, errors="coerce")
        if pd.isna(period_end):
            return
        filing_date = pd.to_datetime(rec.get("filing_date"), errors="coerce")
        flat: Dict[str, Any] = {"period_end": period_end, "filing_date": filing_date}
        for fk, val in rec.items():
            if fk in ("date", "filing_date", "currency_symbol"):
                continue
            flat[fk] = _to_float(val) if isinstance(val, (int, float, str)) else val
        rows.append(flat)

    # Schema A: stmt["quarterly"] (or "yearly") is dict-of-dicts.
    primary_node = stmt.get(frequency)
    if isinstance(primary_node, dict) and primary_node:
        for ds, rec in primary_node.items():
            if ds == "currency_symbol":
                continue
            _push_record(rec, fallback_period_end=ds)

    # Schema B: top-level "quarterly_last_*" / "yearly_last_*" keys.
    prefix = f"{frequency}_last_"
    for k, rec in stmt.items():
        if isinstance(k, str) and k.startswith(prefix):
            _push_record(rec)

    if not rows:
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .drop_duplicates(subset=["period_end"])
            .sort_values("period_end")
            .reset_index(drop=True))
    df["available_date"] = df["filing_date"].fillna(df["period_end"] + pd.Timedelta(days=45))
    return df


def _statement_frame(node: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Backward-compatible alias used by older callers that pass the
    quarterly node directly (Schema A only). New code should call
    `_flatten_statement` with the whole statement dict."""
    if not node:
        return pd.DataFrame()
    return _flatten_statement({"quarterly": node}, frequency="quarterly")


def _ttm_sum(df: pd.DataFrame, field: str, asof: pd.Timestamp, q: int = 4) -> float:
    if df.empty or field not in df.columns:
        return float("nan")
    sub = df[df["available_date"] <= asof].tail(q)
    if len(sub) < q:
        return float("nan")
    vals = pd.to_numeric(sub[field], errors="coerce")
    if vals.isna().any():
        return float("nan")
    return float(vals.sum())


def _last_value(df: pd.DataFrame, field: str, asof: pd.Timestamp) -> float:
    if df.empty or field not in df.columns:
        return float("nan")
    sub = df[df["available_date"] <= asof]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub[field].iloc[-1], errors="coerce"))


def _avg_last_n(df: pd.DataFrame, field: str, asof: pd.Timestamp, n: int = 4) -> float:
    if df.empty or field not in df.columns:
        return float("nan")
    sub = df[df["available_date"] <= asof].tail(n)
    if sub.empty:
        return float("nan")
    vals = pd.to_numeric(sub[field], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def _shares_asof(blob: Dict[str, Any], asof: pd.Timestamp) -> float:
    quarters = (blob.get("outstandingShares") or {}).get("quarterly") or {}
    if not quarters:
        return float("nan")
    rows = []
    for _, rec in quarters.items():
        d = pd.to_datetime(rec.get("dateFormatted") or rec.get("date"), errors="coerce")
        sh = _to_float(rec.get("shares"))
        if pd.notna(d) and np.isfinite(sh):
            rows.append((d, sh))
    if not rows:
        return float("nan")
    df = pd.DataFrame(rows, columns=["date", "shares"]).sort_values("date")
    sub = df[df["date"] <= asof]
    if sub.empty:
        return float("nan")
    return float(sub["shares"].iloc[-1])


def point_in_time_features(
    fund_dir: str | Path,
    ticker: str,
    asof: pd.Timestamp,
    *,
    last_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Rebuild TTM ratios using only filings available on/before `asof`.

    `last_price` is the as-of close; passing it lets us compute pe, pb, ps
    even when the JSON's stale Highlights would be wrong. Without it we fall
    back to the JSON's MarketCapitalization (which is current, may overstate
    history) — this is logged as `_market_cap_is_current=True`.
    """
    p = _path_for(Path(fund_dir), ticker)
    if p is None:
        return {}
    blob = _load_json(p)
    asof = pd.Timestamp(asof).normalize()

    fin = blob.get("Financials") or {}
    inc_stmt = fin.get("Income_Statement") or {}
    bal_stmt = fin.get("Balance_Sheet")    or {}
    cf_stmt  = fin.get("Cash_Flow")        or {}

    inc_q = _flatten_statement(inc_stmt, "quarterly")
    bal_q = _flatten_statement(bal_stmt, "quarterly")
    cf_q  = _flatten_statement(cf_stmt,  "quarterly")

    rev_ttm    = _ttm_sum(inc_q, "totalRevenue", asof)
    gp_ttm     = _ttm_sum(inc_q, "grossProfit", asof)
    op_ttm     = _ttm_sum(inc_q, "operatingIncome", asof)
    ni_ttm     = _ttm_sum(inc_q, "netIncome", asof)
    ebitda_ttm = _ttm_sum(inc_q, "ebitda", asof)

    cfo_ttm    = _ttm_sum(cf_q, "totalCashFromOperatingActivities", asof)
    capex_ttm  = _ttm_sum(cf_q, "capitalExpenditures", asof)
    fcf_ttm    = float(cfo_ttm + capex_ttm) if _is_finite(cfo_ttm) and _is_finite(capex_ttm) else float("nan")

    # ---- Yearly fallback. Schema B carries only the 4 most-recent quarters,
    # so when the as-of date is far enough in the past that those 4 are all
    # post-asof, the quarterly TTM is NaN. The yearly_last_* slice still
    # covers the historical period; use the most-recent yearly available_date
    # ≤ asof as a coarser fallback.
    if not _is_finite(rev_ttm):
        inc_y = _flatten_statement(inc_stmt, "yearly")
        rev_ttm    = _last_value(inc_y, "totalRevenue", asof) if not _is_finite(rev_ttm) else rev_ttm
        gp_ttm     = _last_value(inc_y, "grossProfit", asof) if not _is_finite(gp_ttm) else gp_ttm
        op_ttm     = _last_value(inc_y, "operatingIncome", asof) if not _is_finite(op_ttm) else op_ttm
        ni_ttm     = _last_value(inc_y, "netIncome", asof) if not _is_finite(ni_ttm) else ni_ttm
        ebitda_ttm = _last_value(inc_y, "ebitda", asof) if not _is_finite(ebitda_ttm) else ebitda_ttm
    if not _is_finite(fcf_ttm):
        cf_y = _flatten_statement(cf_stmt, "yearly")
        cfo_y   = _last_value(cf_y, "totalCashFromOperatingActivities", asof)
        capex_y = _last_value(cf_y, "capitalExpenditures", asof)
        if _is_finite(cfo_y) and _is_finite(capex_y):
            fcf_ttm = float(cfo_y + capex_y)

    equity_last = _last_value(bal_q, "totalStockholderEquity", asof)
    if not _is_finite(equity_last):
        equity_last = _last_value(bal_q, "totalShareholderEquity", asof)
    equity_avg4 = _avg_last_n(bal_q, "totalStockholderEquity", asof, 4)
    if not _is_finite(equity_avg4):
        equity_avg4 = equity_last

    debt_last = _last_value(bal_q, "totalDebt", asof)
    if not _is_finite(debt_last):
        debt_last = _last_value(bal_q, "shortLongTermDebtTotal", asof)
        if not _is_finite(debt_last):
            debt_last = _last_value(bal_q, "longTermDebt", asof)

    cash_last = _last_value(bal_q, "cashAndCashEquivalents", asof)
    if not _is_finite(cash_last):
        cash_last = _last_value(bal_q, "cash", asof)

    # Balance-sheet yearly fallback (same rationale as the income statement).
    if not _is_finite(equity_last) or not _is_finite(debt_last) or not _is_finite(cash_last):
        bal_y = _flatten_statement(bal_stmt, "yearly")
        if not _is_finite(equity_last):
            equity_last = _last_value(bal_y, "totalStockholderEquity", asof)
            if not _is_finite(equity_last):
                equity_last = _last_value(bal_y, "totalShareholderEquity", asof)
            if not _is_finite(equity_avg4):
                equity_avg4 = equity_last
        if not _is_finite(debt_last):
            debt_last = _last_value(bal_y, "totalDebt", asof)
            if not _is_finite(debt_last):
                debt_last = _last_value(bal_y, "shortLongTermDebtTotal", asof)
                if not _is_finite(debt_last):
                    debt_last = _last_value(bal_y, "longTermDebt", asof)
        if not _is_finite(cash_last):
            cash_last = _last_value(bal_y, "cashAndCashEquivalents", asof)
            if not _is_finite(cash_last):
                cash_last = _last_value(bal_y, "cash", asof)

    shares = _shares_asof(blob, asof)
    market_cap_current = _to_float((blob.get("Highlights") or {}).get("MarketCapitalization"))
    market_cap = float(last_price) * shares if (last_price is not None and _is_finite(last_price) and _is_finite(shares)) else market_cap_current
    market_cap_is_current = (last_price is None or not _is_finite(last_price) or not _is_finite(shares))

    def _safe_div(a: float, b: float) -> float:
        return float(a / b) if _is_finite(a) and _is_finite(b) and b != 0 else float("nan")

    out: Dict[str, Any] = {
        "ticker":               ticker.upper(),
        "sector":               (blob.get("General") or {}).get("GicSector"),
        "asof":                 asof,

        "revenue_ttm":          rev_ttm,
        "gross_profit_ttm":     gp_ttm,
        "operating_income_ttm": op_ttm,
        "net_income_ttm":       ni_ttm,
        "ebitda_ttm":           ebitda_ttm,
        "fcf_ttm":              fcf_ttm,
        "equity":               equity_last,
        "equity_avg4":          equity_avg4,
        "debt":                 debt_last,
        "cash":                 cash_last,
        "shares_outstanding":   shares,
        "market_cap":           market_cap,
        "_market_cap_is_current": market_cap_is_current,

        "gross_margin":         _safe_div(gp_ttm, rev_ttm),
        "operating_margin":     _safe_div(op_ttm, rev_ttm),
        "net_margin":           _safe_div(ni_ttm, rev_ttm),
        "fcf_margin":           _safe_div(fcf_ttm, rev_ttm),
        "roe":                  _safe_div(ni_ttm, equity_avg4),
        "roa":                  _safe_div(ni_ttm, equity_last) if not _is_finite(equity_avg4) else _safe_div(ni_ttm, equity_avg4),
        "debt_to_equity":       _safe_div(debt_last, equity_last),

        "pe":                   _safe_div(market_cap, ni_ttm),
        "pb":                   _safe_div(market_cap, equity_last),
        "ps":                   _safe_div(market_cap, rev_ttm),
        "ev":                   (market_cap + (debt_last or 0.0) - (cash_last or 0.0))
                                if _is_finite(market_cap) and _is_finite(debt_last) and _is_finite(cash_last)
                                else float("nan"),
        "fcf_yield":            _safe_div(fcf_ttm, market_cap),
    }
    return out


# ---------------------------------------------------------------------------
# Point-in-time earnings — PEAD (post-earnings-announcement-drift) catalyst
# ---------------------------------------------------------------------------

def point_in_time_earnings(
    fund_dir: str | Path,
    ticker: str,
    asof: pd.Timestamp,
    *,
    surprise_window: int = 4,
) -> Dict[str, Any]:
    """Extract point-in-time earnings-surprise / EPS-trend features as of `asof`.

    Uses `Earnings.History`, filtered to records whose **announcement date**
    (`reportDate`, fallback `date`) is on/before `asof` — so there is no
    look-ahead. Returns:

      surprise_last   : most-recent epsActual-vs-estimate surprise % (PEAD)
      surprise_avg    : mean surprise % over the last `surprise_window` reports
      surprise_streak : count of consecutive recent positive surprises
      eps_ttm_growth  : TTM reported EPS vs prior TTM (4q sum / prior-4q sum − 1)
      n_reports       : number of usable point-in-time reports

    All values are NaN when insufficient point-in-time data exists — the
    caller fills/abstains so a missing signal never silently scores as 0.
    """
    p = _path_for(Path(fund_dir), ticker)
    empty = {"ticker": ticker.upper(), "surprise_last": float("nan"),
             "surprise_avg": float("nan"), "surprise_streak": float("nan"),
             "eps_ttm_growth": float("nan"), "n_reports": 0}
    if p is None:
        return empty
    blob = _load_json(p)
    asof = pd.Timestamp(asof).normalize()

    hist = (blob.get("Earnings") or {}).get("History") or {}
    rows: List[Dict[str, Any]] = []
    for rec in hist.values():
        if not isinstance(rec, dict):
            continue
        avail = pd.to_datetime(rec.get("reportDate") or rec.get("date"), errors="coerce")
        actual = _to_float(rec.get("epsActual"))
        if pd.isna(avail) or avail > asof or not _is_finite(actual):
            continue
        rows.append({
            "avail": avail,
            "epsActual": actual,
            "surprise": _to_float(rec.get("surprisePercent")),
        })
    if not rows:
        return empty

    df = pd.DataFrame(rows).sort_values("avail").reset_index(drop=True)

    surp = pd.to_numeric(df["surprise"], errors="coerce").dropna()
    surprise_last = float(surp.iloc[-1]) if len(surp) else float("nan")
    surprise_avg = float(surp.tail(surprise_window).mean()) if len(surp) else float("nan")

    streak = 0
    for v in reversed(surp.tolist()):
        if v > 0:
            streak += 1
        else:
            break
    surprise_streak = float(streak) if len(surp) else float("nan")

    eps = pd.to_numeric(df["epsActual"], errors="coerce").dropna()
    eps_ttm_growth = float("nan")
    if len(eps) >= 8:
        ttm = float(eps.iloc[-4:].sum())
        prior = float(eps.iloc[-8:-4].sum())
        if _is_finite(ttm) and _is_finite(prior) and prior > 0:
            eps_ttm_growth = ttm / prior - 1.0

    return {
        "ticker": ticker.upper(),
        "surprise_last": surprise_last,
        "surprise_avg": surprise_avg,
        "surprise_streak": surprise_streak,
        "eps_ttm_growth": eps_ttm_growth,
        "n_reports": int(len(df)),
    }
