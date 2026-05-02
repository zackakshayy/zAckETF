# project_alpha/utils/recs.py
from __future__ import annotations
import io
import re
from typing import Dict, Optional, Tuple
import requests
import pandas as pd

_COL_HOLDINGS_URL = "https://www.columbiathreadneedleus.com/cmg.svc/exportETFholdings"

def fetch_recs_holdings(cusip: str = "19761L706", timeout: int = 30) -> pd.DataFrame:
    params = {"cusip": cusip, "fileType": "csv", "fundGroupName": "ETF"}
    r = requests.get(_COL_HOLDINGS_URL, params=params, timeout=timeout)
    r.raise_for_status()
    text = r.text.replace("\ufeff", "")
    df = pd.read_csv(io.StringIO(text))

    # normalize headers
    cols = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in df.columns}
    def pick(keys):
        for k in keys:
            kk = re.sub(r"[^a-z0-9]+", "", k.lower())
            if kk in cols: return cols[kk]
        return None

    col_ticker = pick(["Ticker","Symbol"])
    col_name   = pick(["Description","Security Name","Name"])
    col_cusip  = pick(["CUSIP"])
    col_weight = pick(["Weight (%)","Weight","% of Net Assets"])
    col_sector = pick(["Sector","GICS Sector"])

    out = pd.DataFrame()
    if col_ticker: out["ticker"] = df[col_ticker].astype(str).str.strip().str.upper()
    if col_name:   out["name"]   = df[col_name].astype(str).str.strip()
    if col_cusip:  out["cusip"]  = df[col_cusip].astype(str).str.strip()
    if col_weight:
        w = pd.to_numeric(df[col_weight], errors="coerce")
        if (w.dropna() > 1.5).any(): w = w / 100.0
        out["weight"] = w
    if col_sector:
        out["sector"] = df[col_sector].astype(str).str.strip()

    return out.dropna(subset=["ticker","weight"]).reset_index(drop=True)

def holdings_has_sector(holdings: pd.DataFrame) -> Tuple[bool, list]:
    cols = [c.lower() for c in holdings.columns]
    has = any(c in cols for c in ["sector","gics sector","gics_sector"])
    return has, list(holdings.columns)

def sector_weights_from_holdings(holdings: pd.DataFrame, sector_map: Optional[Dict[str,str]] = None) -> pd.Series:
    h = holdings.copy()
    if "sector" not in [c.lower() for c in h.columns]:
        if sector_map is None:
            raise ValueError("Holdings have no sector column and no sector_map was provided.")
        h["sector"] = h["ticker"].map(sector_map)
    sw = h.groupby("sector")["weight"].sum().fillna(0.0)
    return sw / sw.sum()

def load_recs_prices_csv(path: str, use_column: str = "NAV") -> pd.Series:
    df = pd.read_csv(path)
    # find date & price columns flexibly
    def norm(x): return re.sub(r"[^a-z0-9]", "", str(x).lower())
    m = {norm(c): c for c in df.columns}
    date_col = next((c for k,c in m.items() if k in {"date","pricedate","navdate"}), None)
    if date_col is None:
        raise ValueError("Could not find a Date column in RECS prices CSV.")
    price_col = m.get(norm(use_column)) or m.get("close") or m.get("price") or m.get("nav")
    if price_col is None:
        raise ValueError(f"Could not find a price column like '{use_column}'/'Close' in RECS prices CSV.")

    s = pd.Series(pd.to_numeric(df[price_col], errors="coerce").values,
                  index=pd.to_datetime(df[date_col])).sort_index().dropna()
    return s.pct_change().dropna()
