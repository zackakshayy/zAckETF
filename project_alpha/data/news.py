"""
News loader and sentiment-window aggregator.

Each monthly file is a JSON list of articles with keys:
  date (ISO 8601 with offset), title, content, link, symbols, tags,
  sentiment {polarity, neg, neu, pos}.

We rely on the **pre-computed** sentiment.polarity rather than re-scoring
text — that is faster and consistent with what the data vendor produced.

`symbols` is a stringified Python list (vendor quirk); we parse it to count
how many tickers each article mentions, which we use as an inverse-relevance
weight (a story tagged with 50 symbols is less ticker-specific than one with
two).
"""

from __future__ import annotations

import ast
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _months_in_window(asof: pd.Timestamp, window_days: int) -> List[str]:
    start = (asof - pd.Timedelta(days=window_days)).to_pydatetime()
    end = asof.to_pydatetime()
    out: List[str] = []
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        out.append(f"{cur.year:04d}-{cur.month:02d}")
        cur = date(cur.year + (1 if cur.month == 12 else 0),
                   1 if cur.month == 12 else cur.month + 1,
                   1)
    return out


def list_monthly_files(news_dir: str | Path, ticker: str, months: Iterable[str]) -> List[Path]:
    nd = Path(news_dir)
    t = ticker.upper()
    out: List[Path] = []
    for m in months:
        for cand in (nd / f"{t}.US_{m}.news.json", nd / f"{t}_{m}.news.json"):
            if cand.is_file():
                out.append(cand)
                break
    return out


# ---------------------------------------------------------------------------
# Article parsing
# ---------------------------------------------------------------------------

def _parse_symbols(raw: Any) -> List[str]:
    """vendor quirk: `symbols` is stringified list. Be defensive."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            return [str(x) for x in v] if isinstance(v, list) else []
        except (ValueError, SyntaxError):
            return []
    return []


def _parse_sentiment(raw: Any) -> Optional[float]:
    if isinstance(raw, dict):
        return float(raw.get("polarity", float("nan"))) if raw.get("polarity") is not None else None
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            return float(v.get("polarity", float("nan"))) if isinstance(v, dict) else None
        except (ValueError, SyntaxError):
            return None
    return None


def _load_one(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return v
    return []


def load_news_window(
    news_dir: str | Path,
    ticker: str,
    asof: pd.Timestamp,
    *,
    window_days: int,
) -> pd.DataFrame:
    """Load and parse all articles in [asof - window_days, asof].

    Returns DataFrame with columns: date, polarity, n_symbols, ticker_in_symbols (bool).
    """
    asof = pd.Timestamp(asof).normalize()
    months = _months_in_window(asof, window_days)
    files = list_monthly_files(news_dir, ticker, months)
    if not files:
        return pd.DataFrame(columns=["date", "polarity", "n_symbols", "ticker_in_symbols"])

    t_us = f"{ticker.upper()}.US"
    rows: List[Dict[str, Any]] = []
    for p in files:
        for art in _load_one(p):
            d = pd.to_datetime(art.get("date"), errors="coerce")
            if pd.isna(d):
                continue
            d = d.tz_localize(None) if d.tzinfo is not None else d
            if d > asof or d < asof - pd.Timedelta(days=window_days):
                continue
            polarity = _parse_sentiment(art.get("sentiment"))
            if polarity is None or not np.isfinite(polarity):
                continue
            syms = _parse_symbols(art.get("symbols"))
            rows.append({
                "date": d,
                "polarity": float(polarity),
                "n_symbols": len(syms),
                "ticker_in_symbols": (t_us in syms) or (ticker.upper() in syms),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "polarity", "n_symbols", "ticker_in_symbols"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def aggregate_sentiment_window(
    df: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    half_life_days: float = 30.0,
    relevance_weight: str = "inverse_symbol_count",
    require_ticker_in_symbols: bool = True,
    min_articles: int = 3,
) -> Dict[str, float]:
    """Compute decay-weighted sentiment statistics from an article frame."""
    if df is None or df.empty:
        return {"polarity_decay": float("nan"), "polarity_slope": float("nan"),
                "article_count": 0.0, "article_count_log": 0.0}

    if require_ticker_in_symbols and "ticker_in_symbols" in df.columns:
        df = df[df["ticker_in_symbols"]]

    if df.empty or len(df) < min_articles:
        return {"polarity_decay": float("nan"), "polarity_slope": float("nan"),
                "article_count": float(len(df)), "article_count_log": float(np.log1p(len(df)))}

    asof = pd.Timestamp(asof).normalize()
    age_days = (asof - df["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    decay_w = np.exp(-np.log(2.0) * age_days / float(half_life_days))

    if relevance_weight == "inverse_symbol_count":
        rel_w = 1.0 / np.maximum(1.0, df["n_symbols"].to_numpy(dtype=float))
    else:
        rel_w = np.ones(len(df), dtype=float)

    weights = decay_w * rel_w
    polarity = df["polarity"].to_numpy(dtype=float)

    if weights.sum() <= 0 or not np.isfinite(weights.sum()):
        return {"polarity_decay": float("nan"), "polarity_slope": float("nan"),
                "article_count": float(len(df)), "article_count_log": float(np.log1p(len(df)))}

    polarity_decay = float(np.average(polarity, weights=weights))

    # Slope: 30d window vs 90d window
    age = age_days
    pol_30 = polarity[age <= 30.0]
    pol_90 = polarity[age <= 90.0]
    if len(pol_30) >= max(2, min_articles) and len(pol_90) >= max(2, min_articles):
        slope = float(np.mean(pol_30) - np.mean(pol_90))
    else:
        slope = float("nan")

    return {
        "polarity_decay":    polarity_decay,
        "polarity_slope":    slope,
        "article_count":     float(len(df)),
        "article_count_log": float(np.log1p(len(df))),
    }
