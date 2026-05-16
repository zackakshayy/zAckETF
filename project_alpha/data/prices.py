"""
EOD price loader and derived panels.

The prices_dir contains parquet files of three flavors:
  TICKER.US.eod.parquet         columns: date, open, high, low, close, adjusted_close, volume
  TICKER.US.dividends.parquet
  TICKER.US.splits.parquet

Returns are computed on `adjusted_close` (split- and dividend-adjusted).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _candidates(prices_dir: Path, ticker: str, kind: str) -> List[Path]:
    t = ticker.upper()
    return [
        prices_dir / f"{t}.US.{kind}.parquet",
        prices_dir / f"{t}.{kind}.parquet",
    ]


def _read_parquet(paths: Iterable[Path]) -> pd.DataFrame:
    for p in paths:
        if p.is_file():
            try:
                return pd.read_parquet(p)
            except Exception:
                continue
    return pd.DataFrame()


def load_eod(prices_dir: str | Path, ticker: str) -> pd.DataFrame:
    """Load OHLCV for `ticker`. Empty DataFrame if missing.

    Columns returned: date (Timestamp), open, high, low, close,
    adjusted_close, volume. `date` is naive (no tz).
    """
    df = _read_parquet(_candidates(Path(prices_dir), ticker, "eod"))
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    for c in ("open", "high", "low", "close", "adjusted_close", "volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def load_dividends(prices_dir: str | Path, ticker: str) -> pd.DataFrame:
    df = _read_parquet(_candidates(Path(prices_dir), ticker, "dividends"))
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=[c for c in ["date"] if c in df.columns])


def load_splits(prices_dir: str | Path, ticker: str) -> pd.DataFrame:
    df = _read_parquet(_candidates(Path(prices_dir), ticker, "splits"))
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    return df.dropna(subset=[c for c in ["date"] if c in df.columns])


def build_returns_panel(
    prices_dir: str | Path,
    tickers: Iterable[str],
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    use_adjusted: bool = True,
) -> pd.DataFrame:
    """Return wide DataFrame of daily simple returns indexed by date, columns=tickers.

    Missing tickers are silently skipped. Misaligned dates become NaN.
    """
    cols: Dict[str, pd.Series] = {}
    for t in tickers:
        eod = load_eod(prices_dir, t)
        if eod.empty:
            continue
        col = "adjusted_close" if (use_adjusted and "adjusted_close" in eod.columns) else "close"
        s = eod.set_index("date")[col].astype(float).pct_change()
        if start is not None:
            s = s.loc[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s.loc[s.index <= pd.Timestamp(end)]
        cols[t] = s.rename(t)
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols.values(), axis=1).sort_index()
    return df


def adv20_panel(
    prices_dir: str | Path,
    tickers: Iterable[str],
    *,
    window: int = 60,
) -> pd.DataFrame:
    """Average daily dollar volume (price * volume) panel."""
    cols: Dict[str, pd.Series] = {}
    for t in tickers:
        eod = load_eod(prices_dir, t)
        if eod.empty or "volume" not in eod.columns or "close" not in eod.columns:
            continue
        s = eod.set_index("date")
        dv = (s["close"].astype(float) * s["volume"].astype(float)).rolling(window, min_periods=10).mean()
        cols[t] = dv.rename(t)
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols.values(), axis=1).sort_index()


def last_price_series(prices_dir: str | Path, tickers: Iterable[str]) -> pd.DataFrame:
    """Wide panel of close prices keyed by date (used for as-of queries)."""
    cols: Dict[str, pd.Series] = {}
    for t in tickers:
        eod = load_eod(prices_dir, t)
        if eod.empty:
            continue
        cols[t] = eod.set_index("date")["close"].astype(float).rename(t)
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols.values(), axis=1).sort_index()
