"""
Macro panel loader for FRED CSVs.

Each CSV in `categorized/macro/` is exported as:
  ,0
  YYYY-MM-DD,value
  ...

i.e. a two-column file with an unnamed (index) column and a value column
labelled `0`. `read_csv(..., index_col=0, header=0)` recovers the date-indexed
series properly.

We expose a normalized panel with all known FRED series side-by-side, plus
two derivatives commonly used downstream (CPI YoY and FedFunds change).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# Known FRED files in the macro directory. Add more as needed.
KNOWN_SERIES: List[str] = [
    "VIXCLS",     # CBOE VIX (daily)
    "T10Y2Y",     # 10Y-2Y spread (daily)
    "FEDFUNDS",   # Fed Funds rate (monthly)
    "TB3MS",      # 3-month T-bill (monthly)
    "CPIAUCSL",   # CPI level (monthly)
    "UNRATE",     # Unemployment (monthly)
    "GDPC1",      # Real GDP (quarterly)
]


def _read_one(path: Path) -> pd.Series:
    """Read a FRED CSV → date-indexed float series named after the file stem."""
    df = pd.read_csv(path, index_col=0, header=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    if df.shape[1] < 1:
        return pd.Series(dtype=float, name=path.stem)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.index.name = "date"
    s.name = path.stem.upper()
    return s.dropna().sort_index()


def load_macro_panel(macro_dir: str | Path) -> pd.DataFrame:
    """Load all known FRED series into a unified panel.

    Each series is preserved at its native frequency in the wide DataFrame;
    we do **not** forward-fill across rows here. Derived series (YoY, ΔN
    months) are computed *on each native series* before joining so that
    differences span the correct calendar interval.
    """
    md = Path(macro_dir)
    if not md.is_dir():
        raise FileNotFoundError(f"macro_dir not found: {md}")
    series_map: Dict[str, pd.Series] = {}
    for name in KNOWN_SERIES:
        p = md / f"{name}.csv"
        if not p.is_file():
            continue
        try:
            s = _read_one(p)
        except Exception:
            continue
        if not s.empty:
            series_map[name] = s
    if not series_map:
        return pd.DataFrame()

    # Derived series — computed on the native series (so .diff(3) on a
    # monthly series like FEDFUNDS spans 3 months, not 3 panel rows).
    if "CPIAUCSL" in series_map:
        series_map["CPI_YOY"] = series_map["CPIAUCSL"].pct_change(12, fill_method=None)
    if "FEDFUNDS" in series_map:
        series_map["FEDFUNDS_DIFF_3M"] = series_map["FEDFUNDS"].diff(3)
    if "UNRATE" in series_map:
        series_map["UNRATE_DIFF_3M"] = series_map["UNRATE"].diff(3)

    panel = pd.concat(series_map.values(), axis=1).sort_index()
    panel.columns = list(series_map.keys())
    return panel


def asof_macro(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Return the last-observed value of every macro series on/before `asof`.

    Each series is forward-filled *only up to* `asof`, so we don't leak data
    that arrived after the rebalance date.
    """
    if panel.empty:
        return pd.Series(dtype=float)
    asof = pd.Timestamp(asof).normalize()
    sub = panel.loc[panel.index <= asof]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.ffill().iloc[-1]
