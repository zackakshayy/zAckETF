"""
Universe loader for Russell 1000.

The R1000.xlsx workbook has annual snapshots on/around 06-30. Per the strategy
spec, both June and December rebalances of year Y use the Y-06-30 snapshot.

Columns we rely on:
  Date, Ticker, Port. Ending Weight, Asset Type (Client Definition/ FactSet),
  GICS Sector Extended.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "Date",
    "Ticker",
    "Port. Ending Weight",
    "GICS Sector Extended",
    "Asset Type (Client Definition/ FactSet)",
}


def load_russell_panel(xlsx_path: str | Path, sheet: str = "R1K") -> pd.DataFrame:
    """Load all R1000 snapshots into a long-form panel.

    Returns columns: snapshot_date (Timestamp), ticker, sector, weight (decimal), asset_type.
    """
    p = Path(xlsx_path)
    if not p.is_file():
        raise FileNotFoundError(f"R1000 workbook not found: {p}")

    df = pd.read_excel(p, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"R1000 sheet '{sheet}' missing columns: {sorted(missing)}")

    out = pd.DataFrame({
        "snapshot_date": pd.to_datetime(df["Date"], errors="coerce"),
        "ticker":        df["Ticker"].astype(str).str.strip().str.upper(),
        "sector":        df["GICS Sector Extended"].astype(str).str.strip(),
        "weight":        pd.to_numeric(df["Port. Ending Weight"], errors="coerce") / 100.0,
        "asset_type":    df["Asset Type (Client Definition/ FactSet)"].astype(str).str.strip(),
    })
    out = out.dropna(subset=["snapshot_date", "ticker", "sector", "weight"])
    out = out[out["ticker"].ne("") & out["sector"].ne("")]
    return out.sort_values(["snapshot_date", "ticker"]).reset_index(drop=True)


def _anchor_snapshot(
    panel: pd.DataFrame,
    target: pd.Timestamp,
    tolerance_days: int,
) -> Optional[pd.Timestamp]:
    """Find the snapshot date most appropriate for `target`.

    Rule: use the latest snapshot on-or-before `target`. If none on-or-before,
    fall back to the nearest within `tolerance_days`.
    """
    snaps = pd.DatetimeIndex(sorted(panel["snapshot_date"].unique()))
    if snaps.empty:
        return None
    earlier = snaps[snaps <= target]
    if len(earlier) > 0:
        return earlier[-1]
    diffs = np.abs((snaps - target).days)
    i = int(np.argmin(diffs))
    return snaps[i] if diffs[i] <= tolerance_days else None


def universe_for_date(
    panel: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    anchor_month_day: str = "06-30",
    tolerance_days: int = 14,
    exclude_asset_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return the universe that should be used for a rebalance at `asof`.

    Per spec: both Jun-Y and Dec-Y rebalances use the Y-Jun-30 snapshot.
    For dates after the latest available Jun snapshot, we fall back to that
    last snapshot (a known-and-logged limitation of the dataset).
    """
    if exclude_asset_types is None:
        exclude_asset_types = ["Cash", "Bond"]

    asof = pd.Timestamp(asof).normalize()
    year = asof.year
    if asof.month < int(anchor_month_day.split("-")[0]):
        year -= 1
    anchor_target = pd.Timestamp(date(year, *map(int, anchor_month_day.split("-"))))
    snap = _anchor_snapshot(panel, anchor_target, tolerance_days)
    if snap is None:
        snap = _anchor_snapshot(panel, asof, tolerance_days * 365)
    if snap is None:
        raise ValueError(f"No Russell snapshot found for asof={asof.date()}")

    sub = panel[panel["snapshot_date"] == snap].copy()
    if exclude_asset_types:
        sub = sub[~sub["asset_type"].isin(exclude_asset_types)]
    sub = sub.drop_duplicates(subset=["ticker"], keep="last")
    sub["weight"] = sub["weight"] / sub["weight"].sum()  # renormalize after exclusions
    sub.attrs["snapshot_date"] = snap
    return sub.reset_index(drop=True)


def sector_targets_for_date(universe: pd.DataFrame) -> pd.Series:
    """Aggregate constituent weights into sector targets (sums to 1.0)."""
    s = universe.groupby("sector")["weight"].sum()
    total = float(s.sum())
    if total <= 0:
        raise ValueError("Universe weights sum to zero; cannot compute sector targets.")
    return (s / total).sort_values(ascending=False)
