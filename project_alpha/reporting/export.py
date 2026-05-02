"""
Project Alpha – Reporting Exports

Exports:
  • Per-rebalance scorebooks to Excel.
  • NAVs to CSV.
  • Performance summary (KPIs + $1M growth) to Excel.
"""

from __future__ import annotations

from typing import Dict
import os
import pandas as pd


def export_scores_xlsx(path: str, frames: Dict[str, pd.DataFrame]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as xw:
        for sheet, df in frames.items():
            if df is None:
                continue
            df.to_excel(xw, sheet_name=sheet[:31], index=False)


def export_navs_csv(path: str, navs: Dict[str, pd.Series]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.concat({k: v for k, v in navs.items() if v is not None and len(v) > 0}, axis=1)
    df.to_csv(path, index=True)


def export_performance_summary_xlsx(path: str,
                                    kpis_df: pd.DataFrame,
                                    growth_df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as xw:
        kpis_df.to_excel(xw, sheet_name="KPIs", index=False)
        growth_df.to_excel(xw, sheet_name="Growth_$1M", index=False)
