"""
Project Alpha – Reporting Visuals (matplotlib-only)

Generates:
  • Equity curves (Alpha vs. Benchmarks)
  • Drawdown curves (Alpha vs. Benchmarks)
All figures saved as high-DPI PNGs in charts_dir.

No seaborn dependency; clean aesthetic; resilient to missing series.
"""

from __future__ import annotations

import os
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _style():
    plt.rcParams.update({
        "figure.figsize": (10, 5.5),
        "figure.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "savefig.dpi": 200,
        "lines.linewidth": 2.2,
    })


def _safe_align(navs: Dict[str, pd.Series]) -> pd.DataFrame:
    frames = []
    for k, s in navs.items():
        if s is None or len(s) == 0:
            continue
        s = s.dropna().astype(float)
        if s.empty:
            continue
        frames.append(s.rename(k))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1).dropna(how="all")
    # Forward-fill small holes on benchmarks (yfinance quirks)
    df = df.ffill().dropna(how="any")
    return df


def plot_equity_curves(navs: Dict[str, pd.Series], charts_dir: str, filename: str = "equity_curves.png") -> Optional[str]:
    _style()
    df = _safe_align(navs)
    if df.empty:
        return None
    # Normalize to 1.0 at common start
    df = df / df.iloc[0]
    fig, ax = plt.subplots()
    for col in df.columns:
        ax.plot(df.index, df[col].values, label=col)
    ax.set_title("Equity Curves (Normalized to 1.0)")
    ax.set_ylabel("Growth (×)")
    ax.set_xlabel("Date")
    ax.legend(ncol=min(3, len(df.columns)))
    os.makedirs(charts_dir, exist_ok=True)
    out = os.path.join(charts_dir, filename)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _drawdown(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    s = s.dropna().astype(float)
    roll_max = s.cummax()
    dd = s / roll_max - 1.0
    return dd


def plot_drawdowns(navs: Dict[str, pd.Series], charts_dir: str, filename: str = "drawdowns.png") -> Optional[str]:
    _style()
    df = _safe_align(navs)
    if df.empty:
        return None
    # Convert to drawdowns per series
    dd = pd.DataFrame({k: _drawdown(df[k]) for k in df.columns})
    fig, ax = plt.subplots()
    for col in dd.columns:
        ax.plot(dd.index, dd[col].values, label=col)
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(ncol=min(3, len(dd.columns)))
    ax.set_ylim(dd.min().min() * 1.05, 0.01)
    os.makedirs(charts_dir, exist_ok=True)
    out = os.path.join(charts_dir, filename)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
