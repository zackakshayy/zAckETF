"""
Sentiment score using the **pre-computed** polarity field on each article.

We do not re-run VADER/FinBERT on title+content — the news data was already
scored by the vendor and consistently re-scoring it would only add noise.

Per-ticker features are produced by `data.news.aggregate_sentiment_window`
and consumed here cross-sectionally. The cross-section is sector-z scored.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def _winsorize_z(series: pd.Series, *, k: float = 3.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    mu = float(s.mean(skipna=True))
    sd = float(s.std(ddof=0, skipna=True))
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return ((s.clip(lower=mu - k * sd, upper=mu + k * sd) - mu) / sd)


def _z_by_sector(values: pd.Series, sectors: pd.Series, *, k: float = 3.0) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for sec, idx in sectors.groupby(sectors).groups.items():
        sub = values.loc[idx]
        if sub.dropna().shape[0] >= 3:
            out.loc[idx] = _winsorize_z(sub, k=k).values
    missing = out.isna() & values.notna()
    if missing.any():
        out.loc[missing] = _winsorize_z(values.loc[missing], k=k).values
    return out


def compute_sentiment_panel(
    feature_rows: Iterable[Dict[str, object]],
    sector_map: Dict[str, str],
    *,
    winsor_k: float = 3.0,
    polarity_w: float = 0.65,
    slope_w: float = 0.25,
    coverage_w: float = 0.10,
) -> pd.DataFrame:
    """Cross-sectional sentiment panel.

    `feature_rows` rows must include: ticker, polarity_decay, polarity_slope,
    article_count_log. Tickers with `article_count < min_articles` arrive
    with NaN polarity_decay and end up with score 0 (neutral).
    """
    df = pd.DataFrame(list(feature_rows))
    if df.empty:
        return pd.DataFrame(columns=["sector", "sentiment_score"])
    df = df.set_index("ticker")
    sectors = pd.Series({t: sector_map.get(t, "Unknown") for t in df.index}, name="sector")

    polarity_z = _z_by_sector(df.get("polarity_decay", pd.Series(np.nan, index=df.index)), sectors, k=winsor_k).fillna(0.0)
    slope_z    = _z_by_sector(df.get("polarity_slope", pd.Series(np.nan, index=df.index)), sectors, k=winsor_k).fillna(0.0)
    cov_z      = _z_by_sector(df.get("article_count_log", pd.Series(0.0, index=df.index)), sectors, k=winsor_k).fillna(0.0)

    score = (
        float(polarity_w) * polarity_z
        + float(slope_w)  * slope_z
        + float(coverage_w) * cov_z
    )

    out = pd.DataFrame({
        "sector":          sectors,
        "polarity_z":      polarity_z,
        "slope_z":         slope_z,
        "coverage_z":      cov_z,
        "sentiment_score": _z_by_sector(score, sectors, k=winsor_k).fillna(0.0),
    })
    return out
