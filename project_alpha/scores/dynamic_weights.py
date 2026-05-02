"""
IC-weighted dynamic pillar blending (Phase A1).

At each rebalance we maintain a rolling history of (pillar score panel, prices)
keyed by as-of date. When a new event arrives, we use realized returns from the
*previous* event to update the trailing IC for each fact-based pillar
(technical / fundamental / sentiment), then return regime-aware pillar weights:

    raw_p   = max(floor, max(0, mean trailing IC_p))
    norm_p  = raw_p / Σ raw_q                       (over T/F/S only)
    weight  = norm_p · (1 - macro_weight)

`macro` weight stays fixed at the regime-tilted base value because the macro
pillar is a per-sector tilt, not a per-stock score — IC of a sector-broadcast
value vs. forward stock returns is not informative.

During the warmup period (fewer than `lookback_events` of IC history) we fall
back to the regime-tilted base weights.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


PILLARS = ("technical", "fundamental", "sentiment", "lowvol")


def _spearman_ic(scores: pd.Series, returns: pd.Series, *, min_obs: int = 30) -> float:
    """Cross-sectional Spearman rank correlation between scores and forward returns."""
    s = pd.to_numeric(scores, errors="coerce")
    r = pd.to_numeric(returns, errors="coerce")
    df = pd.concat([s, r], axis=1, join="inner").dropna()
    if len(df) < min_obs:
        return float("nan")
    sd_s = df.iloc[:, 0].std(ddof=0)
    sd_r = df.iloc[:, 1].std(ddof=0)
    if sd_s == 0 or sd_r == 0 or not np.isfinite(sd_s) or not np.isfinite(sd_r):
        return float("nan")
    rs = df.iloc[:, 0].rank()
    rr = df.iloc[:, 1].rank()
    return float(rs.corr(rr))


class PillarICTracker:
    """Stateful tracker that produces IC-blended pillar weights at each rebalance.

    Usage:
        tracker = PillarICTracker(lookback_events=12, floor=0.05, cap=0.60)
        for asof, score_panel, prices_at_t, fixed_weights in events:
            w = tracker.update_and_weights(asof, score_panel, prices_at_t, fixed_weights)
    """

    def __init__(
        self,
        *,
        lookback_events: int = 12,
        floor: float = 0.05,
        cap: float = 0.60,
        min_obs_for_ic: int = 30,
        exclude_pillars: Optional[Iterable[str]] = None,
    ) -> None:
        self.lookback_events = int(lookback_events)
        self.floor = float(floor)
        self.cap = float(cap)
        self.min_obs_for_ic = int(min_obs_for_ic)
        self.exclude_pillars = set(exclude_pillars or [])
        self._history: List[Dict] = []   # each: {asof, panel, prices}
        self._ic_log: List[Dict] = []    # each: {asof, technical, fundamental, sentiment, n_obs}

    @property
    def n_ic_observations(self) -> int:
        return len([row for row in self._ic_log
                    if any(np.isfinite(row.get(p, np.nan)) for p in PILLARS)])

    def update_and_weights(
        self,
        asof: pd.Timestamp,
        score_panel: pd.DataFrame,
        prices_at_t: pd.Series,
        fixed_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute and return pillar weights for `asof`.

        Args:
            asof: rebalance date
            score_panel: index=ticker, columns include `<pillar>_score` for each fact pillar
            prices_at_t: index=ticker, last close at `asof`
            fixed_weights: regime-tilted base weights {technical, fundamental, sentiment, macro}.
                Used for warmup, and `macro` weight is preserved verbatim.
        """
        # Step 1: realize the *previous* event's IC using current prices.
        if self._history:
            self._record_realized_ic(prices_at_t)

        # Step 2: append current event so it can be evaluated next time.
        self._history.append({
            "asof": pd.Timestamp(asof),
            "panel": score_panel.copy(),
            "prices": pd.to_numeric(prices_at_t, errors="coerce").dropna().copy(),
        })
        # Keep history bounded — we only need the previous event's panel+prices.
        if len(self._history) > 2:
            self._history.pop(0)

        # Step 3: compute weights from the trailing IC log.
        return self._weights_from_ic(fixed_weights)

    def _record_realized_ic(self, prices_at_t: pd.Series) -> None:
        prev = self._history[-1]
        prev_prices = prev["prices"]
        cur_prices = pd.to_numeric(prices_at_t, errors="coerce").dropna()
        common = prev_prices.index.intersection(cur_prices.index)
        if len(common) < self.min_obs_for_ic:
            self._ic_log.append({"asof": prev["asof"], "n_obs": int(len(common)),
                                 **{p: float("nan") for p in PILLARS}})
            return
        fwd_ret = (cur_prices.loc[common] / prev_prices.loc[common] - 1.0)
        # Drop unrealistic returns (data glitches)
        fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan).dropna()
        fwd_ret = fwd_ret[(fwd_ret > -0.95) & (fwd_ret < 5.0)]

        row: Dict = {"asof": prev["asof"], "n_obs": int(len(fwd_ret))}
        panel = prev["panel"]
        for pillar in PILLARS:
            if pillar in self.exclude_pillars:
                row[pillar] = float("nan")
                continue
            col = f"{pillar}_score"
            if col not in panel.columns:
                row[pillar] = float("nan")
                continue
            scores = panel[col]
            ic = _spearman_ic(scores.reindex(fwd_ret.index), fwd_ret,
                              min_obs=self.min_obs_for_ic)
            row[pillar] = ic
        self._ic_log.append(row)

    def _weights_from_ic(self, fixed_weights: Dict[str, float]) -> Dict[str, float]:
        # Trailing window of IC observations
        window = [r for r in self._ic_log[-self.lookback_events:]]
        # Need at least `lookback_events` realized observations to switch on dynamic weighting.
        if len(window) < self.lookback_events:
            return dict(fixed_weights)

        # Pillars to include (exclude any specified in exclude_pillars)
        active_pillars = [p for p in PILLARS if p not in self.exclude_pillars]
        if not active_pillars:
            return dict(fixed_weights)

        # Mean IC per pillar (skip NaNs)
        ic_means: Dict[str, float] = {}
        for p in active_pillars:
            vals = [r.get(p) for r in window if r.get(p) is not None and np.isfinite(r.get(p))]
            ic_means[p] = float(np.mean(vals)) if vals else float("nan")

        # Build raw weights: max(floor, IC^+)
        raw: Dict[str, float] = {}
        for p in active_pillars:
            ic = ic_means.get(p, 0.0)
            if not np.isfinite(ic):
                ic = 0.0
            raw[p] = max(self.floor, max(0.0, ic))

        s = sum(raw.values())
        if s <= 0:
            return dict(fixed_weights)

        macro_w = float(fixed_weights.get("macro", 0.0))
        macro_w = float(np.clip(macro_w, 0.0, 1.0))
        non_macro_budget = 1.0 - macro_w

        # Allocate non_macro_budget across active pillars in proportion to raw IC, then enforce
        # the per-pillar absolute cap with a fixed-point: cap exceeding pillars at
        # `cap`, redistribute the remaining budget among the rest in raw-IC proportion.
        cap_p = float(self.cap)
        final: Dict[str, float] = {p: None for p in active_pillars}  # type: ignore[dict-item]
        remaining = non_macro_budget

        # Hard upper-bound: if cap × |active_pillars| < non_macro_budget, every pillar caps out.
        if cap_p * len(active_pillars) <= non_macro_budget + 1e-12:
            for p in active_pillars:
                final[p] = cap_p
        else:
            for _ in range(len(active_pillars) + 1):
                free = [p for p in active_pillars if final[p] is None]
                if not free:
                    break
                sub_raw = sum(raw[q] for q in free)
                if sub_raw <= 0:
                    share = remaining / max(1, len(free))
                    for p in free:
                        final[p] = min(share, cap_p)
                    break
                tentative = {p: raw[p] / sub_raw * remaining for p in free}
                over = [p for p in free if tentative[p] > cap_p]
                if not over:
                    for p in free:
                        final[p] = tentative[p]
                    break
                for p in over:
                    final[p] = cap_p
                    remaining -= cap_p

        out = {p: float(final.get(p) or 0.0) for p in PILLARS}
        out["macro"] = macro_w
        # Numeric safety: ensure sum ≈ 1
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """Full IC log for persistence."""
        if not self._ic_log:
            return pd.DataFrame(columns=["asof", "n_obs", *PILLARS])
        df = pd.DataFrame(self._ic_log)
        return df.sort_values("asof").reset_index(drop=True)
