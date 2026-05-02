"""
Three-cadence rebalance schedule.

A `ScheduleEvent` tells the orchestrator what needs to be refreshed at this
month-end:

  reconstitute (universe pulled from R1000)        — semi-annual months
  refresh_fundamental                              — quarterly months
  refresh_technical, refresh_sentiment, refresh_macro — every event (monthly)

We snap rule-driven dates to the actual trading calendar (so `12-31` becomes
`12-29` if that's the last business day).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScheduleEvent:
    date: pd.Timestamp
    reconstitute: bool
    refresh_fundamental: bool
    refresh_technical: bool
    refresh_sentiment: bool
    refresh_macro: bool

    def as_dict(self) -> dict:
        d = asdict(self)
        d["date"] = self.date.strftime("%Y-%m-%d")
        return d


def build_schedule(
    trading_dates: Sequence[pd.Timestamp],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    quarterly_fundamental_months: Iterable[int],
    semiannual_reconstitution_months: Iterable[int],
) -> List[ScheduleEvent]:
    """Produce one event per month-end in [start, end].

    Each event is anchored to the last trading day of that calendar month.
    """
    cal = pd.DatetimeIndex(sorted(set(pd.Timestamp(d).normalize() for d in trading_dates)))
    cal = cal[(cal >= pd.Timestamp(start).normalize()) & (cal <= pd.Timestamp(end).normalize())]
    if len(cal) == 0:
        return []

    fund_months = {int(m) for m in quarterly_fundamental_months}
    recon_months = {int(m) for m in semiannual_reconstitution_months}

    months = pd.PeriodIndex(cal, freq="M").unique()
    events: List[ScheduleEvent] = []
    for period in months:
        in_month = cal[(cal.year == period.year) & (cal.month == period.month)]
        if len(in_month) == 0:
            continue
        last_d = in_month[-1]
        events.append(ScheduleEvent(
            date=last_d,
            reconstitute=(int(last_d.month) in recon_months),
            refresh_fundamental=(int(last_d.month) in fund_months) or (int(last_d.month) in recon_months),
            refresh_technical=True,
            refresh_sentiment=True,
            refresh_macro=True,
        ))
    return events
