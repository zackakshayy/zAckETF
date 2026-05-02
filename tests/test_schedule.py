"""Rebalance-schedule tests."""

from __future__ import annotations

import pandas as pd


def test_schedule_emits_one_event_per_month():
    from project_alpha.backtest.schedule import build_schedule
    cal = pd.bdate_range("2020-01-01", "2020-12-31")
    events = build_schedule(
        cal, start=cal[0], end=cal[-1],
        quarterly_fundamental_months=[2, 5, 8, 11],
        semiannual_reconstitution_months=[6, 12],
    )
    months = {(e.date.year, e.date.month) for e in events}
    assert len(months) == 12


def test_schedule_flags_recon_and_fundamental():
    from project_alpha.backtest.schedule import build_schedule
    cal = pd.bdate_range("2020-01-01", "2020-12-31")
    events = build_schedule(
        cal, start=cal[0], end=cal[-1],
        quarterly_fundamental_months=[2, 5, 8, 11],
        semiannual_reconstitution_months=[6, 12],
    )
    by_month = {e.date.month: e for e in events}
    assert by_month[6].reconstitute is True
    assert by_month[12].reconstitute is True
    assert by_month[2].reconstitute is False
    assert by_month[6].refresh_fundamental is True
    assert by_month[5].refresh_fundamental is True
    assert by_month[7].refresh_fundamental is False
    # Tech/sentiment/macro refresh every event
    assert all(e.refresh_technical for e in events)
    assert all(e.refresh_sentiment for e in events)
    assert all(e.refresh_macro for e in events)


def test_schedule_anchors_to_last_business_day():
    from project_alpha.backtest.schedule import build_schedule
    cal = pd.bdate_range("2020-01-01", "2020-12-31")
    events = build_schedule(
        cal, start=cal[0], end=cal[-1],
        quarterly_fundamental_months=[2, 5, 8, 11],
        semiannual_reconstitution_months=[6, 12],
    )
    # Dec 31 2020 was a Thursday → last bday of Dec 2020.
    dec = [e for e in events if e.date.year == 2020 and e.date.month == 12][0]
    assert dec.date == pd.Timestamp("2020-12-31")
    # June 30 2020 = Tuesday, last business day.
    jun = [e for e in events if e.date.year == 2020 and e.date.month == 6][0]
    assert jun.date == pd.Timestamp("2020-06-30")
