# test_trend_gate.py
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from tradingbot.risk import build_risk_model
from tradingbot.risk.trend_gate import TrendGate, daily_closes_from_minutes
from tests.conftest import make_bars


def closes(values, start=date(2025, 1, 1)):
    days = pd.bdate_range(start, periods=len(values)).date
    return pd.Series(values, index=days)


def at(d):
    return datetime(d.year, d.month, d.day, 15, 0, tzinfo=timezone.utc)


def test_gate_open_in_uptrend_closed_in_downtrend():
    up = closes(range(100, 220))
    down = closes(range(220, 100, -1))
    later = up.index[-1] + timedelta(days=7)

    assert TrendGate(up, days=100).is_open(at(later))
    assert not TrendGate(down, days=100).is_open(at(later))


def test_gate_decides_weekly_from_closes_before_monday():
    # Uptrend, then a crash starting on a Wednesday.
    values = list(range(100, 220)) + [50] * 10
    series = closes(values)
    crash_day = series.index[120]
    gate = TrendGate(series, days=100)

    monday = crash_day - timedelta(days=crash_day.weekday())
    # Same week as the crash: still decided by the previous Friday -> invested.
    assert gate.is_open(at(crash_day))
    # Following week sees the crash -> cash.
    assert not gate.is_open(at(monday + timedelta(days=7)))


def test_gate_stays_invested_without_enough_history():
    assert TrendGate(closes(range(100, 110)), days=100).is_open(at(date(2025, 2, 1)))


def test_risk_model_goes_to_cash_when_gate_closed(settings):
    settings["risk"]["trend_filter"] = {"enabled": True, "symbol": "SPY", "days": 100}
    settings["risk"]["flatten_at_close"] = False
    risk = build_risk_model(settings, closes(range(220, 100, -1)))

    state = {"equity": 100_000, "cash": 0, "positions": {"A": 10},
             "day_start_equity": 100_000, "minutes_to_close": 120}
    assert risk.manage_risk(at(date(2025, 9, 1)), {"A": 0.1}, state) == {"A": 0.0}


def test_daily_closes_from_minutes():
    bars = make_bars(days=3)
    daily = daily_closes_from_minutes(bars)

    assert len(daily) == 3
    assert daily.iloc[-1] == bars["close"].iloc[-1]
