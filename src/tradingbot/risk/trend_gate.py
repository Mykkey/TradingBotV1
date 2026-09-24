# trend_gate.py
# Market-timing gate: be invested only while the market (SPY) closed above its
# N-day average. Decided once per week from closes before that week's Monday,
# so the answer is the same all week and survives restarts.
#
# Research (2024-2026): cost ~2%/yr of return but roughly halved the worst
# drawdowns (2025: -10% vs -20%), switching only ~7 times in 2.7 years.
import logging
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd


log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def daily_closes_from_minutes(bars):
    """Last close of each trading day (index: ET dates) from minute bars."""
    et = bars.index.tz_convert(ET)
    return bars["close"].groupby(et.date).last()


class TrendGate:

    def __init__(self, daily_closes, days=100):
        """daily_closes: Series indexed by ET date (datetime.date)."""
        self.closes = pd.Series(daily_closes).sort_index()
        self.days = days
        self.cache = {}

    def update_closes(self, daily_closes):
        self.closes = pd.Series(daily_closes).sort_index()
        self.cache = {}

    def is_open(self, now):
        today = now.astimezone(ET).date()
        monday = today - timedelta(days=today.weekday())

        if monday not in self.cache:
            past = self.closes[self.closes.index < monday]

            if len(past) < self.days:
                log.warning("Trend gate: only %d daily closes, need %d; staying invested",
                            len(past), self.days)
                self.cache[monday] = True
            else:
                average = past.iloc[-self.days:].mean()
                self.cache[monday] = bool(past.iloc[-1] > average)
                log.info("Trend gate for week of %s: %s (last close %.2f vs %d-day avg %.2f)",
                         monday, "INVESTED" if self.cache[monday] else "CASH",
                         past.iloc[-1], self.days, average)

        return self.cache[monday]
