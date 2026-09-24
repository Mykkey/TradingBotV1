# mean_reversion.py
from datetime import timedelta

from alphas.signals import mean_reversion_score
from core.framework import AlphaModel
from core.insight import Direction, Insight


class VwapMeanReversionAlpha(AlphaModel):
    """Price stretched far from session VWAP tends to revert toward it."""

    name = "mean_reversion"

    def __init__(self, vwap_z_entry=2.0, lookback_min=60, period_min=15):
        self.entry = vwap_z_entry
        self.lookback = lookback_min
        self.period = timedelta(minutes=period_min)

    def update(self, now, history):
        insights = []

        for symbol, bars in history.items():

            if len(bars) < self.lookback:
                continue

            # Needs the whole session (history holds >= 1 day) for session VWAP.
            score = mean_reversion_score(bars, self.entry, self.lookback).iloc[-1]

            if score == 0:
                continue

            insights.append(Insight(
                symbol=symbol,
                direction=Direction.UP if score > 0 else Direction.DOWN,
                generated_at=now,
                period=self.period,
                confidence=abs(score),
                source=self.name,
            ))

        return insights
