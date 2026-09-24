# momentum.py
from datetime import timedelta

from tradingbot.alphas.signals import momentum_score
from tradingbot.core.framework import AlphaModel
from tradingbot.core.insight import Direction, Insight


class MomentumAlpha(AlphaModel):
    """Intraday momentum: a strong move over `lookback_min` tends to continue."""

    name = "momentum"

    def __init__(self, lookback_min=30, threshold=0.002, period_min=15):
        self.lookback = lookback_min
        self.threshold = threshold
        self.period = timedelta(minutes=period_min)

    def update(self, now, history):
        insights = []

        for symbol, bars in history.items():

            if len(bars) <= self.lookback:
                continue

            window = bars.iloc[-(self.lookback + 1):]
            score = momentum_score(window, self.lookback, self.threshold).iloc[-1]

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
