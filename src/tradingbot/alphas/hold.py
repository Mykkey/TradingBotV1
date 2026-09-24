# hold.py
from datetime import timedelta

from tradingbot.core.framework import AlphaModel
from tradingbot.core.insight import Direction, Insight


class HoldAlpha(AlphaModel):
    """Always long every symbol in `symbols`.

    Research (2024-2026 minute data) showed nearly all of these stocks'
    returns arrive overnight, and intraday signals don't survive trading
    costs. Holding a diversified basket across days is what made money, so
    this alpha simply keeps each name in the portfolio; portfolio
    construction sizes it and the rebalance band keeps turnover near zero.
    """

    name = "hold"

    def __init__(self, symbols, period_min=24 * 60):
        self.symbols = set(symbols)
        self.period = timedelta(minutes=period_min)

    def update(self, now, history):
        return [
            Insight(symbol, Direction.UP, now, self.period, 1.0, source=self.name)
            for symbol in history
            if symbol in self.symbols
        ]
