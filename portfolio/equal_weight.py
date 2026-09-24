# equal_weight.py
from collections import defaultdict

from core.framework import PortfolioConstructionModel


def net_scores(insights):
    """Combine every active insight into one score per symbol."""
    scores = defaultdict(float)

    for insight in insights:
        scores[insight.symbol] += int(insight.direction) * insight.confidence

    return dict(scores)


class EqualWeightPortfolio(PortfolioConstructionModel):
    """Long-only: equal weight across every symbol with a positive net score."""

    def create_targets(self, now, insights, state):
        scores = net_scores(insights)
        longs = [symbol for symbol, score in scores.items() if score > 0]

        targets = {symbol: 0.0 for symbol in state["positions"]}

        for symbol in longs:
            targets[symbol] = 1.0 / len(longs)

        return targets
