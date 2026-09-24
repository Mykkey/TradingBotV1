# base.py
import math

from core.framework import ExecutionModel


def orders_for_targets(now, targets, state, prices, context, min_trade_value,
                       rebalance_band=0.0):
    """Turn target weights into whole-share buy/sell orders (long-only).
    Resizes of an open position smaller than `rebalance_band` (fraction of
    equity) are skipped to cut turnover; entries and full exits always trade."""
    orders = []

    for symbol, weight in targets.items():
        price = prices.get(symbol)

        if not price:
            continue

        held = state["positions"].get(symbol, 0)
        wanted = max(0, math.floor(max(weight, 0.0) * state["equity"] / price))
        delta = wanted - held

        if delta == 0:
            continue

        # Always allow a full exit; otherwise skip small rebalances.
        if wanted != 0:
            value = abs(delta) * price
            is_resize = held != 0

            if value < min_trade_value or (is_resize and value < rebalance_band * state["equity"]):
                continue

        insights = context.get(symbol, [])

        orders.append({
            "symbol": symbol,
            "side": "buy" if delta > 0 else "sell",
            "qty": abs(delta),
            "target_weight": round(weight, 4),
            "sources": "+".join(sorted({i.source for i in insights})),
            "confidence": round(max((i.confidence for i in insights), default=0.0), 3),
            "decided_at": now,
        })

    # Sells first so their cash is available for buys.
    orders.sort(key=lambda o: o["side"] != "sell")

    return orders


class TargetExecution(ExecutionModel):
    """Computes orders; subclasses decide how they are filled."""

    def __init__(self, min_trade_value=200, rebalance_band=0.0):
        self.min_trade_value = min_trade_value
        self.rebalance_band = rebalance_band

    def execute(self, now, targets, state, prices, context):
        orders = orders_for_targets(
            now, targets, state, prices, context, self.min_trade_value,
            self.rebalance_band,
        )
        return self.submit(orders)

    def submit(self, orders):
        return orders
