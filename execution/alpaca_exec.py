# alpaca_exec.py
# Sends orders to Alpaca paper trading and logs every fill.
import logging
import threading
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.stream import TradingStream

from execution.base import TargetExecution
from utils.market_data import API_KEY, SECRET_KEY


log = logging.getLogger(__name__)


def with_retries(fn, *args, attempts=3, **kwargs):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == attempts:
                raise
            log.warning("%s failed (attempt %d), retrying", fn.__name__, attempt, exc_info=True)
            time.sleep(2 ** attempt)


class AlpacaExecution(TargetExecution):

    def __init__(self, trade_logger, min_trade_value=200, rebalance_band=0.0):
        super().__init__(min_trade_value, rebalance_band)
        self.client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self.logger = trade_logger
        self.orders = {}       # client_order_id -> order metadata
        self.avg_cost = {}     # symbol -> average entry price, for realized P&L
        self.lock = threading.Lock()
        self.stream = None

    # ----- account state (Alpaca is the source of truth) -----

    def account_state(self):
        account = with_retries(self.client.get_account)
        positions = with_retries(self.client.get_all_positions)

        with self.lock:
            self.avg_cost = {p.symbol: float(p.avg_entry_price) for p in positions}

        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "last_equity": float(account.last_equity),
            "positions": {p.symbol: int(float(p.qty)) for p in positions},
            "market_values": {p.symbol: float(p.market_value) for p in positions},
        }

    def cancel_open_orders(self):
        with_retries(self.client.cancel_orders)

    def close_all(self):
        with_retries(self.client.close_all_positions, cancel_orders=True)

    # ----- order submission -----

    def submit(self, orders):
        # Cancel stale unfilled orders first so we don't double up.
        self.cancel_open_orders()

        submitted = []

        for order in orders:
            client_id = (
                f"tb-{order['symbol']}-{order['side']}-{order['decided_at']:%Y%m%d%H%M}"
            )
            request = MarketOrderRequest(
                symbol=order["symbol"],
                qty=order["qty"],
                side=OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_id,
            )

            with self.lock:
                self.orders[client_id] = order

            try:
                with_retries(self.client.submit_order, request)
                submitted.append(order)
                log.info("Submitted %s %s %s", order["side"], order["qty"], order["symbol"])
            except Exception:
                log.exception("Order rejected: %s", client_id)

        return submitted

    # ----- fills -----

    async def _on_trade_update(self, update):
        if update.event not in ("fill", "partial_fill"):
            if update.event in ("rejected", "canceled", "expired"):
                log.info("Order %s %s", update.order.client_order_id, update.event)
            return

        order = update.order
        symbol = order.symbol
        side = "buy" if order.side == OrderSide.BUY else "sell"
        qty = float(update.qty)
        price = float(update.price)

        with self.lock:
            meta = self.orders.get(order.client_order_id, {})
            held_before = float(update.position_qty) + (qty if side == "sell" else -qty)
            cost = self.avg_cost.get(symbol, price)

            if side == "buy":
                total = held_before + qty
                self.avg_cost[symbol] = (held_before * cost + qty * price) / total if total else price
                pnl = 0.0
            else:
                pnl = qty * (price - cost)

        self.logger.log_trade(
            update.timestamp,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            order_id=order.client_order_id,
            sources=meta.get("sources", ""),
            confidence=meta.get("confidence", ""),
            target_weight=meta.get("target_weight", ""),
            commission=0.0,
            realized_pnl=round(pnl, 4),
        )

    def start_fill_stream(self):
        self.stream = TradingStream(API_KEY, SECRET_KEY, paper=True)
        self.stream.subscribe_trade_updates(self._on_trade_update)
        threading.Thread(target=self.stream.run, daemon=True, name="fills").start()

    def stop(self):
        if self.stream:
            self.stream.stop()
