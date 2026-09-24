# portfolio.py
# Local ledger used by the backtester. In live trading, Alpaca's account is
# the source of truth instead.
import logging


log = logging.getLogger(__name__)


class Portfolio:

    def __init__(self, starting_cash, commission_per_share=0.0):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.commission_per_share = commission_per_share
        self.positions = {}
        self.avg_cost = {}
        self.realized_pnl = 0.0
        self.trades = []

    def buy(self, symbol, quantity, price, timestamp=None):
        commission = quantity * self.commission_per_share
        cost = quantity * price + commission

        if cost > self.cash:
            log.warning("Not enough cash to buy %s %s", quantity, symbol)
            return False

        self.cash -= cost

        held = self.positions.get(symbol, 0)
        prev_cost = self.avg_cost.get(symbol, 0.0)

        self.positions[symbol] = held + quantity
        self.avg_cost[symbol] = (held * prev_cost + quantity * price) / (held + quantity)

        self._record(timestamp, symbol, "buy", quantity, price, commission, 0.0)
        log.info("BUY %s %s @ $%.2f", quantity, symbol, price)

        return True

    def sell(self, symbol, quantity, price, timestamp=None):

        if symbol not in self.positions:
            log.warning("You don't own %s", symbol)
            return False

        if quantity > self.positions[symbol]:
            log.warning("You don't own enough %s", symbol)
            return False

        commission = quantity * self.commission_per_share
        pnl = quantity * (price - self.avg_cost[symbol]) - commission

        self.cash += quantity * price - commission
        self.realized_pnl += pnl
        self.positions[symbol] -= quantity

        if self.positions[symbol] == 0:
            del self.positions[symbol]
            del self.avg_cost[symbol]

        self._record(timestamp, symbol, "sell", quantity, price, commission, pnl)
        log.info("SELL %s %s @ $%.2f", quantity, symbol, price)

        return True

    def _record(self, timestamp, symbol, side, quantity, price, commission, pnl):
        self.trades.append({
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "qty": quantity,
            "price": price,
            "commission": commission,
            "realized_pnl": pnl,
        })

    def print_status(self):

        print()
        print("===== PORTFOLIO =====")
        print(f"Cash: ${self.cash:.2f}")
        print("Positions:")

        for symbol, quantity in self.positions.items():
            print(f"  {symbol}: {quantity} @ ${self.avg_cost[symbol]:.2f}")

        print(f"Realized P&L: ${self.realized_pnl:.2f}")
        print("=====================")

    def get_value(self, prices):

        value = self.cash

        for symbol, quantity in self.positions.items():

            if symbol in prices:
                value += quantity * prices[symbol]

        return value

    def get_unrealized_pnl(self, prices):

        return sum(
            quantity * (prices[symbol] - self.avg_cost[symbol])
            for symbol, quantity in self.positions.items()
            if symbol in prices
        )

    def get_profit_loss(self, prices, starting_cash=None):

        if starting_cash is None:
            starting_cash = self.starting_cash

        return self.get_value(prices) - starting_cash
