# portfolio.py
class Portfolio:

    def __init__(self, starting_cash):
        self.cash = starting_cash
        self.positions = {}

    def buy(self, symbol, quantity, price):
        cost = quantity * price

        if cost > self.cash:
            print("Not enough cash!")
            return False

        self.cash -= cost

        if symbol not in self.positions:
            self.positions[symbol] = 0

        self.positions[symbol] += quantity

        print(f"BUY {quantity} {symbol} @ ${price:.2f}")

        return True

    def sell(self, symbol, quantity, price):

        if symbol not in self.positions:
            print("You don't own this stock!")
            return False

        if quantity > self.positions[symbol]:
            print("You don't own enough shares!")
            return False

        revenue = quantity * price

        self.cash += revenue
        self.positions[symbol] -= quantity

        if self.positions[symbol] == 0:
            del self.positions[symbol]

        print(f"SELL {quantity} {symbol} @ ${price:.2f}")

        return True

    def print_status(self):

        print()
        print("===== PORTFOLIO =====")
        print(f"Cash: ${self.cash:.2f}")
        print("Positions:")

        for symbol, quantity in self.positions.items():
            print(f"  {symbol}: {quantity}")

        print("=====================")

    def get_value(self, prices):

        value = self.cash

        for symbol, quantity in self.positions.items():

            if symbol in prices:
                value += quantity * prices[symbol]

        return value

    def get_profit_loss(self, prices, starting_cash):

        current_value = self.get_value(prices)

        return current_value - starting_cash