# main.py
from trading.portfolio import Portfolio
from utils.market_data import get_stock_price


STARTING_CASH = 10_000
SYMBOL = "AAPL"
QUANTITY = 10


portfolio = Portfolio(STARTING_CASH)

# Get current market price
price = get_stock_price(SYMBOL)

print(f"AAPL price: ${price:.2f}")

# Buy AAPL
portfolio.buy(SYMBOL, QUANTITY, price)

portfolio.print_status()


# Check portfolio value
prices = {
    SYMBOL: price + 10
}

value = portfolio.get_value(prices)

print(f"Portfolio value: ${value:.2f}")


# Check profit/loss
profit_loss = portfolio.get_profit_loss(
    prices,
    STARTING_CASH
)

print(f"Profit/Loss: ${profit_loss:.2f}")


# Sell AAPL
portfolio.sell(SYMBOL, QUANTITY, price)

portfolio.print_status()