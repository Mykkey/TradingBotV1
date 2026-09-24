# market_data.py
import os

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

from tradingbot.config import ROOT


load_dotenv(ROOT / ".env")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY
)


def get_stock_price(symbol):

    request = StockLatestTradeRequest(
        symbol_or_symbols=symbol
    )

    trades = client.get_stock_latest_trade(request)

    trade = trades[symbol]

    return trade.price