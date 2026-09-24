# live.py
# Real-time minute bars from Alpaca's websocket (IEX feed on the free plan),
# kept in a rolling in-memory buffer per symbol.
import logging
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from data.history import regular_hours
from utils.market_data import API_KEY, SECRET_KEY, client


log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]


class BarBuffer:

    def __init__(self, symbols, max_bars=1000, feed="iex"):
        self.symbols = symbols
        self.max_bars = max_bars
        self.feed = DataFeed(feed)
        self.lock = threading.Lock()
        self.frames = {s: pd.DataFrame(columns=COLUMNS) for s in symbols}
        self.pending = {s: [] for s in symbols}
        self.stream = None
        self.thread = None

    def warm_up(self, days=3):
        """Seed the buffer with recent bars so alphas have lookback at the open."""
        request = StockBarsRequest(
            symbol_or_symbols=self.symbols,
            timeframe=TimeFrame.Minute,
            start=datetime.now(timezone.utc) - timedelta(days=days),
            feed=self.feed,
        )
        df = client.get_stock_bars(request).df

        if df.empty:
            return

        with self.lock:
            for symbol, group in df.groupby(level="symbol"):
                bars = regular_hours(group.droplevel("symbol"))
                self.frames[symbol] = bars[COLUMNS].iloc[-self.max_bars:]

        log.info("Warmed up %d symbols", df.index.get_level_values("symbol").nunique())

    async def _on_bar(self, bar):
        row = {
            "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close,
            "volume": bar.volume, "trade_count": bar.trade_count, "vwap": bar.vwap,
        }

        with self.lock:
            self.pending[bar.symbol].append((pd.Timestamp(bar.timestamp), row))

    def start(self):
        self.stream = StockDataStream(API_KEY, SECRET_KEY, feed=self.feed)
        self.stream.subscribe_bars(self._on_bar, *self.symbols)
        self.thread = threading.Thread(target=self.stream.run, daemon=True, name="bars")
        self.thread.start()

    def stop(self):
        if self.stream:
            self.stream.stop()

    def snapshot(self):
        """Merge newly streamed bars and return (history, latest prices)."""
        with self.lock:
            for symbol, rows in self.pending.items():
                if not rows:
                    continue

                new = pd.DataFrame(
                    [r for _, r in rows],
                    index=pd.DatetimeIndex([t for t, _ in rows], name="timestamp"),
                )
                frame = pd.concat([self.frames[symbol], new]) if len(self.frames[symbol]) else new
                frame = frame[~frame.index.duplicated(keep="last")].sort_index()
                self.frames[symbol] = frame.iloc[-self.max_bars:]
                self.pending[symbol] = []

            history = {s: f.copy() for s, f in self.frames.items() if len(f)}

        prices = {s: float(f["close"].iloc[-1]) for s, f in history.items()}

        return history, prices
