# history.py
# Download Alpaca minute bars once and cache them as parquet (one file per
# symbol). Later calls only fetch what is missing.
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from utils.market_data import client


log = logging.getLogger(__name__)

# Free plan: SIP data is only available once it is >15 minutes old.
SIP_DELAY = timedelta(minutes=16)


def regular_hours(bars):
    et = bars.index.tz_convert("America/New_York")
    minutes = et.hour * 60 + et.minute
    return bars[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)]


def _fetch(symbols, start, end, feed):
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed(feed),
    )
    df = client.get_stock_bars(request).df

    if df.empty:
        return {}

    return {
        symbol: regular_hours(group.droplevel("symbol"))
        for symbol, group in df.groupby(level="symbol")
    }


def update_cache(symbols, cache_dir, history_start, feed="sip"):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc) - SIP_DELAY

    for symbol in symbols:
        path = cache / f"{symbol}.parquet"
        existing = pd.read_parquet(path) if path.exists() else None

        if existing is not None and len(existing):
            start = existing.index[-1].to_pydatetime() + timedelta(minutes=1)
        else:
            start = pd.Timestamp(history_start, tz="UTC").to_pydatetime()

        chunks = [] if existing is None else [existing]

        # Monthly chunks keep each request (and memory use) small.
        while start < end:
            chunk_end = min(start + timedelta(days=31), end)
            fetched = _fetch([symbol], start, chunk_end, feed).get(symbol)

            if fetched is not None and len(fetched):
                chunks.append(fetched)

            start = chunk_end

        if not chunks:
            log.warning("No data for %s", symbol)
            continue

        bars = pd.concat(chunks)
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        bars.to_parquet(path)

        log.info("%s: %d bars cached (%s -> %s)", symbol, len(bars), bars.index[0], bars.index[-1])


def load_bars(symbols, cache_dir, start=None, end=None):
    bars = {}

    for symbol in symbols:
        path = Path(cache_dir) / f"{symbol}.parquet"

        if not path.exists():
            log.warning("No cached data for %s; run `python main.py download`", symbol)
            continue

        df = pd.read_parquet(path)

        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="America/New_York")]

        if end is not None:
            df = df[df.index < pd.Timestamp(end, tz="America/New_York") + timedelta(days=1)]

        bars[symbol] = df

    return bars
