# engine.py
# Event-driven minute-bar backtester. Runs the same framework Algorithm as the
# live bot: decide on bar t's close, fill at bar t+1's open plus costs.
import logging
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from tradingbot.monitoring.trade_logger import TradeLogger, load_equity
from tradingbot.backtest.ledger import Ledger


log = logging.getLogger(__name__)

HISTORY_WINDOW = 400   # bars of history handed to alphas each step


class Backtest:

    def __init__(self, algorithm, bars, settings, out_dir, starting_cash=100_000):
        self.algorithm = algorithm
        self.bars = bars
        self.settings = settings
        self.out_dir = Path(out_dir)
        self.benchmark = settings["benchmark"]

        costs = settings["costs"]
        self.cost_rate = (costs["slippage_bps"] + costs["half_spread_bps"]) / 10_000
        self.ledger = Ledger(starting_cash, costs["commission_per_share"])

        self.interval = settings["strategy"]["decision_interval_min"]
        self.flatten_minutes = settings["risk"]["flatten_minutes_before_close"]
        self.snapshot_every = settings["logging"]["equity_snapshot_min"]

    def run(self):
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)

        logger = TradeLogger(self.out_dir)

        timeline = pd.DatetimeIndex(
            sorted(set().union(*(df.index for df in self.bars.values())))
        )
        et = timeline.tz_convert("America/New_York")
        days = pd.Series(et.date, index=timeline)
        day_close = days.groupby(days.values).transform(lambda s: s.index[-1])

        index_of = {s: df.index for s, df in self.bars.items()}
        opens = {s: df["open"].to_numpy() for s, df in self.bars.items()}
        closes = {s: df["close"].to_numpy() for s, df in self.bars.items()}

        prices = {}
        pending = []
        current_day = None
        day_start_equity = self.ledger.cash

        for step, now in enumerate(timeline):
            day = days.iloc[step]

            if day != current_day:
                if current_day is not None:
                    logger.write_daily_summary(current_day, day_start_equity)
                current_day = day
                day_start_equity = self.ledger.get_value(prices)
                log.info("%s  equity $%.2f", day, day_start_equity)

            positions_now = {}

            for symbol, idx in index_of.items():
                pos = idx.searchsorted(now, side="right") - 1
                positions_now[symbol] = pos

                if pos >= 0 and idx[pos] == now:
                    prices[symbol] = closes[symbol][pos]

            self._fill(pending, now, positions_now, index_of, opens, logger)
            pending = []

            minutes_open = int((et[step].hour - 9) * 60 + et[step].minute - 30)
            minutes_to_close = int((day_close.iloc[step] - now).total_seconds() // 60) + 1

            equity = self.ledger.get_value(prices)

            if minutes_open % self.snapshot_every == 0 or minutes_to_close == 1:
                self._snapshot(logger, now, equity, prices)

            decide = minutes_open % self.interval == 0 or minutes_to_close <= self.flatten_minutes

            if not decide or minutes_to_close <= 1:
                continue

            history = {
                symbol: self.bars[symbol].iloc[max(0, pos - HISTORY_WINDOW + 1): pos + 1]
                for symbol, pos in positions_now.items()
                if pos >= 0
            }

            state = {
                "equity": equity,
                "cash": self.ledger.cash,
                "positions": dict(self.ledger.positions),
                "day_start_equity": day_start_equity,
                "minutes_to_close": minutes_to_close,
            }

            pending = self.algorithm.step(now, history, state, prices)

        if current_day is not None:
            logger.write_daily_summary(current_day, day_start_equity)

        return summarize(self.out_dir, self.ledger.starting_cash)

    def _fill(self, orders, now, positions_now, index_of, opens, logger):
        for order in orders:
            symbol = order["symbol"]
            pos = positions_now[symbol]

            # Only fill if this symbol actually traded on this bar.
            if pos < 0 or index_of[symbol][pos] != now:
                continue

            open_price = opens[symbol][pos]

            if order["side"] == "buy":
                price = open_price * (1 + self.cost_rate)
                qty = min(order["qty"], math.floor(self.ledger.cash / price))

                if qty <= 0 or not self.ledger.buy(symbol, qty, price, now):
                    continue
            else:
                price = open_price * (1 - self.cost_rate)
                qty = min(order["qty"], self.ledger.positions.get(symbol, 0))

                if qty <= 0 or not self.ledger.sell(symbol, qty, price, now):
                    continue

            trade = self.ledger.trades[-1]

            logger.log_trade(
                now,
                symbol=symbol,
                side=order["side"],
                qty=qty,
                price=round(price, 4),
                order_id=f"bt-{len(self.ledger.trades)}",
                sources=order["sources"],
                confidence=order["confidence"],
                target_weight=order["target_weight"],
                commission=trade["commission"],
                realized_pnl=round(trade["realized_pnl"], 4),
            )

    def _snapshot(self, logger, now, equity, prices):
        values = [
            qty * prices[symbol]
            for symbol, qty in self.ledger.positions.items()
            if symbol in prices
        ]

        logger.log_equity(
            now,
            equity=round(equity, 2),
            cash=round(self.ledger.cash, 2),
            gross_exposure=round(sum(abs(v) for v in values) / equity, 4),
            net_exposure=round(sum(values) / equity, 4),
            n_positions=len(self.ledger.positions),
            benchmark_price=prices.get(self.benchmark),
        )


def summarize(log_dir, starting_cash):
    equity = load_equity(log_dir)

    if equity.empty:
        return {}

    curve = equity.set_index("timestamp")["equity"]
    daily = curve.groupby(curve.index.date).last()
    daily_returns = daily.pct_change().dropna()
    # First day's return is measured against the starting cash.
    daily_returns = pd.concat([
        pd.Series([daily.iloc[0] / starting_cash - 1]), daily_returns
    ])

    bench = equity.set_index("timestamp")["benchmark_price"].dropna()

    trades_dir = Path(log_dir) / "trades"
    trades = (
        pd.concat(pd.read_csv(f) for f in trades_dir.glob("*.csv"))
        if trades_dir.exists() and any(trades_dir.glob("*.csv")) else pd.DataFrame()
    )
    exits = trades[trades["side"] == "sell"] if len(trades) else trades
    traded_value = (trades["qty"] * trades["price"]).sum() if len(trades) else 0.0

    std = daily_returns.std()

    return {
        "days": len(daily),
        "total_return_pct": round((curve.iloc[-1] / starting_cash - 1) * 100, 3),
        "benchmark_return_pct": (
            round((bench.iloc[-1] / bench.iloc[0] - 1) * 100, 3) if len(bench) else None
        ),
        "sharpe": round(daily_returns.mean() / std * np.sqrt(252), 3) if std > 0 else None,
        "max_drawdown_pct": round((curve / curve.cummax() - 1).min() * 100, 3),
        "n_trades": len(trades),
        "win_rate": round((exits["realized_pnl"] > 0).mean(), 3) if len(exits) else None,
        "avg_daily_turnover": round(traded_value / starting_cash / max(len(daily), 1), 3),
        "avg_gross_exposure": round(equity["gross_exposure"].mean(), 3),
    }
