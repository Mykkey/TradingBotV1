# trade_logger.py
# Append-only CSV logs, one file per trading day. The same schema is written
# by live trading (logs/) and backtests (reports/backtest/...), so
# monitoring/report.py can graph either.
import csv
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ET = ZoneInfo("America/New_York")

TRADE_FIELDS = [
    "timestamp_utc", "timestamp_et", "symbol", "side", "qty", "price",
    "order_id", "sources", "confidence", "target_weight", "commission",
    "realized_pnl",
]

EQUITY_FIELDS = [
    "timestamp_utc", "timestamp_et", "equity", "cash", "gross_exposure",
    "net_exposure", "n_positions", "benchmark_price",
]

SUMMARY_FIELDS = [
    "date", "start_equity", "end_equity", "return_pct", "cum_return_pct",
    "sharpe_to_date", "max_drawdown_pct", "n_trades", "win_rate",
]


def _append(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")

        if new_file:
            writer.writeheader()

        writer.writerow(row)


def _stamp(timestamp):
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    et = ts.tz_convert(ET)
    return ts.isoformat(), et.isoformat(), et.date().isoformat()


class TradeLogger:

    def __init__(self, log_dir="logs"):
        self.dir = Path(log_dir)

    def log_trade(self, timestamp, **fields):
        utc, et, day = _stamp(timestamp)
        row = {"timestamp_utc": utc, "timestamp_et": et, **fields}
        _append(self.dir / "trades" / f"{day}.csv", TRADE_FIELDS, row)

    def log_equity(self, timestamp, **fields):
        utc, et, day = _stamp(timestamp)
        row = {"timestamp_utc": utc, "timestamp_et": et, **fields}
        _append(self.dir / "equity" / f"{day}.csv", EQUITY_FIELDS, row)

    def write_daily_summary(self, day, start_equity):
        """Summarise `day` (a date) and append it to daily_summary.csv."""
        equity_path = self.dir / "equity" / f"{day.isoformat()}.csv"

        if not equity_path.exists():
            return None

        equity = pd.read_csv(equity_path)
        end_equity = float(equity["equity"].iloc[-1])

        trades_path = self.dir / "trades" / f"{day.isoformat()}.csv"
        trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()

        exits = trades[trades["side"] == "sell"] if len(trades) else trades
        win_rate = (exits["realized_pnl"] > 0).mean() if len(exits) else math.nan

        summary_path = self.dir / "daily_summary.csv"
        previous = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        previous = previous[previous["date"] != day.isoformat()] if len(previous) else previous

        first_equity = (
            float(previous["start_equity"].iloc[0]) if len(previous) else start_equity
        )

        daily_returns = list(previous["return_pct"] / 100) if len(previous) else []
        day_return = end_equity / start_equity - 1
        daily_returns.append(day_return)

        returns = pd.Series(daily_returns)
        sharpe = (
            returns.mean() / returns.std() * math.sqrt(252)
            if len(returns) > 1 and returns.std() > 0 else math.nan
        )

        row = {
            "date": day.isoformat(),
            "start_equity": round(start_equity, 2),
            "end_equity": round(end_equity, 2),
            "return_pct": round(day_return * 100, 4),
            "cum_return_pct": round((end_equity / first_equity - 1) * 100, 4),
            "sharpe_to_date": round(sharpe, 3),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 4),
            "n_trades": len(trades),
            "win_rate": round(win_rate, 3),
        }

        summary = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
        summary.sort_values("date")[SUMMARY_FIELDS].to_csv(summary_path, index=False)

        return row

    def max_drawdown(self):
        equity = load_equity(self.dir)

        if equity.empty:
            return 0.0

        curve = equity["equity"]
        return float((curve / curve.cummax() - 1).min())


def load_equity(log_dir):
    files = sorted((Path(log_dir) / "equity").glob("*.csv"))

    if not files:
        return pd.DataFrame(columns=EQUITY_FIELDS)

    df = pd.concat(pd.read_csv(f) for f in files)
    df["timestamp"] = pd.to_datetime(df["timestamp_et"], utc=True).dt.tz_convert(ET)

    return df.sort_values("timestamp").reset_index(drop=True)


def load_trades(log_dir):
    files = sorted((Path(log_dir) / "trades").glob("*.csv"))

    if not files:
        return pd.DataFrame(columns=TRADE_FIELDS)

    df = pd.concat(pd.read_csv(f) for f in files)
    df["timestamp"] = pd.to_datetime(df["timestamp_et"], utc=True).dt.tz_convert(ET)

    return df.sort_values("timestamp").reset_index(drop=True)
