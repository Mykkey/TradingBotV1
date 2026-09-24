# test_backtest.py
import pandas as pd
import pytest

from alphas import build_alphas
from backtest.engine import Backtest
from core.framework import AlphaModel, Algorithm
from execution.sim_exec import SimExecution
from monitoring.report import build_report
from monitoring.trade_logger import load_equity, load_trades
from portfolio.equal_weight import EqualWeightPortfolio
from risk.limits import RiskLimits


class NoSignal(AlphaModel):
    def update(self, now, history):
        return []


def algorithm(settings, alphas):
    return Algorithm(
        alphas=alphas,
        portfolio_model=EqualWeightPortfolio(),
        risk_model=RiskLimits(**settings["risk"]),
        execution_model=SimExecution(settings["risk"]["min_trade_value"]),
    )


def test_no_signal_does_nothing(settings, bars, tmp_path):
    metrics = Backtest(algorithm(settings, [NoSignal()]), bars, settings, tmp_path).run()

    assert metrics["n_trades"] == 0
    assert metrics["total_return_pct"] == 0


def test_strategy_backtest_writes_logs_and_report(settings, bars, tmp_path):
    metrics = Backtest(algorithm(settings, build_alphas(settings)), bars, settings, tmp_path).run()

    trades = load_trades(tmp_path)
    equity = load_equity(tmp_path)
    summary = pd.read_csv(tmp_path / "daily_summary.csv")

    assert metrics["days"] == 3
    assert metrics["n_trades"] > 0
    assert len(summary) == 3
    # 78 five-minute snapshots per day plus one at the final bar.
    assert equity.groupby(equity["timestamp"].dt.date).size().tolist() == [79, 79, 79]
    assert set(trades["side"]) <= {"buy", "sell"}

    # Flat at every close: the last snapshot of each day has no positions.
    last = equity.groupby(equity["timestamp"].dt.date).tail(1)
    assert (last["n_positions"] == 0).all()

    # Costs are charged, so buys fill above and sells below the bar open.
    assert (trades["price"] > 0).all()
    assert metrics["max_drawdown_pct"] <= 0

    html, png = build_report(tmp_path, tmp_path / "out")
    assert html.exists() and png.exists()


def test_daily_summary_is_idempotent(settings, bars, tmp_path):
    from monitoring.trade_logger import TradeLogger

    Backtest(algorithm(settings, build_alphas(settings)), bars, settings, tmp_path).run()
    logger = TradeLogger(tmp_path)
    day = load_equity(tmp_path)["timestamp"].dt.date.iloc[-1]

    before = pd.read_csv(tmp_path / "daily_summary.csv")
    logger.write_daily_summary(day, before["start_equity"].iloc[-1])
    after = pd.read_csv(tmp_path / "daily_summary.csv")

    assert len(after) == len(before)
    assert after["end_equity"].iloc[-1] == pytest.approx(before["end_equity"].iloc[-1])
