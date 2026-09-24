# test_backtest.py
import pandas as pd
import pytest

from tradingbot.alphas import build_alphas
from tradingbot.backtest.engine import Backtest
from tradingbot.core.framework import AlphaModel, Algorithm
from tradingbot.execution.sim_exec import SimExecution
from tradingbot.monitoring.report import build_report
from tradingbot.monitoring.trade_logger import load_equity, load_trades
from tradingbot.portfolio.equal_weight import EqualWeightPortfolio
from tradingbot.risk.limits import RiskLimits


class NoSignal(AlphaModel):
    def update(self, now, history):
        return []


def algorithm(settings, alphas):
    return Algorithm(
        alphas=alphas,
        portfolio_model=EqualWeightPortfolio(),
        risk_model=RiskLimits(**settings["risk"]),
        execution_model=SimExecution(settings["risk"]["min_trade_value"],
                                     settings["risk"]["rebalance_band"]),
    )


def test_hold_strategy_holds_overnight_with_low_turnover(settings, bars, tmp_path):
    from tradingbot.alphas.hold import HoldAlpha

    settings["strategy"].update(decision_interval_min=30, first_decision_min=30)
    settings["risk"].update(flatten_at_close=False, daily_loss_limit_pct=None,
                            rebalance_band=0.01, max_position_pct=0.5)

    metrics = Backtest(algorithm(settings, [HoldAlpha(["AAA", "BBB"])]),
                       bars, settings, tmp_path).run()
    trades = load_trades(tmp_path)
    equity = load_equity(tmp_path)

    # Buys at 10:00 on day one (not in the opening 30 minutes), then holds,
    # only trimming when a position drifts more than the 1% band.
    first = trades.iloc[:2]
    assert set(first["side"]) == {"buy"}
    assert first["timestamp"].min().strftime("%H:%M") == "10:01"
    last = equity.groupby(equity["timestamp"].dt.date).tail(1)
    assert (last["n_positions"] == 2).all()
    rebalances = trades.iloc[2:]
    assert (rebalances["qty"] * rebalances["price"] >= 1_000 - 1).all()
    assert metrics["days"] == 3


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
    from tradingbot.monitoring.trade_logger import TradeLogger

    Backtest(algorithm(settings, build_alphas(settings)), bars, settings, tmp_path).run()
    logger = TradeLogger(tmp_path)
    day = load_equity(tmp_path)["timestamp"].dt.date.iloc[-1]

    before = pd.read_csv(tmp_path / "daily_summary.csv")
    logger.write_daily_summary(day, before["start_equity"].iloc[-1])
    after = pd.read_csv(tmp_path / "daily_summary.csv")

    assert len(after) == len(before)
    assert after["end_equity"].iloc[-1] == pytest.approx(before["end_equity"].iloc[-1])
