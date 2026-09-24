# test_ml_rl.py
import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from tradingbot.alphas.ml_alpha import MLAlpha
from tradingbot.backtest.engine import Backtest
from tradingbot.core.framework import Algorithm
from tradingbot.execution.sim_exec import SimExecution
from tradingbot.ml.train_alpha import build_dataset, train_alpha
from tradingbot.rl.env import AllocationEnv, action_to_weights
from tradingbot.rl.features import build_panel
from tradingbot.rl.train import baseline_weights, evaluate, train_rl
from tradingbot.risk.limits import RiskLimits
from tests.conftest import make_bars


@pytest.fixture
def long_bars():
    return {s: make_bars(days=8, seed=i) for i, s in enumerate(["SPY", "AAA", "BBB"])}


def test_action_to_weights_caps_and_sums():
    w = action_to_weights(np.array([1, 1, 1, -1]), max_position=0.1)
    assert w[:-1].max() <= 0.1 + 1e-12
    assert w.sum() == pytest.approx(1)


def test_panel_and_env(settings, long_bars):
    panel = build_panel(long_bars, ["SPY", "AAA", "BBB"], "SPY", settings["alphas"])

    assert panel["features"].shape[1:] == (3, 4)
    assert len(np.unique(panel["day"])) == 8
    # 77 decisions per day (09:30 ... 15:50 every 5 minutes).
    assert len(panel["day"]) == 8 * 77

    env = AllocationEnv(panel)
    check_env(env, skip_render_check=True)

    # Holding only cash earns exactly nothing.
    stats = evaluate(panel, lambda o, f: np.array([0, 0, 0, 1.0]), {})
    assert stats["total_return_pct"] == 0

    baseline = evaluate(panel, lambda o, f: baseline_weights(f, 0.1), {})
    assert baseline["days"] == 8


def test_ml_alpha_train_and_predict(settings, long_bars, tmp_path):
    settings["ml"]["min_train_months"] = 0
    data = build_dataset(long_bars, "SPY", settings["alphas"])
    assert set(data["label"].unique()) <= {0, 1}

    path = tmp_path / "lgbm.txt"
    meta = train_alpha(long_bars, settings, path)
    assert meta["features"] and path.exists()

    alpha = MLAlpha(path, "SPY", settings["alphas"], entry=0.5)
    now = long_bars["AAA"].index[-1]
    insights = alpha.update(now, {s: b.iloc[-400:] for s, b in long_bars.items()})
    assert all(i.source == "ml" for i in insights)


def test_rl_train_and_allocator_backtest(settings, long_bars, tmp_path):
    settings["universe"] = ["SPY", "AAA", "BBB"]
    settings["rl"]["total_timesteps"] = 256
    settings["strategy"]["rl_model_path"] = str(tmp_path / "ppo.zip")

    meta = train_rl(long_bars, settings, tmp_path / "ppo.zip")
    assert json.loads((tmp_path / "ppo.json").read_text())["symbols"] == ["SPY", "AAA", "BBB"]
    assert "rl_heldout" in meta and "baseline_heldout" in meta

    from tradingbot.portfolio.rl_allocator import RLAllocator

    algorithm = Algorithm(
        alphas=[],
        portfolio_model=RLAllocator(settings),
        risk_model=RiskLimits(**settings["risk"]),
        execution_model=SimExecution(settings["risk"]["min_trade_value"]),
    )
    bars = {s: b.iloc[-390 * 2:] for s, b in long_bars.items()}
    metrics = Backtest(algorithm, bars, settings, tmp_path / "bt").run()

    assert metrics["days"] == 2
