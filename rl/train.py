# train.py
# Train a PPO allocator on the days up to `train_until` and evaluate it on the
# held-out days after, against the equal-weight baseline in the same simulator.
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rl.env import AllocationEnv, action_to_weights
from rl.features import ASSET_FEATURES, build_panel


log = logging.getLogger(__name__)


def split_panel(panel, until):
    cutoff = pd.Timestamp(until).date()
    train_mask = panel["day"] <= cutoff

    def take(mask):
        return {k: v[mask] for k, v in panel.items()}

    return take(train_mask), take(~train_mask)


def baseline_weights(features, max_position):
    """Equal weight across symbols with a positive combined alpha score."""
    score = features[:, ASSET_FEATURES.index("mom_score")] + features[:, ASSET_FEATURES.index("mr_score")]
    longs = score > 0
    assets = np.zeros(len(score))

    if longs.any():
        assets[longs] = min(max_position, 1.0 / longs.sum())

    return np.append(assets, 1.0 - assets.sum())


def evaluate(panel, choose_weights, env_kwargs):
    env = AllocationEnv(panel, random_days=False, **env_kwargs)
    daily, turnover = [], []

    for _ in range(len(env.day_bounds)):
        obs, _ = env.reset()
        done = False

        while not done:
            weights = choose_weights(obs, panel["features"][env.t])
            obs, _, done, _, info = env.step_weights(weights)
            turnover.append(info["turnover"])

        daily.append(info["equity"] - 1)

    daily = np.array(daily)
    std = daily.std()

    return {
        "days": len(daily),
        "total_return_pct": round((np.prod(1 + daily) - 1) * 100, 3),
        "sharpe": round(daily.mean() / std * np.sqrt(252), 3) if std > 0 else None,
        "win_days_pct": round((daily > 0).mean() * 100, 1),
        "worst_day_pct": round(daily.min() * 100, 3),
        "avg_turnover_per_step": round(float(np.mean(turnover)), 4),
    }


def train_rl(bars, settings, model_path):
    from stable_baselines3 import PPO

    rl = settings["rl"]
    symbols = [s for s in settings["universe"] if s in bars]
    panel = build_panel(bars, symbols, settings["benchmark"], settings["alphas"],
                        settings["strategy"]["decision_interval_min"])

    days = np.unique(panel["day"])
    until = rl.get("train_until") or str(days[int(len(days) * 0.8)])
    train, test = split_panel(panel, until)
    log.info("RL: %d train days, %d held-out days (split %s)",
             len(np.unique(train["day"])), len(np.unique(test["day"])), until)

    env_kwargs = {
        "max_position": settings["risk"]["max_position_pct"],
        "cost_bps": rl["turnover_cost_bps"],
        "risk_penalty": rl["risk_penalty"],
    }

    model = PPO(
        "MlpPolicy",
        AllocationEnv(train, **env_kwargs),
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        policy_kwargs={"net_arch": [256, 256]},
        seed=0,
        verbose=1,
    )
    model.learn(total_timesteps=rl["total_timesteps"])

    max_position = env_kwargs["max_position"]

    def rl_policy(obs, _features):
        action, _ = model.predict(obs, deterministic=True)
        return action_to_weights(action, max_position)

    def baseline_policy(_obs, features):
        return baseline_weights(features, max_position)

    results = {
        "rl_heldout": evaluate(test, rl_policy, env_kwargs),
        "baseline_heldout": evaluate(test, baseline_policy, env_kwargs),
        "rl_train": evaluate(train, rl_policy, env_kwargs),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    meta = {
        "symbols": symbols,
        "features": ASSET_FEATURES,
        "trained_until": until,
        "env": env_kwargs,
        **results,
    }
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    return meta
