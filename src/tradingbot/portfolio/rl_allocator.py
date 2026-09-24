# rl_allocator.py
# Portfolio construction by the PPO policy trained in rl/train.py.
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from tradingbot.config import ROOT
from tradingbot.core.framework import PortfolioConstructionModel
from tradingbot.rl.env import action_to_weights, build_observation
from tradingbot.rl.features import ASSET_FEATURES, asset_features


ET = ZoneInfo("America/New_York")


class RLAllocator(PortfolioConstructionModel):

    def __init__(self, settings):
        from stable_baselines3 import PPO

        path = ROOT / settings["strategy"]["rl_model_path"]
        self.model = PPO.load(path, device="cpu")
        self.meta = json.loads(Path(path).with_suffix(".json").read_text())
        self.symbols = self.meta["symbols"]
        self.max_position = self.meta["env"]["max_position"]
        self.alpha_settings = settings["alphas"]
        self.benchmark = settings["benchmark"]

    def create_targets(self, now, insights, state):
        history, prices = state["history"], state["prices"]
        features = np.zeros((len(self.symbols), len(ASSET_FEATURES)), dtype=np.float32)
        weights = np.zeros(len(self.symbols) + 1)

        for j, symbol in enumerate(self.symbols):
            bars = history.get(symbol)

            if bars is not None and len(bars):
                features[j] = asset_features(bars, self.alpha_settings).iloc[-1].to_numpy()

            if symbol in state["positions"] and symbol in prices:
                weights[j] = state["positions"][symbol] * prices[symbol] / state["equity"]

        weights[-1] = 1.0 - weights[:-1].sum()

        et = now.astimezone(ET)
        tod = ((et.hour - 9) * 60 + et.minute - 30) / 390
        b = self.symbols.index(self.benchmark) if self.benchmark in self.symbols else 0
        day_return = state["equity"] / state["day_start_equity"] - 1

        obs = build_observation(features, weights, tod,
                                features[b, ASSET_FEATURES.index("ret_60")], day_return)
        action, _ = self.model.predict(obs, deterministic=True)
        target = action_to_weights(action, self.max_position)

        targets = {symbol: 0.0 for symbol in state["positions"]}
        targets.update({s: float(w) for s, w in zip(self.symbols, target[:-1]) if s in prices})

        return targets
