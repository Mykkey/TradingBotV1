# env.py
# One episode = one trading day of 5-minute decisions. The agent outputs
# target weights (softmax over symbols + cash, capped per symbol); reward is
# the net-of-cost log return minus a penalty for deepening drawdown.
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def action_to_weights(action, max_position):
    logits = np.asarray(action, dtype=np.float64) * 3.0
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()

    assets = np.minimum(weights[:-1], max_position)
    return np.append(assets, 1.0 - assets.sum())


def build_observation(features, weights, tod, bench_ret, day_return):
    return np.concatenate([
        features.ravel(),
        weights,
        [tod, bench_ret, day_return * 100],
    ]).astype(np.float32).clip(-10, 10)


class AllocationEnv(gym.Env):

    def __init__(self, panel, max_position=0.1, cost_bps=3.0, risk_penalty=0.1,
                 random_days=True):
        super().__init__()
        self.panel = panel
        self.max_position = max_position
        self.cost = cost_bps / 10_000
        self.risk_penalty = risk_penalty
        self.random_days = random_days

        days = panel["day"]
        starts = np.flatnonzero(np.append(True, days[1:] != days[:-1]))
        self.day_bounds = list(zip(starts, np.append(starts[1:], len(days))))
        self.next_day = 0

        _, n, f = panel["features"].shape
        self.n = n
        self.observation_space = spaces.Box(-10, 10, (n * f + n + 1 + 3,), np.float32)
        self.action_space = spaces.Box(-1, 1, (n + 1,), np.float32)

    def _obs(self):
        t = self.t
        return build_observation(
            self.panel["features"][t], self.weights, self.panel["tod"][t],
            self.panel["bench_ret"][t], self.equity - 1,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_days:
            day = self.np_random.integers(len(self.day_bounds))
        else:
            day = self.next_day
            self.next_day = (self.next_day + 1) % len(self.day_bounds)

        self.t, self.end = self.day_bounds[day]
        self.weights = np.append(np.zeros(self.n), 1.0)
        self.equity = 1.0
        self.peak = 1.0

        return self._obs(), {}

    def step(self, action):
        return self.step_weights(action_to_weights(action, self.max_position))

    def step_weights(self, target):
        """target: weights for each symbol plus cash, summing to 1."""
        turnover = np.abs(target[:-1] - self.weights[:-1]).sum()

        r = self.panel["returns"][self.t]
        gross = float(target[:-1] @ r)
        net = gross - turnover * self.cost

        # Weights drift with prices until the next decision.
        grown = np.append(target[:-1] * (1 + r), target[-1])
        self.weights = grown / grown.sum()

        self.t += 1
        terminated = self.t >= self.end

        if terminated:
            net -= np.abs(self.weights[:-1]).sum() * self.cost   # flatten at the close

        prev_drawdown = self.equity / self.peak - 1
        self.equity *= 1 + net
        self.peak = max(self.peak, self.equity)
        drawdown = self.equity / self.peak - 1

        reward = 100 * (np.log1p(net) - self.risk_penalty * max(0.0, prev_drawdown - drawdown))

        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, np.float32)
        info = {"net_return": net, "turnover": turnover, "equity": self.equity}

        return obs, float(reward), bool(terminated), False, info
