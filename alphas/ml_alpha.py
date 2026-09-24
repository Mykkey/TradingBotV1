# ml_alpha.py
# LightGBM classifier predicting whether the next `horizon` minutes' return is
# positive. Trained by ml/train_alpha.py (python main.py train-alpha).
import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np

from alphas.factors import feature_frame
from alphas.signals import mean_reversion_score, momentum_score
from core.framework import AlphaModel
from core.insight import Direction, Insight


log = logging.getLogger(__name__)


def ml_features(bars, benchmark_bars, alpha_settings):
    features = feature_frame(bars, benchmark_bars)
    features["mom_score"] = momentum_score(bars, **alpha_settings.get("momentum", {}))
    features["mr_score"] = mean_reversion_score(bars, **alpha_settings.get("mean_reversion", {}))
    return features


class MLAlpha(AlphaModel):

    name = "ml"

    def __init__(self, model_path="models/lgbm_alpha.txt", benchmark="SPY",
                 alpha_settings=None, entry=0.55, period_min=15):
        import lightgbm as lgb

        self.model = lgb.Booster(model_file=str(model_path))
        self.meta = json.loads(Path(model_path).with_suffix(".json").read_text())
        self.features = self.meta["features"]
        self.benchmark = benchmark
        self.alpha_settings = alpha_settings or {}
        self.entry = entry
        self.period = timedelta(minutes=period_min)
        self.warned = False

    def update(self, now, history):
        if not self.warned and str(now.date()) <= self.meta["trained_until"]:
            log.warning("ML alpha is being used on dates it was trained on "
                        "(trained until %s): results are in-sample", self.meta["trained_until"])
            self.warned = True

        bench = history.get(self.benchmark)
        rows, symbols = [], []

        for symbol, bars in history.items():
            if len(bars) < 61:
                continue

            last = ml_features(bars, bench, self.alpha_settings).iloc[-1]
            rows.append(last[self.features].to_numpy(dtype=float))
            symbols.append(symbol)

        if not rows:
            return []

        probs = self.model.predict(np.vstack(rows))
        insights = []

        for symbol, p in zip(symbols, probs):
            if 1 - self.entry < p < self.entry:
                continue

            insights.append(Insight(
                symbol=symbol,
                direction=Direction.UP if p > 0.5 else Direction.DOWN,
                generated_at=now,
                period=self.period,
                # p = 0.55 -> 0.5, p >= 0.60 -> 1.0
                confidence=float(min(1.0, abs(p - 0.5) / 0.1)),
                magnitude=float(p),
                source=self.name,
            ))

        return insights


def load_ml_alpha(settings, **kwargs):
    from config import ROOT

    path = ROOT / settings["strategy"].get("ml_model_path", "models/lgbm_alpha.txt")
    return MLAlpha(path, settings["benchmark"], settings["alphas"], **kwargs)
