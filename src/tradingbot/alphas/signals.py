# signals.py
# Vectorised alpha scores: signed confidence in [-1, 1] for every bar.
# The alpha models use the latest value; the RL environment uses the whole
# series, so both always see identical signals.
import numpy as np

from tradingbot.alphas.factors import vwap_zscore


def momentum_score(bars, lookback_min=30, threshold=0.002, **_):
    ret = bars["close"].pct_change(lookback_min)
    score = np.sign(ret) * np.minimum(1.0, ret.abs() / (4 * threshold))
    return score.where(ret.abs() >= threshold, 0.0).fillna(0.0)


def mean_reversion_score(bars, vwap_z_entry=2.0, lookback_min=60, **_):
    z = vwap_zscore(bars, lookback_min)
    score = -np.sign(z) * np.minimum(1.0, z.abs() / (2 * vwap_z_entry))
    return score.where(z.abs() >= vwap_z_entry, 0.0).fillna(0.0)
