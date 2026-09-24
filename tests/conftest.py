# conftest.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_settings  # noqa: E402


def make_bars(days=3, seed=0, start="2025-03-03", drift=0.0):
    """Random-walk regular-hours minute bars for `days` weekdays."""
    rng = np.random.default_rng(seed)
    frames = []
    price = 100.0

    for day in pd.bdate_range(start, periods=days):
        index = pd.date_range(
            f"{day.date()} 09:30", periods=390, freq="min", tz="America/New_York"
        ).tz_convert("UTC")
        steps = rng.normal(drift, 0.0008, len(index))
        close = price * np.exp(np.cumsum(steps))
        open_ = np.concatenate([[price], close[:-1]])
        frames.append(pd.DataFrame({
            "open": open_,
            "high": np.maximum(open_, close) * 1.0002,
            "low": np.minimum(open_, close) * 0.9998,
            "close": close,
            "volume": rng.integers(1_000, 10_000, len(index)).astype(float),
            "trade_count": 50.0,
            "vwap": (open_ + close) / 2,
        }, index=index))
        price = close[-1]

    return pd.concat(frames)


@pytest.fixture
def settings():
    s = load_settings()
    s["universe"] = ["SPY", "AAA", "BBB"]
    return s


@pytest.fixture
def bars():
    return {
        "SPY": make_bars(seed=1),
        "AAA": make_bars(seed=2, drift=0.0002),
        "BBB": make_bars(seed=3, drift=-0.0002),
    }
