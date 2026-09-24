# features.py
# Per-symbol observation features for the RL allocator. Used both to build
# the training panel (whole history) and live/backtest (latest row), so the
# policy always sees identically computed inputs.
import numpy as np
import pandas as pd

from alphas.factors import realized_vol, returns
from alphas.signals import mean_reversion_score, momentum_score


ASSET_FEATURES = ["mom_score", "mr_score", "ret_60", "rvol_30"]


def asset_features(bars, alpha_settings):
    return pd.DataFrame({
        "mom_score": momentum_score(bars, **alpha_settings.get("momentum", {})),
        "mr_score": mean_reversion_score(bars, **alpha_settings.get("mean_reversion", {})),
        "ret_60": returns(bars, 60) * 100,
        "rvol_30": realized_vol(bars) * 1000,
    }, index=bars.index).fillna(0.0).clip(-5, 5)


def build_panel(bars, symbols, benchmark, alpha_settings, interval=5, last_decision_min=380):
    """Returns dict of aligned numpy arrays on the decision grid:
    features [T, N, F], returns [T, N] (fill-to-next-fill), tod [T],
    bench_ret [T], day [T] and the decision timestamps."""
    timeline = pd.DatetimeIndex(sorted(set().union(*(bars[s].index for s in symbols if s in bars))))
    et = timeline.tz_convert("America/New_York")
    tod = pd.Series((et.hour - 9) * 60 + et.minute - 30, index=timeline)
    day = pd.Series(et.date, index=timeline)

    decisions = timeline[((tod % interval) == 0) & (tod <= last_decision_min)]
    n, f = len(symbols), len(ASSET_FEATURES)

    features = np.zeros((len(decisions), n, f), dtype=np.float32)
    fills = np.full((len(decisions), n), np.nan)
    day_close = np.full((len(decisions), n), np.nan)

    decision_day = day.loc[decisions].to_numpy()

    for j, symbol in enumerate(symbols):
        if symbol not in bars:
            continue

        df = bars[symbol]
        feats = asset_features(df, alpha_settings).reindex(timeline).ffill()
        features[:, j, :] = feats.loc[decisions].fillna(0.0).to_numpy()

        # Orders decided at t fill at the next bar's open.
        next_open = df["open"].reindex(timeline).shift(-1)
        close = df["close"].reindex(timeline).ffill()
        fills[:, j] = next_open.loc[decisions].fillna(close.loc[decisions]).to_numpy()

        last_close = close.groupby(day.values).last()
        day_close[:, j] = last_close.reindex(decision_day).to_numpy()

    # Return from this decision's fill to the next decision's fill (same day),
    # or to the day's close for the last decision of the day.
    next_fill = np.roll(fills, -1, axis=0)
    last_of_day = np.append(decision_day[1:] != decision_day[:-1], True)
    next_fill[last_of_day] = day_close[last_of_day]
    rets = np.nan_to_num(next_fill / fills - 1, nan=0.0, posinf=0.0, neginf=0.0)

    b = symbols.index(benchmark) if benchmark in symbols else 0

    return {
        "timestamps": decisions,
        "features": features,
        "returns": rets.astype(np.float32),
        "tod": (tod.loc[decisions].to_numpy() / 390).astype(np.float32),
        "bench_ret": features[:, b, ASSET_FEATURES.index("ret_60")],
        "day": decision_day,
    }
