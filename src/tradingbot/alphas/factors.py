# factors.py
# Vectorised minute-bar features. Every function takes a bars DataFrame
# (columns: open, high, low, close, volume, vwap) indexed by timestamp.
import numpy as np
import pandas as pd


def returns(bars, minutes):
    return bars["close"].pct_change(minutes)


def session_vwap(bars):
    """Cumulative VWAP that resets each trading day."""
    day = bars.index.tz_convert("America/New_York").date
    price = bars["vwap"] if "vwap" in bars else bars["close"]
    pv = (price * bars["volume"]).groupby(day).cumsum()
    vol = bars["volume"].groupby(day).cumsum()
    return pv / vol.replace(0, np.nan)


def vwap_zscore(bars, lookback):
    deviation = bars["close"] / session_vwap(bars) - 1
    std = deviation.rolling(lookback, min_periods=lookback // 2).std()
    return deviation / std.replace(0, np.nan)


def rsi(bars, period=14):
    delta = bars["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def volume_zscore(bars, lookback=60):
    v = bars["volume"]
    return (v - v.rolling(lookback).mean()) / v.rolling(lookback).std()


def realized_vol(bars, lookback=30):
    return np.log(bars["close"]).diff().rolling(lookback).std()


def minutes_since_open(bars):
    et = bars.index.tz_convert("America/New_York")
    return pd.Series((et.hour - 9) * 60 + et.minute - 30, index=bars.index)


def relative_strength(bars, benchmark_bars, minutes=30):
    bench = benchmark_bars["close"].reindex(bars.index).ffill()
    return returns(bars, minutes) - bench.pct_change(minutes)


def feature_frame(bars, benchmark_bars=None):
    features = {
        "ret_1": returns(bars, 1),
        "ret_5": returns(bars, 5),
        "ret_15": returns(bars, 15),
        "ret_60": returns(bars, 60),
        "vwap_z": vwap_zscore(bars, 60),
        "rsi_14": rsi(bars),
        "vol_z": volume_zscore(bars),
        "rvol_30": realized_vol(bars),
        "tod": minutes_since_open(bars),
    }

    if benchmark_bars is not None:
        features["rs_30"] = relative_strength(bars, benchmark_bars)

    return pd.DataFrame(features)
