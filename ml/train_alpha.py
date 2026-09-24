# train_alpha.py
# Purged walk-forward training for the LightGBM alpha:
#   for each month m (after `min_train_months`): train on everything before m
#   minus a 1-day embargo, test on m. Then fit a final model on all data up to
#   `until` and save it with its metadata.
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from alphas.ml_alpha import ml_features


log = logging.getLogger(__name__)

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}
NUM_ROUNDS = 300


def build_dataset(bars, benchmark, alpha_settings, horizon=15, dead_band=0.0005,
                  sample_every=5):
    frames = []
    bench = bars.get(benchmark)

    for symbol, df in bars.items():
        features = ml_features(df, bench, alpha_settings)

        et = df.index.tz_convert("America/New_York")
        day = pd.Series(et.date, index=df.index)
        future = df["close"].shift(-horizon)
        same_day = day.shift(-horizon) == day
        features["fwd"] = (future / df["close"] - 1).where(same_day)

        features = features[features["tod"] % sample_every == 0]
        features["symbol"] = symbol
        frames.append(features)

    data = pd.concat(frames).dropna()
    data = data[data["fwd"].abs() >= dead_band]
    data["label"] = (data["fwd"] > 0).astype(int)

    return data.sort_index()


def evaluate(y, fwd, p):
    top = fwd[p >= np.quantile(p, 0.9)].mean()
    bottom = fwd[p <= np.quantile(p, 0.1)].mean()

    return {
        "auc": round(roc_auc_score(y, p), 4),
        "accuracy": round(accuracy_score(y, p > 0.5), 4),
        "top_decile_fwd_bps": round(top * 1e4, 2),
        "bottom_decile_fwd_bps": round(bottom * 1e4, 2),
        "spread_bps": round((top - bottom) * 1e4, 2),
        "n": int(len(y)),
    }


def train_alpha(bars, settings, model_path, until=None):
    ml = settings["ml"]
    data = build_dataset(bars, settings["benchmark"], settings["alphas"],
                         ml["horizon_min"], ml["dead_band"], ml["sample_every_min"])

    if until:
        data = data[data.index < pd.Timestamp(until, tz="America/New_York") + pd.Timedelta(days=1)]

    features = [c for c in data.columns if c not in ("fwd", "label", "symbol")]
    months = data.index.tz_convert("America/New_York").tz_localize(None).to_period("M")
    unique_months = months.unique().sort_values()

    folds = []

    for month in unique_months[ml["min_train_months"]:]:
        test = data[months == month]
        month_start = test.index.min()
        train = data[data.index < month_start - pd.Timedelta(days=1)]   # 1-day embargo

        if len(train) < 10 * PARAMS["min_data_in_leaf"]:
            continue

        model = lgb.train(PARAMS, lgb.Dataset(train[features], train["label"]), NUM_ROUNDS)
        p = model.predict(test[features])

        result = {"month": str(month), **evaluate(test["label"].to_numpy(), test["fwd"].to_numpy(), p)}
        folds.append(result)
        log.info("%s", result)

    final = lgb.train(PARAMS, lgb.Dataset(data[features], data["label"]), NUM_ROUNDS)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(model_path))

    importance = dict(zip(features, final.feature_importance("gain").round(1).tolist()))
    fold_df = pd.DataFrame(folds)
    summary = fold_df[["auc", "accuracy", "spread_bps"]].mean().round(4).to_dict() if folds else {}

    meta = {
        "features": features,
        "trained_until": str(data.index.max().tz_convert("America/New_York").date()),
        "walk_forward_mean": summary,
        "walk_forward_folds": folds,
        "feature_importance": importance,
    }
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    return meta
