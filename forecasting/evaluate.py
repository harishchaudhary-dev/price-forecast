"""Metrics and time-series-correct validation.

No random splits anywhere: every evaluation trains strictly on the past and
scores on the future, which is the only honest way to validate a forecaster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import REGISTRY, SarimaModel


def metrics(actual, pred) -> dict:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    err = a - p
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err / a)) * 100),
        "smape": float(np.mean(2 * np.abs(err) / (np.abs(a) + np.abs(p))) * 100),
        "bias_pct": float(np.mean((p - a) / a) * 100),  # +ve = over-forecasting
    }


def interval_coverage(actual, lower, upper) -> float:
    a = np.asarray(actual, float)
    return float(np.mean((a >= np.asarray(lower)) & (a <= np.asarray(upper))) * 100)


def make_model(name: str, sarima_orders: tuple | None = None):
    if name == "sarima" and sarima_orders:
        return SarimaModel(*sarima_orders)
    return REGISTRY[name]()


def holdout_evaluation(y: pd.Series, names, horizon: int = 12, sarima_orders=None) -> tuple[pd.DataFrame, dict]:
    """Train on everything except the last `horizon` months, test on them.

    Returns a metrics table (train + test per model) and the test forecasts.
    """
    train, test = y.iloc[:-horizon], y.iloc[-horizon:]
    rows, forecasts = [], {}
    for name in names:
        model = make_model(name, sarima_orders)
        if len(train) < model.min_obs:
            continue
        model.fit(train)
        fit = model.fitted()
        fc = model.predict(horizon)
        forecasts[name] = fc
        tr = metrics(train.loc[fit.index], fit)
        te = metrics(test, fc["mean"])
        rows.append({
            "model": name,
            "train_mape": tr["mape"], "train_rmse": tr["rmse"],
            "test_mae": te["mae"], "test_rmse": te["rmse"], "test_mape": te["mape"],
            "test_bias_pct": te["bias_pct"],
            "interval_coverage_95": interval_coverage(test, fc["lower"], fc["upper"]),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("test_mape"), forecasts


def walk_forward(y: pd.Series, names, horizon: int = 12, n_folds: int = 4, step: int = 3,
                 sarima_orders=None) -> pd.DataFrame:
    """Expanding-window backtest with several forecast origins.

    A single holdout year can flatter or punish a model by luck. Averaging over
    several origins gives a more reliable ranking.
    """
    n = len(y)
    rows = []
    for fold in range(n_folds):
        cut = n - horizon - (n_folds - 1 - fold) * step
        train, test = y.iloc[:cut], y.iloc[cut:cut + horizon]
        for name in names:
            model = make_model(name, sarima_orders)
            if len(train) < model.min_obs:
                continue
            fc = model.fit(train).predict(len(test))
            m = metrics(test, fc["mean"])
            rows.append({"fold": fold, "origin": train.index[-1].strftime("%Y-%m"), "model": name, **m})
    return pd.DataFrame(rows)


def summarise_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    return (bt.groupby("model")[["mae", "rmse", "mape", "bias_pct"]]
              .agg(["mean", "std"]).sort_values(("mape", "mean")))
