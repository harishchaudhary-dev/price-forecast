"""End-to-end training pipeline.

    python -m forecasting.train --data data/prices.csv --out models/
    python -m forecasting.train --data data/prices.csv --filter Commodity=Onion --filter Market=Bangalore

Produces models/<version>/bundle.joblib, report.json and forecast.csv, and
updates models/latest.txt to point at the new version. The serving layers
(FastAPI, Django) only ever read the bundle, never the training code paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import load_series
from .evaluate import holdout_evaluation, summarise_backtest, walk_forward, make_model
from .models import BASELINES, REGISTRY, SarimaModel

CANDIDATES = tuple(REGISTRY)


def _data_hash(y: pd.Series) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(y).values.tobytes()).hexdigest()[:12]


def select_best(backtest_summary: pd.DataFrame) -> str:
    """Lowest mean backtest MAPE, but a model must beat seasonal naive to be used.

    If nothing beats the baseline, ship the baseline and say so. Shipping a
    complex model that loses to "same as last year" is a common, avoidable mistake.
    """
    mape = backtest_summary[("mape", "mean")]
    best = mape.idxmin()
    if "seasonal_naive" in mape and best not in BASELINES and mape[best] >= mape["seasonal_naive"]:
        return "seasonal_naive"
    return best


def run(data_path: str, out_dir: str = "models", horizon: int = 12, date_col=None,
        price_col=None, filters=None, n_folds: int = 4) -> dict:
    y, report = load_series(data_path, date_col, price_col, filters)
    if len(y) < 36:
        raise ValueError(f"Only {len(y)} months after cleaning; need >= 36 for seasonal models.")

    sarima_orders = SarimaModel.select_order(y.iloc[:-horizon])  # chosen on train only
    holdout, holdout_fc = holdout_evaluation(y, CANDIDATES, horizon, sarima_orders)
    bt = walk_forward(y, CANDIDATES, horizon, n_folds=n_folds, sarima_orders=sarima_orders)
    bt_summary = summarise_backtest(bt)
    best = select_best(bt_summary)

    final_orders = SarimaModel.select_order(y) if best == "sarima" else None
    model = make_model(best, final_orders).fit(y)
    forecast = model.predict(horizon)

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    log_y = np.log(y)
    meta = {
        "version": version,
        "model_name": best,
        "model_description": model.describe(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_hash": _data_hash(y),
        "filters": filters or {},
        "horizon": horizon,
        "train_start": y.index[0].strftime("%Y-%m"),
        "train_end": y.index[-1].strftime("%Y-%m"),
        "n_months": len(y),
        "cleaning": report.to_dict(),
        "holdout_metrics": holdout.round(4).to_dict(orient="index"),
        "backtest_mape_mean": bt_summary[("mape", "mean")].round(4).to_dict(),
        # Reference statistics used by monitoring/drift.py
        "reference": {
            "log_diff_mean": float(log_y.diff().mean()),
            "log_diff_std": float(log_y.diff().std()),
            "recent_level": float(y.iloc[-12:].mean()),
            "backtest_mape": float(bt_summary.loc[best, ("mape", "mean")]),
            "recent_log_diffs": log_y.diff().dropna().iloc[-60:].round(6).tolist(),
        },
    }

    out = Path(out_dir) / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta, "history": y}, out / "bundle.joblib")
    (out / "report.json").write_text(json.dumps(meta, indent=2, default=str))
    forecast.round(2).rename_axis("month").to_csv(out / "forecast.csv")
    bt.to_csv(out / "backtest.csv", index=False)
    (Path(out_dir) / "latest.txt").write_text(version)

    return {"meta": meta, "y": y, "report": report, "holdout": holdout, "holdout_forecasts": holdout_fc,
            "backtest": bt, "backtest_summary": bt_summary, "model": model, "forecast": forecast,
            "bundle_path": str(out / "bundle.joblib")}


def _parse_filters(items):
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        if not v:
            raise argparse.ArgumentTypeError(f"Bad filter '{item}', expected COLUMN=VALUE")
        out[k.strip()] = v.strip()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the monthly price forecaster.")
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="models")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--date-col")
    p.add_argument("--price-col")
    p.add_argument("--filter", action="append", help="COLUMN=VALUE, repeatable")
    args = p.parse_args(argv)
    res = run(args.data, args.out, args.horizon, args.date_col, args.price_col, _parse_filters(args.filter))
    m = res["meta"]
    print(f"Selected model: {m['model_description']}  (version {m['version']})")
    print(res["backtest_summary"].round(2).to_string())
    print(f"Bundle: {res['bundle_path']}")


if __name__ == "__main__":
    main()
