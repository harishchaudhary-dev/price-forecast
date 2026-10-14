"""Production monitoring: forecast accuracy and data drift.

Two independent signals decide whether to retrain:
  1. Performance decay: live MAPE of forecasts vs actuals that have since
     arrived, compared to the MAPE the model achieved in backtesting.
  2. Data drift: whether recent month-over-month price changes come from a
     different distribution than the training data (KS test + PSI).

Prices are monthly, so actuals arrive slowly. Data drift is the early warning;
performance decay is the confirmation.

    python -m monitoring.drift --bundle models/<v>/bundle.joblib --actuals new_actuals.csv
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

THRESHOLDS = {
    "mape_ratio": 1.5,   # live MAPE > 1.5x backtest MAPE -> degraded
    "psi": 0.25,         # PSI > 0.25 is the conventional "significant shift"
    "ks_pvalue": 0.01,
    "min_actuals": 3,    # don't judge accuracy on fewer months than this
}


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between two samples, bins from reference quantiles."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def performance_check(predictions: pd.DataFrame, actuals: pd.DataFrame, backtest_mape: float) -> dict:
    """predictions: [month, predicted_price]; actuals: [month, price]."""
    merged = predictions.merge(actuals, on="month", how="inner")
    if len(merged) < THRESHOLDS["min_actuals"]:
        return {"status": "insufficient_data", "n_matched": len(merged)}
    ape = (merged["price"] - merged["predicted_price"]).abs() / merged["price"] * 100
    live = float(ape.mean())
    return {
        "status": "degraded" if live > THRESHOLDS["mape_ratio"] * backtest_mape else "ok",
        "n_matched": len(merged),
        "live_mape": round(live, 3),
        "backtest_mape": round(backtest_mape, 3),
        "bias_pct": round(float(((merged["predicted_price"] - merged["price"]) / merged["price"]).mean() * 100), 3),
    }


def drift_check(reference_log_diffs, recent_prices: pd.Series) -> dict:
    """KS test always; PSI only with enough recent points.

    PSI over 10 bins on ~12 monthly observations is noise (most bins hold 0-2
    points), so it raises false alarms. With monthly data you will usually be
    relying on the KS test plus the performance check for the first couple of years.
    """
    recent = np.log(recent_prices.astype(float)).diff().dropna()
    if len(recent) < 6:
        return {"status": "insufficient_data", "n_recent": len(recent)}
    ks = ks_2samp(reference_log_diffs, recent)
    p = psi(reference_log_diffs, recent, bins=min(10, len(recent) // 5)) if len(recent) >= 24 else None
    drifted = ks.pvalue < THRESHOLDS["ks_pvalue"] or (p is not None and p > THRESHOLDS["psi"])
    return {"status": "drift" if drifted else "ok", "n_recent": len(recent),
            "psi": None if p is None else round(p, 4),
            "ks_stat": round(float(ks.statistic), 4), "ks_pvalue": round(float(ks.pvalue), 4)}


def evaluate(bundle: dict, actuals: pd.DataFrame, predictions: pd.DataFrame | None = None) -> dict:
    ref = bundle["meta"]["reference"]
    actuals = actuals.copy()
    actuals["month"] = pd.to_datetime(actuals["month"]).dt.strftime("%Y-%m")
    if predictions is None:
        fc = bundle["model"].predict(24)
        predictions = pd.DataFrame({"month": fc.index.strftime("%Y-%m"), "predicted_price": fc["mean"].values})
    hist = bundle["history"]
    combined = pd.concat([hist.iloc[-12:], pd.Series(actuals["price"].values,
                          index=pd.to_datetime(actuals["month"]))]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    perf = performance_check(predictions, actuals, ref["backtest_mape"])
    drift = drift_check(ref["recent_log_diffs"], combined)
    return {"model_version": bundle["meta"]["version"], "performance": perf, "drift": drift,
            "retrain_recommended": perf.get("status") == "degraded" or drift.get("status") == "drift"}


def main(argv=None):
    import joblib
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--actuals", required=True, help="CSV with columns month,price")
    args = p.parse_args(argv)
    print(json.dumps(evaluate(joblib.load(args.bundle), pd.read_csv(args.actuals)), indent=2))


if __name__ == "__main__":
    main()
