"""Inference helpers shared by the FastAPI and Django services."""
from __future__ import annotations

from pathlib import Path

import joblib

MAX_HORIZON = 24


def resolve_bundle_path(models_dir: str = "models", version: str | None = None) -> Path:
    root = Path(models_dir)
    if version is None:
        latest = root / "latest.txt"
        if not latest.exists():
            raise FileNotFoundError(f"No trained model in {root}. Run: python -m forecasting.train --data <csv>")
        version = latest.read_text().strip()
    path = root / version / "bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(path)
    if not {"model", "meta", "history"} <= set(bundle):
        raise ValueError(f"{path} is not a valid model bundle")
    return bundle


def forecast(bundle: dict, horizon: int = 12) -> list[dict]:
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be between 1 and {MAX_HORIZON}")
    fc = bundle["model"].predict(horizon)
    last = float(bundle["history"].iloc[-1])
    return [
        {
            "month": idx.strftime("%Y-%m"),
            "predicted_price": round(float(r["mean"]), 2),
            "lower_95": round(float(r["lower"]), 2),
            "upper_95": round(float(r["upper"]), 2),
            "change_vs_last_actual_pct": round((float(r["mean"]) / last - 1) * 100, 2),
        }
        for idx, r in fc.iterrows()
    ]
