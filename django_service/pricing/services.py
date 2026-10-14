"""Business logic kept out of views so commands, views and tests share it."""
from __future__ import annotations

import json
import logging
import threading
import urllib.request
from datetime import date

import pandas as pd
from django.conf import settings
from django.db import transaction

from forecasting.predict import forecast as run_forecast, load_bundle
from monitoring.drift import evaluate as evaluate_drift

from .models import ActualPrice, ForecastLog, ModelVersion, MonitoringRun

log = logging.getLogger(__name__)
_cache: dict[str, dict] = {}
_lock = threading.Lock()


class NoActiveModel(Exception):
    pass


def active_version() -> ModelVersion:
    mv = ModelVersion.objects.filter(is_active=True).first()
    if mv is None:
        raise NoActiveModel("No active model. Run: python manage.py train_model --data <csv> --activate")
    return mv


def get_bundle(mv: ModelVersion) -> dict:
    """Load each bundle once per process; joblib loads are too slow per request."""
    with _lock:
        if mv.version not in _cache:
            _cache.clear()  # keep only the current model in memory
            _cache[mv.version] = load_bundle(mv.bundle_path)
        return _cache[mv.version]


def register_from_training(result: dict, activate: bool = False) -> ModelVersion:
    meta = result["meta"]
    mv = ModelVersion.objects.create(
        version=meta["version"], model_name=meta["model_name"], description=meta["model_description"],
        bundle_path=result["bundle_path"], train_start=meta["train_start"], train_end=meta["train_end"],
        backtest_mape=meta["reference"]["backtest_mape"],
        metrics={"holdout": meta["holdout_metrics"], "backtest_mape_mean": meta["backtest_mape_mean"],
                 "cleaning": meta["cleaning"]})
    if activate:
        mv.activate()
        notify_fastapi_reload(mv.version)
    return mv


def notify_fastapi_reload(version: str):
    """Tell the FastAPI service to hot-swap. Failure must not break training."""
    try:
        req = urllib.request.Request(f"{settings.FASTAPI_URL}/admin/reload?version={version}", method="POST",
                                     headers={"X-API-Key": settings.FASTAPI_API_KEY})
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:  # noqa: BLE001
        log.warning("FastAPI reload notification failed (is it running?): %s", e)


def _forecast_via_fastapi(horizon: int) -> dict:
    req = urllib.request.Request(f"{settings.FASTAPI_URL}/predict?horizon={horizon}",
                                 headers={"X-API-Key": settings.FASTAPI_API_KEY})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def get_forecast(horizon: int) -> dict:
    mv = active_version()
    if settings.FORECAST_BACKEND == "fastapi":
        data = _forecast_via_fastapi(horizon)
        if data["model_version"] != mv.version:
            log.warning("FastAPI serves %s but Django registry says %s", data["model_version"], mv.version)
        points = data["forecast"]
    else:
        points = run_forecast(get_bundle(mv), horizon)
    _log_forecasts(mv, points)
    return {"model_version": mv.version, "model": mv.description, "last_actual_month": mv.train_end,
            "backtest_mape": mv.backtest_mape, "forecast": points}


def _log_forecasts(mv: ModelVersion, points: list[dict]):
    with transaction.atomic():
        for p in points:
            y, m = map(int, p["month"].split("-"))
            ForecastLog.objects.get_or_create(
                model_version=mv, target_month=date(y, m, 1),
                defaults={"predicted_price": p["predicted_price"], "lower_95": p["lower_95"],
                          "upper_95": p["upper_95"]})


def run_monitoring(mv: ModelVersion | None = None) -> MonitoringRun:
    mv = mv or active_version()
    actuals = pd.DataFrame(ActualPrice.objects.values("month", "price"))
    if actuals.empty:
        result = {"model_version": mv.version, "performance": {"status": "insufficient_data", "n_matched": 0},
                  "drift": {"status": "insufficient_data"}, "retrain_recommended": False}
    else:
        actuals["month"] = pd.to_datetime(actuals["month"]).dt.strftime("%Y-%m")
        preds = pd.DataFrame(mv.forecasts.values("target_month", "predicted_price"))
        if not preds.empty:
            preds["month"] = pd.to_datetime(preds.pop("target_month")).dt.strftime("%Y-%m")
        result = evaluate_drift(get_bundle(mv), actuals, preds if not preds.empty else None)
    return MonitoringRun.objects.create(model_version=mv, result=result,
                                        retrain_recommended=result["retrain_recommended"])
