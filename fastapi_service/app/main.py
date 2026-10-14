"""FastAPI inference service.

Owns ONE job: serve forecasts fast. It loads the model bundle once at startup
(never per request), validates input with Pydantic, and exposes a hot-reload
endpoint so a newly trained model can be swapped in without a restart.

Run:  MODELS_DIR=models uvicorn fastapi_service.app.main:app --reload --port 8001
Docs: http://localhost:8001/docs
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from forecasting.predict import MAX_HORIZON, forecast, load_bundle, resolve_bundle_path

from .schemas import ForecastRequest, ForecastResponse, ModelInfo

log = logging.getLogger("forecast-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODELS_DIR = os.getenv("MODELS_DIR", "models")
API_KEY = os.getenv("API_KEY")  # unset = auth disabled (local dev only)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")


class ModelStore:
    """Holds the active bundle; swapped atomically under a lock on reload."""

    def __init__(self):
        self._lock = threading.Lock()
        self.bundle = None

    def load(self, version: str | None = None):
        bundle = load_bundle(resolve_bundle_path(MODELS_DIR, version))
        with self._lock:
            self.bundle = bundle
        log.info("Loaded model %s (%s)", bundle["meta"]["version"], bundle["meta"]["model_description"])
        return bundle

    def get(self):
        if self.bundle is None:
            raise HTTPException(503, "No model loaded. Train one with forecasting.train first.")
        return self.bundle


store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store.load()
    except FileNotFoundError as e:
        log.warning("Starting without a model: %s", e)
    yield


app = FastAPI(title="Monthly Price Forecast API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"])


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    log.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


def require_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")


@app.get("/")
def root():
    return {"service": "Monthly Price Forecast API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": store.bundle is not None}


@app.get("/model", response_model=ModelInfo, dependencies=[Depends(require_key)])
def model_info():
    m = store.get()["meta"]
    return ModelInfo(version=m["version"], model=m["model_description"], trained_at=m["trained_at"],
                     train_start=m["train_start"], train_end=m["train_end"],
                     backtest_mape=m["reference"]["backtest_mape"])


@app.post("/predict", response_model=ForecastResponse, dependencies=[Depends(require_key)])
def predict(req: ForecastRequest):
    bundle = store.get()
    return ForecastResponse(model_version=bundle["meta"]["version"],
                            last_actual_month=bundle["meta"]["train_end"],
                            forecast=forecast(bundle, req.horizon))


@app.get("/predict", response_model=ForecastResponse, dependencies=[Depends(require_key)])
def predict_get(horizon: int = 12):
    if not 1 <= horizon <= MAX_HORIZON:
        raise HTTPException(422, f"horizon must be 1..{MAX_HORIZON}")
    return predict(ForecastRequest(horizon=horizon))


@app.post("/admin/reload", dependencies=[Depends(require_key)])
def reload_model(version: str | None = None):
    """Called by the training pipeline (or Django) after a new model is promoted."""
    try:
        b = store.load(version)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"loaded": b["meta"]["version"]}
