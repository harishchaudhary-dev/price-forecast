import importlib

import pytest
from fastapi.testclient import TestClient

from forecasting.train import run


@pytest.fixture
def client(messy_csv, tmp_path, monkeypatch):
    models = tmp_path / "models"
    run(str(messy_csv), str(models), n_folds=2)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("API_KEY", "secret")
    import fastapi_service.app.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_auth_required(client):
    assert client.post("/predict", json={"horizon": 3}).status_code == 401


def test_predict(client):
    r = client.post("/predict", json={"horizon": 3}, headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 3 and "X-Response-Time-ms" in r.headers


def test_validation(client):
    r = client.post("/predict", json={"horizon": 100}, headers={"X-API-Key": "secret"})
    assert r.status_code == 422
