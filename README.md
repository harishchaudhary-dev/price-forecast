# Monthly Price Forecasting: Way2Agribusiness Internship Task

12-month-ahead forecasting of average monthly prices, with a production path through Django + FastAPI and
reference Next.js / React Native clients.

> **⚠️ Before submitting:** the task's dataset link was not available when this was built. `data/sample_prices_SYNTHETIC.csv`
> is **synthetic** and exists only so everything runs. Put the real CSV in `data/`, set `DATA_PATH`, `DATE_COL`,
> `PRICE_COL` (and `FILTERS` if the data has several commodities/markets) at the top of the notebook, then
> **Restart & Run All**. Every number and every written interpretation must come from the real data.

## Layout

```
forecasting/          Shared modelling package (used by notebook, Django AND FastAPI)
  data.py             Loading + cleaning (date parsing, dedupe, monthly aggregation, gaps, outliers)
  models.py           Naive, Seasonal naive, Holt-Winters, SARIMA, Gradient boosting (common interface)
  evaluate.py         Metrics, holdout, walk-forward backtest
  train.py            Pipeline + CLI -> versioned model bundle
  predict.py          Inference helpers
monitoring/drift.py   Live accuracy + data-drift checks, retrain recommendation
notebooks/            price_forecasting.ipynb  <- THE SUBMISSION (answers Q1-Q9)
fastapi_service/      Low-latency inference API (hot model reload, API key, Pydantic)
django_service/       Model registry, forecast logging, actuals ingestion, monitoring, admin (DRF)
frontend/nextjs/      Typed API client + forecast page
frontend/react-native/ Forecast screen
tests/                pytest suite (pipeline, leakage, API, drift)
docker-compose.yml    Postgres + Django + FastAPI
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Train (synthetic example; swap in the real CSV and its column names)
python -m forecasting.train --data data/sample_prices_SYNTHETIC.csv \
    --date-col Arrival_Date --price-col Modal_Price --filter Commodity=Onion

# 2. Notebook
cd notebooks && jupyter notebook price_forecasting.ipynb

# 3. Tests
pytest                                   # from repo root
cd django_service && python manage.py test pricing

# 4. FastAPI  (docs at http://localhost:8001/docs)
MODELS_DIR=models API_KEY=dev-key uvicorn fastapi_service.app.main:app --port 8001

# 5. Django   (admin at http://localhost:8000/admin)
cd django_service
python manage.py migrate
python manage.py createsuperuser
python manage.py train_model --data ../data/sample_prices_SYNTHETIC.csv \
    --date-col Arrival_Date --price-col Modal_Price --filter Commodity=Onion --activate
python manage.py runserver
```

Or everything with Docker: `docker compose up --build`.

## API

| Service | Method & path | Auth | Purpose |
|---|---|---|---|
| FastAPI | `GET /health` | none | Liveness + model loaded |
| FastAPI | `GET/POST /predict` (`horizon` 1–24) | `X-API-Key` | Forecast |
| FastAPI | `GET /model` | `X-API-Key` | Active model metadata |
| FastAPI | `POST /admin/reload?version=` | `X-API-Key` | Hot-swap model |
| Django | `GET /api/forecast/?horizon=12` | public, throttled | Forecast (logged for monitoring) |
| Django | `GET/POST /api/actuals/` | POST: admin | Ingest observed prices |
| Django | `GET/POST /api/monitoring/` | admin | Run / read accuracy + drift checks |
| Django | `GET /api/models/`, `POST /api/models/<v>/activate/` | admin | Registry, promote/rollback |
| Django | `POST /api/auth/token/` | creds | Get DRF token |

Set `FORECAST_BACKEND=fastapi` to make Django proxy inference to FastAPI instead of running it in-process.

## Monthly operations

```bash
# new actuals arrive -> POST /api/actuals/
python manage.py check_drift                                   # accuracy + drift
python manage.py train_model --data <refreshed.csv> --activate-if-better   # champion/challenger
```

## Known limitations

Univariate model (no arrivals/rainfall/policy regressors); approximate intervals for Holt-Winters and gradient boosting;
single series. See the notebook's final section.
