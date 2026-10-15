import tempfile
from pathlib import Path

from django.contrib.auth.models import User
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from forecasting.train import run
from pricing.models import ForecastLog, ModelVersion
from pricing.services import register_from_training

DATA = Path(__file__).resolve().parents[2] / "data" / "sample_prices_SYNTHETIC.csv"


class ForecastApiTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.tmp = tempfile.mkdtemp()
        res = run(str(DATA), cls.tmp, date_col="Arrival_Date", price_col="Modal_Price",
                  filters={"Commodity": "Onion"}, n_folds=2)
        with override_settings(FASTAPI_URL="http://127.0.0.1:9"):
            cls.mv = register_from_training(res, activate=True)
        cls.admin = User.objects.create_superuser("admin", "a@x.com", "pw-123456789")

    def setUp(self):
        self.c = APIClient()

    def test_root_redirects_to_health(self):
        response = self.c.get("/", follow=True)
        self.assertEqual(response.redirect_chain, [("/api/health/", 302)])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_forecast_public_and_logged(self):
        r = self.c.get("/api/forecast/?horizon=6")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["forecast"]), 6)
        self.assertEqual(ForecastLog.objects.count(), 6)
        self.c.get("/api/forecast/?horizon=6")
        self.assertEqual(ForecastLog.objects.count(), 6)  # idempotent

    def test_bad_horizon(self):
        self.assertEqual(self.c.get("/api/forecast/?horizon=0").status_code, 400)

    def test_actuals_require_admin(self):
        payload = [{"month": "2026-01", "price": 1900}]
        self.assertEqual(self.c.post("/api/actuals/", payload, format="json").status_code, 401)
        self.c.force_authenticate(self.admin)
        self.assertEqual(self.c.post("/api/actuals/", payload, format="json").status_code, 201)

    def test_monitoring_flow(self):
        self.c.force_authenticate(self.admin)
        self.c.get("/api/forecast/?horizon=12")
        self.c.post("/api/actuals/", [{"month": f"2026-0{i}", "price": 2000 + 100 * i} for i in range(1, 5)],
                    format="json")
        r = self.c.post("/api/monitoring/")
        self.assertEqual(r.status_code, 201)
        self.assertIn("retrain_recommended", r.json())

    def test_single_active_model(self):
        self.assertEqual(ModelVersion.objects.filter(is_active=True).count(), 1)
