import numpy as np

from monitoring.drift import drift_check, performance_check, psi
import pandas as pd


def test_psi_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=2000)
    assert psi(x, x) < 0.01


def test_performance_degraded():
    preds = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "predicted_price": [100, 100, 100]})
    acts = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "price": [150, 160, 170]})
    assert performance_check(preds, acts, backtest_mape=8)["status"] == "degraded"


def test_drift_detected_on_regime_change():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 0.03, 120)
    recent = pd.Series(np.exp(np.cumsum(rng.normal(0.15, 0.2, 30))) * 1000)
    assert drift_check(ref, recent)["status"] == "drift"
