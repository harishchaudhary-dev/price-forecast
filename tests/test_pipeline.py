import numpy as np
import pandas as pd
import pytest

from forecasting.data import _parse_dates, load_series
from forecasting.evaluate import metrics, walk_forward
from forecasting.models import REGISTRY
from forecasting.predict import forecast, load_bundle
from forecasting.train import run, select_best


def test_dates_parsed_with_one_consistent_format():
    s = pd.Series(["05/03/2016", "25/03/2016", "11/04/2016"])
    parsed = _parse_dates(s)
    assert list(parsed.dt.month) == [3, 3, 4]  # day-first everywhere, not mixed


def test_cleaning_handles_gaps_outliers_duplicates(messy_csv):
    y, rep = load_series(str(messy_csv))
    assert y.index.freqstr == "MS" and y.isna().sum() == 0
    assert rep.months_missing == 3          # 2 dropped + 1 whose only row was "N/A"
    assert rep.dropped_duplicates == 1
    assert rep.outliers_replaced == 1
    assert y.max() < 5000                   # the x10 value is gone


def test_sustained_spike_is_kept(monthly_series, tmp_path):
    y = monthly_series.copy()
    y.iloc[60:63] *= 1.8                    # 3-month genuine shock
    df = pd.DataFrame({"date": y.index.strftime("%Y-%m-%d"), "price": y.values})
    df.to_csv(tmp_path / "s.csv", index=False)
    clean, rep = load_series(str(tmp_path / "s.csv"))
    assert rep.outliers_replaced == 0
    assert np.allclose(clean.iloc[60:63], y.iloc[60:63])


@pytest.mark.parametrize("name", list(REGISTRY))
def test_models_produce_positive_ordered_forecasts(name, monthly_series):
    fc = REGISTRY[name]().fit(monthly_series).predict(12)
    assert len(fc) == 12 and (fc["mean"] > 0).all()
    assert (fc["lower"] <= fc["mean"]).all() and (fc["mean"] <= fc["upper"]).all()
    assert fc.index[0] == monthly_series.index[-1] + pd.offsets.MonthBegin(1)


def test_walk_forward_never_trains_on_test(monthly_series):
    bt = walk_forward(monthly_series, ["seasonal_naive"], horizon=12, n_folds=3, step=6)
    origins = pd.to_datetime(bt["origin"])
    assert (origins <= monthly_series.index[-13]).all()


def test_metrics():
    m = metrics([100, 200], [110, 180])
    assert m["mae"] == pytest.approx(15) and m["mape"] == pytest.approx(10)


def test_baseline_guard():
    summary = pd.DataFrame({("mape", "mean"): {"sarima": 9.0, "seasonal_naive": 8.0, "naive": 20.0}})
    assert select_best(summary) == "seasonal_naive"


def test_end_to_end_training(messy_csv, tmp_path):
    res = run(str(messy_csv), str(tmp_path / "models"), n_folds=2)
    bundle = load_bundle(res["bundle_path"])
    out = forecast(bundle, 6)
    assert len(out) == 6 and out[0]["month"] == "2023-01"
    with pytest.raises(ValueError):
        forecast(bundle, 0)
