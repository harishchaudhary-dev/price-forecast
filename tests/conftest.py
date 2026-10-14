import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def monthly_series():
    idx = pd.date_range("2015-01-01", periods=96, freq="MS")
    rng = np.random.default_rng(0)
    t = np.arange(96)
    y = 1000 * np.exp(0.004 * t + 0.25 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.03, 96))
    return pd.Series(y, index=idx, name="price")


@pytest.fixture
def messy_csv(tmp_path, monthly_series):
    rows = [{"Arrival_Date": d.strftime("%d/%m/%Y").replace("01/", "15/", 1), "Modal_Price": round(v, 0)}
            for d, v in monthly_series.items()]
    df = pd.DataFrame(rows)
    df = df.drop(index=[10, 11])                    # 2 missing months
    df["Modal_Price"] = df["Modal_Price"].astype(object)
    df.loc[40, "Modal_Price"] = df.loc[40, "Modal_Price"] * 10   # entry error
    df.loc[50, "Modal_Price"] = "N/A"                # unparseable
    df = pd.concat([df, df.iloc[[5]]])               # duplicate
    path = tmp_path / "prices.csv"
    df.to_csv(path, index=False)
    return path
