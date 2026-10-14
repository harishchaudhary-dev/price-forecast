"""Generate a SYNTHETIC agri-price dataset for development and tests ONLY.

This is NOT the assignment dataset. It exists so the pipeline, API and tests
run end to end before the real CSV is dropped in. It deliberately includes the
problems real mandi data has: several reports per month, missing months, a
data-entry outlier, duplicates, non-numeric prices and comma-formatted numbers.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
rows = []
months = pd.date_range("2013-01-01", "2025-12-01", freq="MS")
for commodity, base, amp, trend in [("Onion", 1400, 0.30, 0.055), ("Tomato", 1100, 0.40, 0.04)]:
    shock = 0.0
    for i, m in enumerate(months):
        shock = 0.7 * shock + rng.normal(0, 0.06)
        season = amp * np.sin(2 * np.pi * (m.month - 4) / 12)
        level = base * np.exp(trend * i / 12 + season + shock)
        if commodity == "Onion" and m.year == 2019 and m.month in (10, 11, 12):
            level *= 1.8  # genuine price spike (supply shock) - must NOT be removed
        for _ in range(rng.integers(2, 6)):
            day = int(rng.integers(1, 28))
            price = level * np.exp(rng.normal(0, 0.04))
            rows.append({"Commodity": commodity, "Market": "Bangalore",
                         "Arrival_Date": f"{day:02d}/{m.month:02d}/{m.year}",
                         "Modal_Price": round(price, 0)})

df = pd.DataFrame(rows)
# Missing months
for gap in ["/03/2016", "/04/2016", "/08/2021"]:
    df = df[~((df["Commodity"] == "Onion") & df["Arrival_Date"].str.endswith(gap))]
# Data-entry error (extra zero)
idx = df[(df["Commodity"] == "Onion") & df["Arrival_Date"].str.endswith("/06/2018")].index
df.loc[idx, "Modal_Price"] = df.loc[idx, "Modal_Price"] * 10
df["Modal_Price"] = df["Modal_Price"].astype(object)
df.loc[df.sample(6, random_state=1).index, "Modal_Price"] = "N/A"
fmt = df.sample(40, random_state=2).index
df.loc[fmt, "Modal_Price"] = df.loc[fmt, "Modal_Price"].map(lambda v: f"{float(v):,.0f}" if v != "N/A" else v)
df = pd.concat([df, df.sample(15, random_state=3)])  # duplicates

out = Path(__file__).parent / "sample_prices_SYNTHETIC.csv"
df.to_csv(out, index=False)
print(f"wrote {len(df)} rows -> {out}")
