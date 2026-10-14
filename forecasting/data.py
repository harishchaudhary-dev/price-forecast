"""Data loading and cleaning for monthly price series.

The loader is deliberately schema-tolerant: the task's data dictionary was not
available when this was written, so date and price columns are auto-detected
and can be overridden explicitly. Once you have the real data dictionary,
pass the exact column names instead of relying on detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

DATE_HINTS = ("date", "month", "period", "time", "year")
PRICE_HINTS = ("modal", "average", "avg", "price", "value", "rate")


@dataclass
class CleaningReport:
    raw_rows: int = 0
    date_col: str = ""
    price_col: str = ""
    filters: dict = field(default_factory=dict)
    dropped_unparseable_dates: int = 0
    dropped_invalid_prices: int = 0
    dropped_non_positive: int = 0
    dropped_duplicates: int = 0
    raw_rows_per_month_max: int = 0
    months_total: int = 0
    months_missing: int = 0
    missing_months: list = field(default_factory=list)
    outliers_replaced: int = 0
    outlier_months: list = field(default_factory=list)
    start: str = ""
    end: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _pick(columns, hints, exclude=()):
    lowered = {c: c.lower() for c in columns if c not in exclude}
    for hint in hints:
        for col, low in lowered.items():
            if hint in low:
                return col
    return None


def detect_columns(df: pd.DataFrame, date_col: str | None = None,
                   price_col: str | None = None) -> tuple[str, str]:
    """Return (date_col, price_col), auto-detecting any that are not given."""
    date_col = date_col or _pick(df.columns, DATE_HINTS)
    if date_col is None:
        raise ValueError(f"Could not detect a date column in {list(df.columns)}; pass date_col.")
    price_col = price_col or _pick(df.columns, PRICE_HINTS, exclude=(date_col,))
    if price_col is None:
        numeric = df.drop(columns=[date_col]).select_dtypes("number").columns
        if len(numeric) == 0:
            raise ValueError("Could not detect a price column; pass price_col.")
        price_col = numeric[-1]
    return date_col, price_col


# Day-first formats come before month-first ones: Indian government sources
# (Agmarknet etc.) use dd/mm/yyyy, so when a column is ambiguous we assume it.
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%m/%d/%Y", "%Y/%m/%d",
                "%d-%b-%Y", "%d %b %Y", "%Y-%m", "%b-%Y", "%b %Y", "%B %Y", "%b-%y", "%m/%Y")


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse the whole column with ONE format.

    Per-row format inference (pandas format="mixed") is a silent data bug:
    "05/03/2016" becomes May 3 while "25/03/2016" becomes March 25, so the
    same column ends up with two conventions. Pick the format that parses the
    most rows and apply it uniformly.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    text = s.astype(str).str.strip()
    best, best_rate = None, -1.0
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        rate = parsed.notna().mean()
        if rate > best_rate:
            best, best_rate = parsed, rate
        if rate == 1.0:
            break
    if best_rate < 0.9:  # unknown format: last resort, but consistently day-first
        best = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return best


def _parse_prices(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def load_raw(path: str, filters: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col, val in (filters or {}).items():
        if col not in df.columns:
            raise KeyError(f"Filter column '{col}' not in data: {list(df.columns)}")
        df = df[df[col].astype(str).str.strip().str.lower() == str(val).strip().lower()]
    if df.empty:
        raise ValueError("No rows left after applying filters.")
    return df


def detect_outliers(log_y: pd.Series, n_sigmas: float = 5.0, min_jump: float = 0.5) -> pd.Series:
    """Boolean mask of isolated anomalies, judged on STL residuals.

    Raw rolling-median filters (Hampel) misfire on agri prices: strong seasonal
    peaks look like outliers relative to a 7-month median. So we first remove
    trend and seasonality with robust STL, then flag residuals beyond
    n_sigmas robust standard deviations (MAD-based, so the outliers themselves
    do not inflate the threshold the way a mean/std rule would).

    A point must ALSO deviate from its expected value by at least `min_jump`
    (0.5 = 50%). Robust STL leaves a very tight residual spread, so a z-score
    alone flags ordinary volatility; an isolated one-month 50%+ deviation that
    immediately reverses is far more likely an entry error (extra zero, wrong
    unit, per-kg vs per-quintal) than a market move.
    """
    from statsmodels.tsa.seasonal import STL

    filled = log_y.interpolate(limit_direction="both")
    if len(filled) >= 24:
        resid = STL(filled, period=12, robust=True).fit().resid
    else:  # too short for seasonal decomposition
        resid = filled - filled.rolling(5, center=True, min_periods=2).median()
    resid = resid.where(log_y.notna())
    mad = (resid - resid.median()).abs().median()
    z = (resid - resid.median()) / max(1.4826 * mad, 1e-9)
    flagged = ((z.abs() > n_sigmas) & (resid.abs() > np.log1p(min_jump))).fillna(False)
    # Keep sustained moves: two or more consecutive flagged months look like a
    # real market event (supply shock), not a data-entry error. Agri prices DO
    # spike; deleting real spikes makes the model blind to the risk you care about.
    neighbour = flagged.shift(1, fill_value=False) | flagged.shift(-1, fill_value=False)
    return flagged & ~neighbour


def clean_series(df: pd.DataFrame, date_col: str | None = None, price_col: str | None = None,
                 filters: dict | None = None, outlier_sigmas: float = 5.0,
                 max_gap: int = 3) -> tuple[pd.Series, CleaningReport]:
    """Turn raw rows into a clean, gap-free monthly average price series.

    Steps: parse -> drop invalid -> dedupe -> aggregate to monthly mean ->
    reindex to a full monthly calendar -> replace outliers -> interpolate gaps.
    Interpolation happens on the log scale because prices move multiplicatively.
    """
    rep = CleaningReport(raw_rows=len(df), filters=filters or {})
    dcol, pcol = detect_columns(df, date_col, price_col)
    rep.date_col, rep.price_col = dcol, pcol

    work = pd.DataFrame({"date": _parse_dates(df[dcol]), "price": _parse_prices(df[pcol])})
    n = len(work)
    work = work.dropna(subset=["date"])
    rep.dropped_unparseable_dates = n - len(work)
    n = len(work)
    work = work.dropna(subset=["price"])
    rep.dropped_invalid_prices = n - len(work)
    n = len(work)
    work = work[work["price"] > 0]
    rep.dropped_non_positive = n - len(work)
    n = len(work)
    work = work.drop_duplicates()
    rep.dropped_duplicates = n - len(work)
    if work.empty:
        raise ValueError("No valid rows after cleaning.")

    work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
    rep.raw_rows_per_month_max = int(work.groupby("month").size().max())
    monthly = work.groupby("month")["price"].mean().sort_index()

    full_idx = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_idx)
    monthly.index.name = "month"
    missing = monthly[monthly.isna()].index
    rep.months_total = len(monthly)
    rep.months_missing = len(missing)
    rep.missing_months = [d.strftime("%Y-%m") for d in missing]

    log_y = np.log(monthly)
    mask = detect_outliers(log_y, n_sigmas=outlier_sigmas)
    rep.outliers_replaced = int(mask.sum())
    rep.outlier_months = [d.strftime("%Y-%m") for d in monthly[mask].index]
    log_y[mask] = np.nan

    log_y = log_y.interpolate(method="time", limit=max_gap, limit_area="inside")
    if log_y.isna().any():
        # Gaps longer than max_gap or at the edges: fall back to seasonal fill,
        # then nearest value. These are flagged in the report as missing.
        seasonal = log_y.groupby(log_y.index.month).transform("median")
        log_y = log_y.fillna(seasonal).ffill().bfill()

    y = np.exp(log_y).rename("price")
    y.index.freq = "MS"
    rep.start, rep.end = y.index.min().strftime("%Y-%m"), y.index.max().strftime("%Y-%m")
    return y, rep


def load_series(path: str, date_col: str | None = None, price_col: str | None = None,
                filters: dict | None = None, **kwargs) -> tuple[pd.Series, CleaningReport]:
    return clean_series(load_raw(path, filters), date_col, price_col, filters, **kwargs)
