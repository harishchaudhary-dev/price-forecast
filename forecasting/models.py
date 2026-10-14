"""Candidate forecasting models behind one interface.

Every model:
  fit(y)            -> self      (y: monthly pd.Series with a DatetimeIndex)
  fitted()          -> pd.Series (in-sample one-step predictions, for train metrics)
  predict(h)        -> DataFrame [mean, lower, upper] indexed by future months

All models work on log prices. Prices are positive and change multiplicatively,
so log space stabilises variance and keeps forecasts and intervals positive.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

Z95 = 1.96


def _future_index(y: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def _frame(idx, log_mean, log_sd) -> pd.DataFrame:
    log_mean, log_sd = np.asarray(log_mean, float), np.asarray(log_sd, float)
    return pd.DataFrame({
        "mean": np.exp(log_mean),
        "lower": np.exp(log_mean - Z95 * log_sd),
        "upper": np.exp(log_mean + Z95 * log_sd),
    }, index=idx)


class BaseModel:
    name = "base"
    min_obs = 3

    def fit(self, y: pd.Series):
        raise NotImplementedError

    def fitted(self) -> pd.Series:
        raise NotImplementedError

    def predict(self, h: int) -> pd.DataFrame:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class NaiveModel(BaseModel):
    """Baseline: next month equals this month. Any real model must beat it."""
    name = "naive"

    def fit(self, y):
        self.y = y
        self.log_y = np.log(y)
        self.sd = self.log_y.diff().std()
        return self

    def fitted(self):
        return self.y.shift(1).dropna()

    def predict(self, h):
        steps = np.arange(1, h + 1)
        return _frame(_future_index(self.y, h), np.repeat(self.log_y.iloc[-1], h), self.sd * np.sqrt(steps))


class SeasonalNaiveModel(BaseModel):
    """Baseline: next month equals the same month last year."""
    name = "seasonal_naive"
    min_obs = 24

    def fit(self, y):
        self.y = y
        self.log_y = np.log(y)
        self.sd = (self.log_y - self.log_y.shift(12)).std()
        return self

    def fitted(self):
        return self.y.shift(12).dropna()

    def predict(self, h):
        last_year = self.log_y.iloc[-12:].to_numpy()
        means = np.array([last_year[i % 12] for i in range(h)])
        cycles = np.floor(np.arange(h) / 12) + 1
        return _frame(_future_index(self.y, h), means, self.sd * np.sqrt(cycles))


class HoltWintersModel(BaseModel):
    """Exponential smoothing with damped additive trend and additive (log) seasonality."""
    name = "holt_winters"
    min_obs = 24

    def fit(self, y):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.y = y
        log_y = np.log(y).reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = ExponentialSmoothing(
                log_y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12,
                initialization_method="estimated").fit(optimized=True)
        self.sd = float(np.std(self.res.resid, ddof=1))
        return self

    def fitted(self):
        return pd.Series(np.exp(self.res.fittedvalues.to_numpy()), index=self.y.index)

    def predict(self, h):
        mean = self.res.forecast(h).to_numpy()
        # Approximate interval: residual sd growing with sqrt(horizon).
        return _frame(_future_index(self.y, h), mean, self.sd * np.sqrt(np.arange(1, h + 1)))


class SarimaModel(BaseModel):
    """Seasonal ARIMA on log prices, orders chosen by AIC over a small grid."""
    name = "sarima"
    min_obs = 36

    def __init__(self, order=None, seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order

    @staticmethod
    def select_order(y: pd.Series, max_pq: int = 2) -> tuple[tuple, tuple]:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        log_y = np.log(y).reset_index(drop=True)
        best, best_aic = None, np.inf
        for p, q, P, Q in itertools.product(range(max_pq + 1), range(max_pq + 1), (0, 1), (0, 1)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(log_y, order=(p, 1, q), seasonal_order=(P, 1, Q, 12),
                                  enforce_stationarity=False, enforce_invertibility=False
                                  ).fit(disp=False, maxiter=200)
                if res.aic < best_aic:
                    best, best_aic = ((p, 1, q), (P, 1, Q, 12)), res.aic
            except Exception:
                continue
        return best or ((1, 1, 1), (0, 1, 1, 12))

    def fit(self, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self.y = y
        if self.order is None:
            self.order, self.seasonal_order = self.select_order(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(np.log(y).reset_index(drop=True), order=self.order,
                               seasonal_order=self.seasonal_order, enforce_stationarity=False,
                               enforce_invertibility=False).fit(disp=False, maxiter=200)
        return self

    def fitted(self):
        burn = 13  # first year + 1 of differenced fitted values are meaningless
        vals = np.exp(self.res.fittedvalues.to_numpy())
        return pd.Series(vals, index=self.y.index).iloc[burn:]

    def predict(self, h):
        fc = self.res.get_forecast(h)
        mean = fc.predicted_mean.to_numpy()
        sd = np.sqrt(np.asarray(fc.var_pred_mean))
        return _frame(_future_index(self.y, h), mean, sd)

    def describe(self):
        return f"sarima{self.order}x{self.seasonal_order}"


class GradientBoostingModel(BaseModel):
    """Gradient-boosted trees on lag features, forecasting recursively.

    The target is the month-over-month log change rather than the level,
    because tree models cannot extrapolate a level they never saw in training.
    """
    name = "gradient_boosting"
    min_obs = 36
    LAGS = (1, 2, 3, 6, 12)

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _features(self, log_y: np.ndarray, t: int, month: int) -> list[float]:
        diffs = np.diff(log_y[: t])  # diffs[k] = change into month k+1
        feats = [diffs[-lag] for lag in self.LAGS]
        feats += [diffs[-3:].mean(), diffs[-12:].mean(), log_y[t - 1] - log_y[t - 13]]
        feats += [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]
        return feats

    def fit(self, y):
        self.y = y
        self.log_y = np.log(y.to_numpy())
        months = y.index.month.to_numpy()
        start = max(self.LAGS) + 1
        X = [self._features(self.log_y, t, months[t]) for t in range(start, len(y))]
        target = self.log_y[start:] - self.log_y[start - 1:-1]
        self.model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, min_samples_leaf=5,
            random_state=self.random_state).fit(np.array(X), target)
        self.train_start = start
        self.resid_sd = float(np.std(target - self.model.predict(np.array(X)), ddof=1))
        self.X_train = np.array(X)
        return self

    def fitted(self):
        pred_diff = self.model.predict(self.X_train)
        vals = np.exp(self.log_y[self.train_start - 1:-1] + pred_diff)
        return pd.Series(vals, index=self.y.index[self.train_start:])

    def predict(self, h):
        idx = _future_index(self.y, h)
        path = list(self.log_y)
        for d in idx:
            diff = self.model.predict(np.array([self._features(np.array(path), len(path), d.month)]))[0]
            path.append(path[-1] + diff)
        # In-sample residuals understate out-of-sample error; inflate by 1.5x.
        sd = 1.5 * self.resid_sd * np.sqrt(np.arange(1, h + 1))
        return _frame(idx, path[-h:], sd)


REGISTRY = {
    "naive": NaiveModel,
    "seasonal_naive": SeasonalNaiveModel,
    "holt_winters": HoltWintersModel,
    "sarima": SarimaModel,
    "gradient_boosting": GradientBoostingModel,
}
BASELINES = ("naive", "seasonal_naive")
