from pydantic import BaseModel, Field

from forecasting.predict import MAX_HORIZON


class ForecastRequest(BaseModel):
    horizon: int = Field(12, ge=1, le=MAX_HORIZON, description="Months ahead to forecast")


class ForecastPoint(BaseModel):
    month: str
    predicted_price: float
    lower_95: float
    upper_95: float
    change_vs_last_actual_pct: float


class ForecastResponse(BaseModel):
    model_version: str
    last_actual_month: str
    forecast: list[ForecastPoint]


class ModelInfo(BaseModel):
    version: str
    model: str
    trained_at: str
    train_start: str
    train_end: str
    backtest_mape: float
