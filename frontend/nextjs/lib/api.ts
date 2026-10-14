// Typed client for the forecast API. Types mirror the Pydantic/DRF schemas.
// In production, generate these from OpenAPI with `openapi-typescript` instead of hand-writing them.

export interface ForecastPoint {
  month: string;              // "YYYY-MM"
  predicted_price: number;
  lower_95: number;
  upper_95: number;
  change_vs_last_actual_pct: number;
}

export interface ForecastResponse {
  model_version: string;
  model?: string;
  last_actual_month: string;
  backtest_mape?: number;
  forecast: ForecastPoint[];
}

// Public, logged endpoint (Django). Never put the FastAPI API key in client-side code.
const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export async function getForecast(horizon = 12, signal?: AbortSignal): Promise<ForecastResponse> {
  if (horizon < 1 || horizon > 24) throw new ApiError(400, "horizon must be 1-24");
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10_000);
  try {
    const res = await fetch(`${BASE_URL}/api/forecast/?horizon=${horizon}`, {
      signal: signal ?? controller.signal,
      next: { revalidate: 3600 },   // forecasts change monthly; cache for an hour
    } as RequestInit);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new ApiError(res.status, body.detail ?? `Request failed (${res.status})`);
    }
    return (await res.json()) as ForecastResponse;
  } finally {
    clearTimeout(timeout);
  }
}
