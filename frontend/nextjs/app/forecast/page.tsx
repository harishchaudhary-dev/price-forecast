"use client";
// Forecast page: horizon selector, chart with 95% band, table.
// Deps: next, react, recharts  (npm i recharts)
import { useEffect, useState } from "react";
import { Area, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError, ForecastResponse, getForecast } from "../../lib/api";

export default function ForecastPage() {
  const [horizon, setHorizon] = useState(12);
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    getForecast(horizon, ctrl.signal)
      .then(setData)
      .catch((e) => { if (e.name !== "AbortError") setError(e instanceof ApiError ? e.message : "Network error"); })
      .finally(() => setLoading(false));
    return () => ctrl.abort();
  }, [horizon]);

  const rows = data?.forecast.map((p) => ({ ...p, band: [p.lower_95, p.upper_95] })) ?? [];

  return (
    <main style={{ maxWidth: 960, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1>Monthly price forecast</h1>
      <label>
        Horizon:{" "}
        <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
          {[3, 6, 12, 18, 24].map((h) => <option key={h} value={h}>{h} months</option>)}
        </select>
      </label>

      {loading && <p>Loading…</p>}
      {error && <p role="alert" style={{ color: "crimson" }}>{error}</p>}

      {data && !loading && (
        <>
          <p style={{ color: "#666" }}>
            Model {data.model} · v{data.model_version} · data to {data.last_actual_month}
            {data.backtest_mape != null && ` · typical error ±${data.backtest_mape.toFixed(1)}%`}
          </p>
          <div style={{ height: 320 }}>
            <ResponsiveContainer>
              <ComposedChart data={rows}>
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip />
                <Area dataKey="band" stroke="none" fill="#f59e0b" fillOpacity={0.2} name="95% range" />
                <Line dataKey="predicted_price" stroke="#d97706" strokeWidth={2} dot={false} name="Forecast" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 16 }}>
            <thead><tr><th align="left">Month</th><th align="right">Forecast</th><th align="right">95% range</th><th align="right">vs last actual</th></tr></thead>
            <tbody>
              {data.forecast.map((p) => (
                <tr key={p.month} style={{ borderTop: "1px solid #eee" }}>
                  <td>{p.month}</td>
                  <td align="right">₹{p.predicted_price.toLocaleString("en-IN")}</td>
                  <td align="right">₹{p.lower_95.toLocaleString("en-IN")} – ₹{p.upper_95.toLocaleString("en-IN")}</td>
                  <td align="right" style={{ color: p.change_vs_last_actual_pct >= 0 ? "#b45309" : "#047857" }}>
                    {p.change_vs_last_actual_pct > 0 ? "+" : ""}{p.change_vs_last_actual_pct.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </main>
  );
}
