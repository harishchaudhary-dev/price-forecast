// React Native screen consuming the same Django endpoint.
// Deps: @tanstack/react-query (caching, retries, offline display of last good data).
import React, { useState } from "react";
import { ActivityIndicator, FlatList, Pressable, RefreshControl, StyleSheet, Text, View } from "react-native";
import { useQuery } from "@tanstack/react-query";

// Use your machine's LAN IP for a physical device; 10.0.2.2 reaches localhost from the Android emulator.
const API_URL = process.env.EXPO_PUBLIC_API_URL ?? "http://10.0.2.2:8000";

type Point = { month: string; predicted_price: number; lower_95: number; upper_95: number; change_vs_last_actual_pct: number };
type Forecast = { model_version: string; last_actual_month: string; backtest_mape?: number; forecast: Point[] };

async function fetchForecast(horizon: number): Promise<Forecast> {
  const res = await fetch(`${API_URL}/api/forecast/?horizon=${horizon}`);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? `HTTP ${res.status}`);
  return res.json();
}

export default function ForecastScreen() {
  const [horizon, setHorizon] = useState(12);
  const { data, error, isLoading, refetch, isRefetching } = useQuery({
    queryKey: ["forecast", horizon],
    queryFn: () => fetchForecast(horizon),
    staleTime: 60 * 60 * 1000,
    retry: 2,
  });

  if (isLoading) return <ActivityIndicator style={{ marginTop: 48 }} />;
  if (error && !data) return <Text style={styles.error}>Could not load forecast: {(error as Error).message}</Text>;

  return (
    <View style={styles.container}>
      <View style={styles.row}>
        {[6, 12, 24].map((h) => (
          <Pressable key={h} onPress={() => setHorizon(h)} style={[styles.chip, h === horizon && styles.chipActive]}>
            <Text style={h === horizon ? styles.chipTextActive : undefined}>{h}m</Text>
          </Pressable>
        ))}
      </View>
      <Text style={styles.meta}>
        Data to {data!.last_actual_month}{data!.backtest_mape ? ` · typical error ±${data!.backtest_mape.toFixed(1)}%` : ""}
      </Text>
      <FlatList
        data={data!.forecast}
        keyExtractor={(p) => p.month}
        refreshControl={<RefreshControl refreshing={isRefetching} onRefresh={refetch} />}
        renderItem={({ item }) => (
          <View style={styles.item}>
            <Text style={styles.month}>{item.month}</Text>
            <View style={{ alignItems: "flex-end" }}>
              <Text style={styles.price}>₹{item.predicted_price.toLocaleString("en-IN")}</Text>
              <Text style={styles.range}>₹{item.lower_95.toFixed(0)} – ₹{item.upper_95.toFixed(0)}</Text>
            </View>
          </View>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16 },
  row: { flexDirection: "row", gap: 8, marginBottom: 8 },
  chip: { paddingHorizontal: 14, paddingVertical: 6, borderRadius: 16, borderWidth: 1, borderColor: "#ccc" },
  chipActive: { backgroundColor: "#d97706", borderColor: "#d97706" },
  chipTextActive: { color: "white" },
  meta: { color: "#666", marginBottom: 8 },
  item: { flexDirection: "row", justifyContent: "space-between", paddingVertical: 12, borderBottomWidth: 1, borderColor: "#eee" },
  month: { fontSize: 16 },
  price: { fontSize: 16, fontWeight: "600" },
  range: { color: "#888", fontSize: 12 },
  error: { color: "crimson", margin: 24 },
});
