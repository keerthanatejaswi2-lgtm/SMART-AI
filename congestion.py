"""Congestion scoring: Traffic Load = Vehicle Count / Road Capacity."""
from __future__ import annotations
import pandas as pd
from .graph_network import classify_load, travel_minutes, ROADS
LEVEL_RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "SEVERE": 3}
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cap = out["road_capacity"].replace(0, pd.NA).fillna(1)
    out["traffic_load"] = (out["vehicle_count"].fillna(0) / cap).astype(float)
    out["congestion"] = [
        classify_load(float(l or 0), float(s or 40))
        for l, s in zip(out["traffic_load"], out["average_speed"].fillna(40))
    ]
    out["congestion_score"] = (out["traffic_load"].clip(0, 2) * 50 + (1 - (out["average_speed"].fillna(40) / 55).clip(0, 1)) * 50).round(1)
    out["vehicle_density"] = out["traffic_load"].round(3)
    return out
def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ts = df["timestamp"].max()
    snap = df[df["timestamp"] == ts].copy()
    return enrich(snap)
def speed_reduction(row: pd.Series) -> float:
    ff = ROADS.get(str(row["road_id"]), {}).get("ff_speed", 50)
    spd = float(row.get("average_speed") or ff)
    return round(max(0, (ff - spd) / ff * 100), 1)
def congestion_table(df: pd.DataFrame) -> list:
    snap = latest_snapshot(df)
    if snap.empty:
        return []
    hist = enrich(df)
    rows = []
    for _, row in snap.iterrows():
        rid = row["road_id"]
        road_hist = hist[hist["road_id"] == rid]
        hist_avg = float(road_hist["congestion_score"].mean()) if len(road_hist) else 0
        ff = ROADS.get(rid, {}).get("ff_speed", 50)
        length = ROADS.get(rid, {}).get("length_km", 1.2)
        rows.append(
            {
                "road_id": rid,
                "name": ROADS.get(rid, {}).get("name", rid),
                "congestion": row["congestion"],
                "congestion_score": float(row["congestion_score"]),
                "traffic_load": round(float(row["traffic_load"]), 3),
                "vehicle_density": float(row["vehicle_density"]),
                "vehicle_count": float(row["vehicle_count"] or 0),
                "road_capacity": float(row["road_capacity"] or 0),
                "average_speed": float(row["average_speed"] or 0),
                "speed_reduction_pct": speed_reduction(row),
                "travel_time_min": travel_minutes(length, row["average_speed"] or ff),
                "historical_avg_score": round(hist_avg, 1),
                "vs_history": round(float(row["congestion_score"]) - hist_avg, 1),
                "latitude": float(row["latitude"]) if pd.notna(row["latitude"]) else None,
                "longitude": float(row["longitude"]) if pd.notna(row["longitude"]) else None,
            }
        )
    rows.sort(key=lambda r: r["congestion_score"], reverse=True)
    return rows
def congestion_series(df: pd.DataFrame, road_id: str | None = None, limit: int = 48) -> list:
    hist = enrich(df)
    if road_id:
        hist = hist[hist["road_id"] == road_id]
    if hist.empty:
        return []
    grouped = (
        hist.groupby("timestamp")
        .agg(
            congestion_score=("congestion_score", "mean"),
            average_speed=("average_speed", "mean"),
            traffic_load=("traffic_load", "mean"),
        )
        .reset_index()
        .sort_values("timestamp")
        .tail(limit)
    )
    return [
        {
            "timestamp": str(r["timestamp"]),
            "congestion_score": round(float(r["congestion_score"]), 1),
            "average_speed": round(float(r["average_speed"] or 0), 1),
            "traffic_load": round(float(r["traffic_load"]), 3),
        }
        for _, r in grouped.iterrows()
    ]