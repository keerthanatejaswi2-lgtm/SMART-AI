"""15–60 minute traffic forecasts using trained models with heuristic fallback."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_PATH
from .congestion import enrich
from .database import data_quality_report
from .graph_network import ROADS, classify_load
from .traffic import current_frame, snapshot_enriched
HORIZONS = [15, 30, 45, 60]
def _features_row(row, hour, dow, horizon) -> np.ndarray:
    weather_rain = 1 if str(row.get("weather") or "") in ("rain", "storm", "fog") else 0
    event = 1 if str(row.get("event") or "").strip() else 0
    work = 1 if str(row.get("road_work") or "").strip() else 0
    load = float(row.get("traffic_load") or 0)
    speed = float(row.get("average_speed") or 30)
    count = float(row.get("vehicle_count") or 0)
    cap = float(row.get("road_capacity") or 200)
    return np.array(
        [[hour, dow, horizon, speed, count, cap, load, weather_rain, event, work]],
        dtype=float,
    )
def _load_forecast_model():
    path = MODEL_PATH / "forecast_model.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None
def _heuristic_speed(speed: float, load: float, hour: int, horizon: int, weather: str, event: str) -> float:
    s = speed
    # Peak-hour decay then slight recovery after 45–60
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        s *= 0.92 if horizon <= 30 else 0.88
        if horizon >= 60:
            s *= 1.12
    else:
        s *= 1.02 if horizon >= 45 else 0.98
    if weather in ("rain", "storm", "fog"):
        s *= 0.9
    if event:
        s *= 0.93
    if load > 1:
        s *= 0.9
    if horizon == 60 and load > 0.9:
        s *= 1.08  # dissipation
    return float(np.clip(s, 6, 70))
def forecast_road(road_id: str) -> dict:
    df = enrich(current_frame())
    quality = data_quality_report(df)
    snap = snapshot_enriched()
    if snap.empty:
        return {"error": "Traffic data could not be loaded. Please upload a valid CSV file."}
    row = snap[snap["road_id"] == road_id]
    if row.empty:
        row = snap.iloc[[0]]
        road_id = row.iloc[0]["road_id"]
    rec = row.iloc[0]
    ts = pd.to_datetime(rec["timestamp"])
    hour, dow = int(ts.hour), int(ts.dayofweek)
    model = _load_forecast_model()
    points = [{"horizon_min": 0, "predicted_speed": round(float(rec["average_speed"] or 0), 1), "congestion": rec.get("congestion", "LOW")}]
    used_model = False
    for h in HORIZONS:
        pred = None
        if model is not None:
            try:
                pred = float(model.predict(_features_row(rec, hour, dow, h))[0])
                used_model = True
            except Exception:
                pred = None
        if pred is None:
            pred = _heuristic_speed(
                float(rec["average_speed"] or 20),
                float(rec.get("traffic_load") or 0.7),
                hour,
                h,
                str(rec.get("weather") or ""),
                str(rec.get("event") or ""),
            )
        load = float(rec.get("traffic_load") or 0.7) * (1.05 if h <= 30 else 0.97)
        level = classify_load(load, pred)
        points.append({"horizon_min": h, "predicted_speed": round(float(np.clip(pred, 5, 80)), 1), "congestion": level})
    conf = quality["data_quality"] * 0.9
    if not used_model:
        conf *= 0.9
    conf = round(max(55, min(94, conf)), 1)
    return {
        "road_id": road_id,
        "road_name": ROADS.get(road_id, {}).get("name", road_id),
        "current_speed": points[0]["predicted_speed"],
        "horizons": points,
        "forecast_confidence": conf,
        "data_quality": quality["data_quality"],
        "model": "xgboost/random_forest" if used_model else "heuristic_fallback",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": "Forecast is a decision-support estimate, not a guaranteed outcome.",
    }
def network_forecast() -> dict:
    snap = snapshot_enriched()
    quality = data_quality_report()
    if snap.empty:
        return {"error": "Traffic data could not be loaded. Please upload a valid CSV file."}
    # Focus on most congested road
    snap = snap.sort_values("congestion_score", ascending=False)
    focus = str(snap.iloc[0]["road_id"])
    series = []
    worsening = 0
    improving = 0
    by_horizon = {h: [] for h in [0] + HORIZONS}
    roads = []
    for rid in snap["road_id"].tolist():
        fc = forecast_road(rid)
        roads.append(fc)
        speeds = {p["horizon_min"]: p["predicted_speed"] for p in fc["horizons"]}
        if speeds[30] < speeds[0] - 1:
            worsening += 1
        if speeds[60] > speeds[30]:
            improving += 1
        for h, v in speeds.items():
            by_horizon[h].append(v)
    network_line = [{"horizon_min": h, "predicted_speed": round(float(np.mean(v)), 1)} for h, v in by_horizon.items()]
    outlook = "worsening" if worsening >= max(1, len(roads) // 3) else "stable"
    if improving > worsening:
        outlook = "recovering"
    return {
        "focus_road": focus,
        "focus": forecast_road(focus),
        "network_line": network_line,
        "roads": roads,
        "outlook": outlook,
        "forecast_confidence": round(min(94, quality["data_quality"] * 0.92), 1),
        "data_quality": quality["data_quality"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }