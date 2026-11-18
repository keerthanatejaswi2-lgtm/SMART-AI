"""Incident intelligence from abnormal traffic behavior (advisory, not confirmed facts)."""
from __future__ import annotations
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
from .config import MODEL_PATH
from .congestion import enrich
from .database import data_quality_report
from .graph_network import ADJACENCY, ROADS
from .traffic import current_frame, snapshot_enriched
TYPE_MAP = {
    "accident": "Possible Accident",
    "crash": "Possible Accident",
    "road_work": "Road Work",
    "work": "Road Work",
    "weather": "Weather-related slowdown",
    "rain": "Weather-related slowdown",
    "event": "Event-driven traffic",
    "concert": "Event-driven traffic",
    "unknown": "Unknown abnormality",
}
def _neighbor_congestion_rise(hist: pd.DataFrame, rid: str, ts) -> bool:
    nbrs = ADJACENCY.get(rid, [])
    if not nbrs:
        return False
    window = hist[hist["timestamp"] <= ts].tail(80)
    prev = window[window["timestamp"] < ts]
    cur = window[window["timestamp"] == ts]
    if prev.empty or cur.empty:
        return False
    rises = 0
    for n in nbrs:
        p = prev[prev["road_id"] == n]["traffic_load"].mean()
        c = cur[cur["road_id"] == n]["traffic_load"].mean()
        if pd.notna(p) and pd.notna(c) and c > p * 1.15:
            rises += 1
    return rises >= 1
def classify_type(row, speed_drop, dens_up, weather, labeled) -> str:
    lab = str(labeled or "").lower()
    for key, val in TYPE_MAP.items():
        if key in lab:
            return val
    if str(row.get("road_work") or "").strip():
        return "Road Work"
    if str(row.get("event") or "").strip():
        return "Event-driven traffic"
    if weather in ("rain", "storm", "fog"):
        return "Weather-related slowdown"
    if speed_drop >= 55 and dens_up >= 30:
        return "Possible Accident"
    if speed_drop >= 35:
        return "Unknown abnormality"
    return "Unknown abnormality"
def detect_incidents() -> List[dict]:
    df = enrich(current_frame())
    quality = data_quality_report(df)
    if df.empty:
        return []
    snap = snapshot_enriched()
    if snap.empty:
        return []
    ts = snap["timestamp"].max()
    hist = df.sort_values("timestamp")
    alerts = []
    iso_model = None
    iso_path = MODEL_PATH / "anomaly_model.pkl"
    if iso_path.exists():
        try:
            iso_model = joblib.load(iso_path)
        except Exception:
            iso_model = None
    for _, row in snap.iterrows():
        rid = row["road_id"]
        road_hist = hist[hist["road_id"] == rid].sort_values("timestamp")
        if len(road_hist) < 3:
            continue
        recent = road_hist.tail(6)
        prev = recent.iloc[-3]
        cur_speed = float(row["average_speed"] or 0)
        prev_speed = float(prev["average_speed"] or cur_speed)
        cur_load = float(row.get("traffic_load") or 0)
        prev_load = float(prev.get("traffic_load") or cur_load)
        speed_drop = ((prev_speed - cur_speed) / max(prev_speed, 1)) * 100
        dens_up = ((cur_load - prev_load) / max(prev_load, 0.05)) * 100
        neighbor = _neighbor_congestion_rise(hist, rid, ts)
        labeled = str(row.get("incident") or "")
        iso_flag = False
        if iso_model is not None:
            try:
                hour = pd.to_datetime(row["timestamp"]).hour
                feat = np.array([[cur_speed, cur_load, speed_drop, dens_up, hour]])
                iso_flag = int(iso_model.predict(feat)[0]) == -1
            except Exception:
                iso_flag = False
        abnormal = (
            speed_drop >= 28
            and dens_up >= 18
            or (row.get("congestion") == "SEVERE" and speed_drop >= 20)
            or iso_flag
            or bool(labeled.strip())
        )
        if not abnormal:
            continue
        evidence = []
        if speed_drop >= 15:
            evidence.append(f"Speed decreased by {max(speed_drop, 0):.0f}%")
        if dens_up >= 10:
            evidence.append(f"Vehicle density increased by {max(dens_up, 0):.0f}%")
        if neighbor:
            evidence.append("Neighboring road congestion increased")
        if str(row.get("weather") or "") in ("rain", "storm", "fog"):
            evidence.append(f"Weather condition: {row.get('weather')}")
        if str(row.get("event") or "").strip():
            evidence.append(f"Nearby event: {row.get('event')}")
        if not evidence:
            evidence.append("Pattern differs from recent baseline on this corridor")
        itype = classify_type(row, speed_drop, dens_up, str(row.get("weather") or ""), labeled)
        if speed_drop >= 55 or row.get("congestion") == "SEVERE":
            severity = "HIGH"
        elif speed_drop >= 30:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        conf = quality["data_quality"] * 0.7 + min(25, speed_drop / 3) + (5 if neighbor else 0)
        conf = max(55, min(93, conf))
        action = "SIMULATED ADVISORY: evaluate diversion and staging; do not treat as a confirmed crash."
        alerts.append(
            {
                "title": "POSSIBLE INCIDENT",
                "incident_type": itype,
                "road_id": rid,
                "road_name": ROADS.get(rid, {}).get("name", rid),
                "severity": severity,
                "confidence": round(conf, 1),
                "evidence": evidence,
                "timestamp": str(ts),
                "advisory": True,
                "disclaimer": "Uncertain prediction — not a confirmed incident.",
                "recommended_action": action,
                "speed_drop_pct": round(float(speed_drop), 1),
                "density_increase_pct": round(float(dens_up), 1),
            }
        )
    alerts.sort(key=lambda a: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[a["severity"]], reverse=True)
    return alerts
def alert_feed() -> list:
    from .congestion import congestion_table
    from .forecasting import network_forecast
    items = []
    for inc in detect_incidents():
        color = "orange" if inc["severity"] != "HIGH" else "red"
        items.append(
            {
                "color": color,
                "icon": "incident",
                "message": f"Possible incident detected on {inc['road_id']}",
                "timestamp": inc["timestamp"],
                "road": inc["road_id"],
                "severity": inc["severity"],
                "confidence": inc["confidence"],
                "type": inc["incident_type"],
                "evidence": inc["evidence"],
                "advisory": inc["advisory"],
                "disclaimer": inc["disclaimer"],
                "recommended_action": inc["recommended_action"],
            }
        )