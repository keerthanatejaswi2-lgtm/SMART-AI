"""Traffic snapshots, KPIs, search/filter, and map payloads."""
from __future__ import annotations
from datetime import datetime
from typing import Optional
import pandas as pd
from .congestion import congestion_table, enrich, latest_snapshot
from .database import data_quality_report, get_meta, load_frame
from .graph_network import ROADS, graph_payload, travel_minutes
DEMO_STATE = {
    "active": False,
    "step": 0,
    "started_at": None,
    "overlay": {},
}
def apply_demo_overlay(snap: pd.DataFrame) -> pd.DataFrame:
    if snap.empty or not DEMO_STATE.get("active"):
        return snap
    overlay = DEMO_STATE.get("overlay") or {}
    out = snap.copy()
    for rid, vals in overlay.items():
        mask = out["road_id"] == rid
        for k, v in vals.items():
            if k in out.columns:
                out.loc[mask, k] = v
    return out
def current_frame() -> pd.DataFrame:
    df = load_frame()
    return df
def snapshot_enriched() -> pd.DataFrame:
    df = current_frame()
    snap = latest_snapshot(df)
    return enrich(apply_demo_overlay(snap))
def kpis() -> dict:
    snap = snapshot_enriched()
    quality = data_quality_report()
    if snap.empty:
        return {
            "active_road_segments": 0,
            "congested_roads": 0,
            "active_incidents": 0,
            "average_speed": 0,
            "average_travel_time": 0,
            "prediction_confidence": 0,
            "system_status": "ATTENTION REQUIRED",
            "data_quality": quality["data_quality"],
            "last_data_update": None,
            "prediction_updated": None,
            "message": "Traffic data could not be loaded. Please upload a valid CSV file.",
        }
    congested = int((snap["congestion"].isin(["HIGH", "SEVERE"])).sum())
    incidents = 0
    if "incident" in snap.columns:
        incidents = int(snap["incident"].fillna("").astype(str).str.len().gt(0).sum())
    # heuristic extra incidents from severe+sudden handled by incidents module later
    avg_speed = float(snap["average_speed"].mean())
    times = []
    for _, row in snap.iterrows():
        length = ROADS.get(row["road_id"], {}).get("length_km", 1.2)
        times.append(travel_minutes(length, row["average_speed"] or 20))
    avg_tt = sum(times) / max(len(times), 1)
    conf = min(95, quality["data_quality"] * 0.92 + 8)
    if DEMO_STATE.get("active"):
        conf = max(conf, 86)
    severe = int((snap["congestion"] == "SEVERE").sum())
    if severe >= 3 or incidents >= 3:
        status = "CRITICAL TRAFFIC CONDITION"
    elif congested >= 1 or incidents >= 1:
        status = "ATTENTION REQUIRED"
    else:
        status = "SYSTEM NORMAL"
    last_ts = str(snap["timestamp"].max())
    return {
        "active_road_segments": int(snap["road_id"].nunique()),
        "congested_roads": congested,
        "active_incidents": incidents,
        "average_speed": round(avg_speed, 1),
        "average_travel_time": round(avg_tt, 1),
        "prediction_confidence": round(conf, 1),
        "system_status": status,
        "data_quality": quality["data_quality"],
        "missing_records": quality["missing_records"],
        "anomalies": quality["anomalies"],
        "last_data_update": last_ts,
        "prediction_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_ingest": get_meta("last_ingest"),
        "demo_active": DEMO_STATE["active"],
        "demo_step": DEMO_STATE["step"],
        "message": "ok",
    }
def map_roads() -> list:
    table = congestion_table(current_frame())
    snap = snapshot_enriched()
    incident_map = {}
    if not snap.empty:
        for _, row in snap.iterrows():
            incident_map[row["road_id"]] = str(row.get("incident") or "")
    quality = data_quality_report()
    out = []
    for row in table:
        meta = ROADS.get(row["road_id"], {})
        inc = incident_map.get(row["road_id"], "")
        out.append(
            {
                **row,
                "polyline": meta.get("polyline", []),
                "incident_status": inc if inc else ("Possible incident" if row["congestion"] == "SEVERE" and row["speed_reduction_pct"] > 50 else "None"),
                "confidence": round(min(96, quality["data_quality"] - (4 if row["congestion"] == "SEVERE" else 0)), 1),
                "corridor": meta.get("corridor"),
                "free_flow_speed": meta.get("ff_speed"),
                "length_km": meta.get("length_km"),
            }
        )
    return out
def filters(df: Optional[pd.DataFrame], road_id=None, congestion=None, incident=None, severity=None, time_from=None, time_to=None) -> pd.DataFrame:
    if df is None:
        df = enrich(current_frame())
    else:
        df = enrich(df)
    if road_id:
        df = df[df["road_id"].str.upper() == str(road_id).upper()]
    if congestion:
        df = df[df["congestion"].str.upper() == str(congestion).upper()]
    if severity:
        df = df[df["congestion"].str.upper() == str(severity).upper()]
    if incident in ("yes", "true", "1"):
        df = df[df["incident"].fillna("").astype(str).str.len() > 0]
    if incident in ("no", "false", "0"):
        df = df[df["incident"].fillna("").astype(str).str.len() == 0]
    if time_from:
        df = df[df["timestamp"] >= pd.to_datetime(time_from)]
    if time_to:
        df = df[df["timestamp"] <= pd.to_datetime(time_to)]
    return df
def search_road(q: str) -> dict:
    qn = (q or "").strip().upper()
    roads = map_roads()
    matches = [r for r in roads if qn in r["road_id"] or qn in str(r.get("name", "")).upper()]
    hist = enrich(current_frame())
    detail = None
    series = []
    if matches:
        rid = matches[0]["road_id"]
        detail = matches[0]
        sub = hist[hist["road_id"] == rid].tail(36)
        series = [
            {
                "timestamp": str(r["timestamp"]),
                "average_speed": float(r["average_speed"] or 0),
                "vehicle_count": float(r["vehicle_count"] or 0),
                "congestion": r["congestion"],
            }
            for _, r in sub.iterrows()
        ]
    return {"query": q, "matches": matches, "detail": detail, "history": series}
def network() -> dict:
    snap = snapshot_enriched()
    payload = graph_payload(snap.set_index("road_id").to_dict(orient="index"))
    return payload