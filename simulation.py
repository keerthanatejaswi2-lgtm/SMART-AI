"""What-If traffic simulator — calculated before/after, not real-world control."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from .congestion import enrich, latest_snapshot
from .graph_network import ROADS, classify_load, k_shortest_paths, path_metrics, travel_minutes
from .traffic import current_frame, snapshot_enriched
SCENARIOS = [
    {"id": "increase_capacity", "label": "Increase road capacity"},
    {"id": "add_alternate_route", "label": "Add alternate route"},
    {"id": "reduce_demand", "label": "Reduce traffic demand"},
    {"id": "road_closure", "label": "Simulate road closure"},
    {"id": "event_traffic", "label": "Simulate event traffic"},
    {"id": "modify_road", "label": "Modify selected road conditions"},
]
def _metrics(snap: pd.DataFrame) -> dict:
    if snap.empty:
        return {
            "average_speed": 0,
            "congestion": "LOW",
            "travel_time": 0,
            "vehicle_queue": 0,
            "congested_roads": 0,
        }
    avg_speed = float(snap["average_speed"].mean())
    loads = []
    times = []
    queue = float(snap["vehicle_count"].sum())
    for _, row in snap.iterrows():
        load = float(row.get("traffic_load") or 0)
        loads.append(load)
        length = ROADS.get(row["road_id"], {}).get("length_km", 1.2)
        times.append(travel_minutes(length, row["average_speed"] or 15))
    avg_load = float(np.mean(loads)) if loads else 0
    level = classify_load(avg_load, avg_speed)
    return {
        "average_speed": round(avg_speed, 1),
        "congestion": level,
        "travel_time": round(float(np.mean(times)), 1),
        "vehicle_queue": int(round(queue)),
        "congested_roads": int((snap["congestion"].isin(["HIGH", "SEVERE"])).sum()) if "congestion" in snap else 0,
        "avg_load": round(avg_load, 3),
    }
def simulate(payload: dict) -> dict:
    scenario = payload.get("scenario") or payload.get("action") or "increase_capacity"
    road_id = (payload.get("road_id") or "R103").upper()
    params = payload.get("params") or {}
    df = current_frame()
    snap = snapshot_enriched()
    if snap.empty:
        snap = latest_snapshot(enrich(df))
    if snap.empty:
        return {"error": "Traffic data could not be loaded. Please upload a valid CSV file."}
    before_df = enrich(snap.copy())
    after = before_df.copy()
    notes = []
    def touch(rid, **kwargs):
        mask = after["road_id"] == rid
        if not mask.any():
            return
        for k, v in kwargs.items():
            after.loc[mask, k] = v
    if scenario == "increase_capacity":
        factor = float(params.get("capacity_factor", 1.35))
        cap = float(after.loc[after["road_id"] == road_id, "road_capacity"].mean() or ROADS[road_id]["capacity"])
        new_cap = cap * factor
        touch(road_id, road_capacity=new_cap)
        # speed rises as load drops
        row = after[after["road_id"] == road_id].iloc[0]
        new_load = float(row["vehicle_count"] or 0) / max(new_cap, 1)
        new_speed = min(ROADS[road_id]["ff_speed"] * 0.95, float(row["average_speed"] or 20) * (1 + (float(row.get("traffic_load") or 1) - new_load)))
        touch(road_id, average_speed=new_speed)
        notes.append(f"Capacity on {road_id} increased by {int((factor-1)*100)}% (simulated geometry / lane add).")
    elif scenario == "reduce_demand":
        factor = float(params.get("demand_factor", 0.72))
        for rid in after["road_id"]:
            row = after[after["road_id"] == rid].iloc[0]
            touch(rid, vehicle_count=max(10, float(row["vehicle_count"] or 0) * factor), average_speed=min(ROADS.get(rid, {}).get("ff_speed", 50), float(row["average_speed"] or 20) / max(factor, 0.4) * 0.85))
        notes.append(f"Network demand reduced to {int(factor*100)}% (simulated travel-demand management).")
    elif scenario == "road_closure":
        touch(road_id, average_speed=4, vehicle_count=float(after.loc[after["road_id"]==road_id, "road_capacity"].mean() or 180) * 1.4)
        # divert to neighbors
        from .graph_network import ADJACENCY
        for n in ADJACENCY.get(road_id, [])[:3]:
            if n not in set(after["road_id"]):
                continue
            row = after[after["road_id"] == n].iloc[0]
            touch(n, vehicle_count=float(row["vehicle_count"] or 0) * 1.18, average_speed=max(8, float(row["average_speed"] or 20) * 0.88))
        notes.append(f"{road_id} closed in simulation. Spillover assigned to adjacent corridors.")
    elif scenario == "event_traffic":
        boost = float(params.get("event_factor", 1.45))
        targets = [road_id, "R114", "R109"]
        for rid in targets:
            if rid not in set(after["road_id"]):
                continue
            row = after[after["road_id"] == rid].iloc[0]
            touch(rid, vehicle_count=float(row["vehicle_count"] or 0) * boost, average_speed=max(7, float(row["average_speed"] or 20) * 0.7), event="stadium_event")
        notes.append("Event demand overlay applied near stadium / selected corridor.")
    elif scenario == "add_alternate_route":
        # Simulated new parallel capacity: 25% of focus traffic shifts, speeds rise on focus
        row = after[after["road_id"] == road_id].iloc[0]
        shifted = float(row["vehicle_count"] or 0) * 0.28
        touch(road_id, vehicle_count=max(20, float(row["vehicle_count"] or 0) - shifted), average_speed=min(ROADS[road_id]["ff_speed"], float(row["average_speed"] or 18) * 1.28))
        # load neighbors slightly
        from .graph_network import ADJACENCY
        for n in ADJACENCY.get(road_id, [])[:2]:
            if n not in set(after["road_id"]):
                continue
            nrow = after[after["road_id"] == n].iloc[0]
            touch(n, vehicle_count=float(nrow["vehicle_count"] or 0) + shifted / 2)
        notes.append("Hypothetical parallel facility absorbs ~28% of corridor demand.")
    else:  # modify_road
        speed = params.get("speed")
        count = params.get("vehicle_count")
        cap = params.get("capacity")
        kwargs = {}
        if speed is not None:
            kwargs["average_speed"] = float(speed)
        if count is not None:
            kwargs["vehicle_count"] = float(count)
        if cap is not None:
            kwargs["road_capacity"] = float(cap)
        touch(road_id, **kwargs)
        notes.append(f"Manual condition override applied to {road_id}.")
    after = enrich(after)
    before_m = _metrics(before_df)
    after_m = _metrics(after)
    improvement = 0.0
    if before_m["travel_time"]:
        improvement = (before_m["travel_time"] - after_m["travel_time"]) / before_m["travel_time"] * 100
    # Also compare a sample path R101-R107
    snap_b = {r["road_id"]: r for r in before_df.to_dict(orient="records")}
    snap_a = {r["road_id"]: r for r in after.to_dict(orient="records")}
    path = k_shortest_paths("R101", "R107", k=1)
    path = path[0] if path else ["R101", "R102", "R103", "R104", "R107"]
    pb = path_metrics(path, snap_b)
    pa = path_metrics(path, snap_a)
    chart = [
        {"metric": "Average Speed", "before": before_m["average_speed"], "after": after_m["average_speed"]},
        {"metric": "Travel Time", "before": before_m["travel_time"], "after": after_m["travel_time"]},
        {"metric": "Vehicle Queue", "before": before_m["vehicle_queue"], "after": after_m["vehicle_queue"]},
        {"metric": "Congested Roads", "before": before_m["congested_roads"], "after": after_m["congested_roads"]},
    ]
    return {
        "banner": "SIMULATION — NOT REAL-WORLD CONTROL",
        "scenario": scenario,
        "road_id": road_id,
        "notes": notes,
        "before": before_m,
        "after": after_m,
        "estimated_improvement_pct": round(improvement, 1),
        "path_before": pb,
        "path_after": pa,
        "chart": chart,
        "comparison_table": [
            {"metric": "Average Speed", "before": f"{before_m['average_speed']} km/h", "after": f"{after_m['average_speed']} km/h"},
            {"metric": "Congestion", "before": before_m["congestion"], "after": after_m["congestion"]},
            {"metric": "Travel Time", "before": f"{before_m['travel_time']} min", "after": f"{after_m['travel_time']} min"},
            {"metric": "Vehicle Queue", "before": before_m["vehicle_queue"], "after": after_m["vehicle_queue"]},
        ],
    }