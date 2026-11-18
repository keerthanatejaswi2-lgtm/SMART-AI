"""Graph-based simulated diversion advisories with explainability."""
from __future__ import annotations
from .database import data_quality_report
from .forecasting import forecast_road
from .graph_network import (
    ADJACENCY,
    ROADS,
    k_shortest_paths,
    path_metrics,
)
from .traffic import snapshot_enriched
def _snapshot_dict() -> dict:
    snap = snapshot_enriched()
    out = {}
    if snap.empty:
        return out
    for _, row in snap.iterrows():
        out[row["road_id"]] = row.to_dict()
    return out
def _predicted_speed_map() -> dict:
    snap = snapshot_enriched()
    pred = {}
    if snap.empty:
        return pred
    for rid in snap["road_id"]:
        fc = forecast_road(rid)
        if "horizons" in fc:
            # 30-min horizon used for routing
            h30 = next((p for p in fc["horizons"] if p["horizon_min"] == 30), fc["horizons"][-1])
            pred[rid] = h30["predicted_speed"]
    return pred
def recommend_for_road(road_id: str | None = None) -> dict:
    quality = data_quality_report()
    snap = _snapshot_dict()
    if not snap:
        return {"error": "Traffic data could not be loaded. Please upload a valid CSV file."}
    if not road_id:
        # pick most congested
        road_id = max(snap, key=lambda r: float(snap[r].get("traffic_load") or 0))
    road_id = road_id.upper()
    neighbors = ADJACENCY.get(road_id, [])
    if len(neighbors) < 2:
        origin, dest = road_id, neighbors[0] if neighbors else road_id
    else:
        origin, dest = neighbors[0], neighbors[-1]
        # Prefer a corridor that includes the congested road as current path
        origin, dest = "R101", "R107"
        if road_id not in ("R103", "R102", "R101", "R104", "R107"):
            # path from a neighbor through this road to another neighbor
            origin = neighbors[0]
            dest = neighbors[1] if len(neighbors) > 1 else "R107"
    current = None
    # Current route should include the congested segment when possible
    candidates = k_shortest_paths(origin, dest, k=5)
    if not candidates:
        candidates = [[road_id]]
    for path in candidates:
        if road_id in path:
            current = path
            break
    if current is None:
        current = candidates[0]
    current_m = path_metrics(current, snap)
    alts = []
    for path in candidates:
        if path == current:
            continue
        m = path_metrics(path, snap)
        alts.append(m)
    if not alts and len(candidates) > 1:
        alts = [path_metrics(candidates[1], snap)]
    if not alts:
        # fabricate civic bypass alternative using R105
        alt_path = k_shortest_paths("R101", "R107", k=4)
        for p in alt_path:
            if p != current:
                alts.append(path_metrics(p, snap))
                break
    alts.sort(key=lambda x: x["travel_time_min"])
    best = alts[0] if alts else current_m
    improve = round(current_m["travel_time_min"] - best["travel_time_min"], 1)
    cong_delta = 0
    if current_m["avg_load"]:
        cong_delta = round((current_m["avg_load"] - best["avg_load"]) / max(current_m["avg_load"], 0.05) * 100)
    cap_delta = 0
    if current_m["spare_capacity"]:
        cap_delta = round((best["spare_capacity"] - current_m["spare_capacity"]) / max(current_m["spare_capacity"], 1) * 100)
    reasons = []
    if cong_delta > 0:
        reasons.append(f"{cong_delta}% lower predicted congestion")
    if cap_delta > 0:
        reasons.append(f"{abs(cap_delta)}% higher available capacity")
    if improve > 0:
        reasons.append(f"{improve}-minute estimated travel-time reduction")
    if not reasons:
        reasons.append("Comparable travel time; selected for spare capacity and network redundancy")
    pred = _predicted_speed_map()
    def enrich_path(m):
        speeds = [pred.get(r, m["avg_speed"]) for r in m["path"]]
        return {**m, "predicted_speed": round(sum(speeds) / max(len(speeds), 1), 1)}
    current_m = enrich_path(current_m)
    best = enrich_path(best)
    alts = [enrich_path(a) for a in alts]
    conf = round(min(93, quality["data_quality"] * 0.95), 1)
    return {
        "banner": "SIMULATED ADVISORY",
        "disclaimer": "Never treat this as live traffic control. Software-only decision support.",
        "focus_road": road_id,
        "origin": origin,
        "destination": dest,
        "current_route": current_m,
        "recommended": best,
        "alternatives": alts[:3],
        "expected_improvement_min": improve,
        "why": reasons,
        "confidence": conf,
        "data_quality": quality["data_quality"],
        "congestion_reduction_pct": max(0, cong_delta),
        "capacity_gain_pct": max(0, cap_delta),
    }
def all_recommendations() -> list:
    snap = snapshot_enriched()
    if snap.empty:
        return []
    severe = snap[snap["congestion"].isin(["HIGH", "SEVERE"])]["road_id"].tolist()
    if not severe:
        severe = [snap.sort_values("traffic_load", ascending=False).iloc[0]["road_id"]]
    out = []
    seen = set()
    for rid in severe[:4]:
        rec = recommend_for_road(rid)
        key = rec.get("recommended", {}).get("label")
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out