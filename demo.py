"""Guided demo scenario for hackathon judging."""
from __future__ import annotations
from datetime import datetime
from .graph_network import ROADS
from .traffic import DEMO_STATE, snapshot_enriched
STEPS = [
    {"id": 1, "title": "Normal traffic", "narrative": "Network operating near free-flow. Civic Bypass spare capacity is high."},
    {"id": 2, "title": "Traffic increases", "narrative": "AM peak demand rises on River North and Harbor Link."},
    {"id": 3, "title": "Congestion detected", "narrative": "Traffic load on R103 exceeds capacity. Congestion classified SEVERE."},
    {"id": 4, "title": "Incident detected", "narrative": "Abnormal speed drop and density spike — possible incident flagged with evidence."},
    {"id": 5, "title": "Forecast worsening", "narrative": "15–45 minute horizon shows further speed deterioration before partial recovery at 60 minutes."},
    {"id": 6, "title": "Diversion recommendation", "narrative": "SIMULATED ADVISORY: R103 → R105 → R107 reduces estimated travel time."},
    {"id": 7, "title": "Recurring bottleneck", "narrative": "Historical pattern: R103 is severe at 08:00 across weekdays."},
    {"id": 8, "title": "What-if simulation", "narrative": "Capacity increase on R103 is simulated — not real-world control."},
    {"id": 9, "title": "Before / after impact", "narrative": "Estimated travel-time and queue reductions are shown for decision support."},
]
def _overlay_for_step(step: int) -> dict:
    ov = {}
    if step <= 1:
        for rid, meta in ROADS.items():
            ov[rid] = {"average_speed": meta["ff_speed"] * 0.88, "vehicle_count": meta["capacity"] * 0.45, "incident": "", "event": ""}
    elif step == 2:
        for rid, meta in ROADS.items():
            ov[rid] = {"average_speed": meta["ff_speed"] * 0.7, "vehicle_count": meta["capacity"] * 0.78, "incident": ""}
        ov["R103"] = {"average_speed": 22, "vehicle_count": 170, "incident": ""}
        ov["R107"] = {"average_speed": 24, "vehicle_count": 165, "incident": ""}
    elif step == 3:
        for rid, meta in ROADS.items():
            ov[rid] = {"average_speed": meta["ff_speed"] * 0.62, "vehicle_count": meta["capacity"] * 0.9}
        ov["R103"] = {"average_speed": 12, "vehicle_count": 280, "road_capacity": 200, "incident": ""}
        ov["R102"] = {"average_speed": 18, "vehicle_count": 200}
        ov["R107"] = {"average_speed": 16, "vehicle_count": 210}
    elif step >= 4:
        for rid, meta in ROADS.items():
            ov[rid] = {"average_speed": meta["ff_speed"] * 0.6, "vehicle_count": meta["capacity"] * 0.92}
        ov["R103"] = {
            "average_speed": 11,
            "vehicle_count": 290,
            "road_capacity": 200,
            "incident": "possible_accident",
        }
        ov["R104"] = {"average_speed": 14, "vehicle_count": 210}
        ov["R107"] = {"average_speed": 15, "vehicle_count": 220}
        ov["R105"] = {"average_speed": 36, "vehicle_count": 140}
        ov["R106"] = {"average_speed": 19, "vehicle_count": 190}
    return ov
def start_demo() -> dict:
    DEMO_STATE["active"] = True
    DEMO_STATE["step"] = 1
    DEMO_STATE["started_at"] = datetime.now().isoformat(timespec="seconds")
    DEMO_STATE["overlay"] = _overlay_for_step(1)
    return status()
def set_step(step: int) -> dict:
    step = int(max(1, min(9, step)))
    DEMO_STATE["active"] = True
    DEMO_STATE["step"] = step
    DEMO_STATE["overlay"] = _overlay_for_step(step)
    return status()
def next_step() -> dict:
    return set_step(DEMO_STATE.get("step", 1) + 1)
def stop_demo() -> dict:
    DEMO_STATE["active"] = False
    DEMO_STATE["step"] = 0
    DEMO_STATE["overlay"] = {}
    return status()
def status() -> dict:
    step = DEMO_STATE.get("step") or 0
    info = next((s for s in STEPS if s["id"] == step), None)
    return {
        "active": DEMO_STATE["active"],
        "step": step,
        "total_steps": 9,
        "info": info,
        "steps": STEPS,
        "started_at": DEMO_STATE.get("started_at"),
    }