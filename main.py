"""SmartFlow AI FastAPI application."""
from __future__ import annotations
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import APP_NAME, DATA_PATH, DEBUG, FRONTEND_ORIGIN, ROOT
from . import analytics as analytics_mod
from . import bottlenecks as bottlenecks_mod
from . import congestion as congestion_mod
from . import demo as demo_mod
from . import forecasting as forecasting_mod
from . import incidents as incidents_mod
from . import recommendations as rec_mod
from . import simulation as sim_mod
from . import traffic as traffic_mod
from .database import (
    REQUIRED_COLUMNS,
    data_quality_report,
    init_db,
    load_frame,
    preview_csv,
    replace_traffic,
)
from .graph_network import ROADS, road_list
app = FastAPI(title=APP_NAME, version="1.0.0", description="Urban Traffic Flow & Incident Intelligence (simulated advisory).")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def _startup():
    init_db()
    load_frame()
@app.exception_handler(Exception)
async def all_errors(_, exc: Exception):
    if DEBUG:
        detail = f"{exc}\n{traceback.format_exc()}"
    else:
        detail = "An internal error occurred while processing traffic intelligence."
    return JSONResponse(status_code=500, content={"error": str(exc), "detail": detail, "ok": False})
@app.get("/api/health")
def health():
    q = data_quality_report()
    return {
        "ok": True,
        "app": APP_NAME,
        "time": datetime.now().isoformat(timespec="seconds"),
        "rows": q["row_count"],
        "data_quality": q["data_quality"],
        "models": {
            "forecast": (ROOT / "models" / "forecast_model.pkl").exists(),
            "anomaly": (ROOT / "models" / "anomaly_model.pkl").exists(),
        },
    }
@app.get("/api/status")
def status():
    return traffic_mod.kpis()
@app.get("/api/traffic")
def traffic(
    road_id: str | None = None,
    congestion: str | None = None,
    incident: str | None = None,
    severity: str | None = None,
    time_from: str | None = None,
    time_to: str | None = None,
    limit: int = 500,
):
    df = traffic_mod.filters(
        None,
        road_id=road_id,
        congestion=congestion,
        incident=incident,
        severity=severity,
        time_from=time_from,
        time_to=time_to,
    )
    if df.empty:
        return {"rows": [], "message": "Traffic data could not be loaded. Please upload a valid CSV file."}
    recs = df.sort_values("timestamp", ascending=False).head(limit).fillna("").to_dict(orient="records")
    for r in recs:
        r["timestamp"] = str(r["timestamp"])
    return {"rows": recs, "count": len(recs)}
@app.get("/api/roads")
def roads():
    return {"roads": road_list(), "map": traffic_mod.map_roads()}
@app.get("/api/network")
def network():
    return traffic_mod.network()
@app.get("/api/kpis")
def kpis():
    return traffic_mod.kpis()
@app.get("/api/congestion")
def congestion(road_id: str | None = None):
    df = traffic_mod.current_frame()
    return {
        "table": congestion_mod.congestion_table(df),
        "series": congestion_mod.congestion_series(df, road_id=road_id),
        "formula": "Traffic Load = Vehicle Count / Road Capacity",
    }
@app.get("/api/incidents")
def incidents():
    return {"incidents": incidents_mod.detect_incidents(), "disclaimer": "Uncertain detections are not confirmed facts."}
@app.get("/api/alerts")
def alerts():
    return {"alerts": incidents_mod.alert_feed()}
@app.get("/api/forecast")
def forecast(road_id: str | None = Query(default=None)):
    if road_id:
        return forecasting_mod.forecast_road(road_id.upper())
    return forecasting_mod.network_forecast()
@app.get("/api/recommendations")
def recommendations(road_id: str | None = None):
    if road_id:
        return rec_mod.recommend_for_road(road_id.upper())
    return {"items": rec_mod.all_recommendations(), "banner": "SIMULATED ADVISORY"}
@app.get("/api/bottlenecks")
def bottlenecks():
    return bottlenecks_mod.bottleneck_report()
@app.post("/api/simulate")
def simulate(payload: dict):
    return sim_mod.simulate(payload)
@app.get("/api/simulate/scenarios")
def simulate_scenarios():
    return {"scenarios": sim_mod.SCENARIOS}
@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a valid CSV file.")
    dest_dir = DATA_PATH / "uploads"
    dest_dir.mkdir(exist_ok=True)
    file_path = dest_dir / file.filename
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)
    try:
        preview = preview_csv(file_path)
        if not preview:
            raise HTTPException(400, "Uploaded CSV file is empty or invalid.")
        replace_traffic(preview)
    except Exception as e:
        raise HTTPException(500, f"Error processing uploaded file: {str(e)}")
    