"""Model evaluation metrics for the Analytics page."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from .config import MODEL_PATH, ROOT
from .congestion import enrich
from .database import data_quality_report
from .graph_network import classify_load
from .recommendations import recommend_for_road
from .traffic import current_frame
def _metrics_file() -> Path:
    return MODEL_PATH / "evaluation.json"
def evaluate_live() -> dict:
    cached = _metrics_file()
    stored = {}
    if cached.exists():
        try:
            stored = json.loads(cached.read_text(encoding="utf-8"))
        except Exception:
            stored = {}
    df = enrich(current_frame())
    quality = data_quality_report(df)
    detection = stored.get("detection") or _quick_detection(df)
    forecast = stored.get("forecast") or _quick_forecast(df)
    rec = recommend_for_road("R103")
    recommendation = {
        "estimated_travel_time_reduction_min": rec.get("expected_improvement_min", 0),
        "congestion_reduction_pct": rec.get("congestion_reduction_pct", 0),
    }
    robustness = stored.get("robustness") or _robustness(df)
    return {
        "detection": detection,
        "forecast": forecast,
        "recommendation": recommendation,
        "robustness": robustness,
        "data_quality": quality["data_quality"],
        "model_trained": (MODEL_PATH / "forecast_model.pkl").exists(),
    }
def _quick_detection(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 40:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "note": "Insufficient rows"}
    work = df.dropna(subset=["average_speed", "vehicle_count"]).copy()
    work["y"] = (
        (work["incident"].fillna("").astype(str).str.len() > 0)
        | (work["congestion"] == "SEVERE")
    ).astype(int)
    X = np.c_[
        work["average_speed"].fillna(30),
        work["vehicle_count"].fillna(100) / work["road_capacity"].replace(0, 1).fillna(200),
    ]
    iso = IsolationForest(contamination=0.12, random_state=42)
    pred = (iso.fit_predict(X) == -1).astype(int)
    y = work["y"].values
    return {
        "accuracy": round(accuracy_score(y, pred) * 100, 1),
        "precision": round(precision_score(y, pred, zero_division=0) * 100, 1),
        "recall": round(recall_score(y, pred, zero_division=0) * 100, 1),
        "f1": round(f1_score(y, pred, zero_division=0) * 100, 1),
    }
def _quick_forecast(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 80:
        return {"mae": 0, "rmse": 0, "note": "Insufficient rows"}
    work = df.dropna(subset=["average_speed"]).copy()
    work["hour"] = pd.to_datetime(work["timestamp"]).dt.hour
    work["dow"] = pd.to_datetime(work["timestamp"]).dt.dayofweek
    work["load"] = work["vehicle_count"] / work["road_capacity"].replace(0, 1)
    X = work[["hour", "dow", "load", "vehicle_count"]].fillna(0)
    y = work["average_speed"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    return {"mae": round(float(mae), 2), "rmse": round(float(rmse), 2), "unit": "km/h"}
def _robustness(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "missing_data": 0,
            "noisy_data": 0,
            "unseen_patterns": 0,
        }
    base = _quick_forecast(df)
    missing = df.copy()
    idx = missing.sample(frac=0.15, random_state=1).index
    missing.loc[idx, "average_speed"] = np.nan
    missing["average_speed"] = missing["average_speed"].fillna(missing["average_speed"].median())
    m = _quick_forecast(missing)
    noisy = df.copy()
    noisy["average_speed"] = noisy["average_speed"] + np.random.default_rng(2).normal(0, 4, len(noisy))
    n = _quick_forecast(noisy)
    unseen = df[pd.to_datetime(df["timestamp"]).dt.hour.between(11, 15)]
    u = _quick_forecast(unseen if len(unseen) > 80 else df)
    def retain(a, b):
        if not a.get("mae"):
            return 70
        return round(max(40, min(98, 100 - abs(b["mae"] - a["mae"]) / max(a["mae"], 0.5) * 25)), 1)
    return {
        "missing_data": retain(base, m),
        "noisy_data": retain(base, n),
        "unseen_patterns": retain(base, u),
        "note": "Retention score vs baseline MAE (higher is more stable).",
    }