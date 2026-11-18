"""SQLite persistence, CSV ingest, and data-quality scoring."""
from __future__ import annotations
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from .config import DATA_PATH, DB_FILE
from .graph_network import ROADS
REQUIRED_COLUMNS = [
    "timestamp",
    "road_id",
    "latitude",
    "longitude",
    "vehicle_count",
    "average_speed",
    "road_capacity",
    "weather",
    "incident",
    "event",
    "road_work",
]
def connect() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn
def init_db() -> None:
    conn = connect()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS traffic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            road_id TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            vehicle_count REAL,
            average_speed REAL,
            road_capacity REAL,
            weather TEXT,
            incident TEXT,
            event TEXT,
            road_work TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_traffic_ts ON traffic(timestamp);
        CREATE INDEX IF NOT EXISTS idx_traffic_road ON traffic(road_id);
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )
    conn.commit()
    conn.close()
def set_meta(key: str, value: str) -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()
def get_meta(key: str, default: Optional[str] = None) -> Optional[str]:
    conn = connect()
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default
def table_count() -> int:
    conn = connect()
    n = conn.execute("SELECT COUNT(*) AS n FROM traffic").fetchone()["n"]
    conn.close()
    return int(n)
def replace_traffic(df: pd.DataFrame) -> int:
    cleaned = clean_dataframe(df)
    conn = connect()
    conn.execute("DELETE FROM traffic")
    cleaned.to_sql("traffic", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    set_meta("last_ingest", datetime.now().isoformat(timespec="seconds"))
    set_meta("row_count", str(len(cleaned)))
    return len(cleaned)
def append_traffic(df: pd.DataFrame) -> int:
    cleaned = clean_dataframe(df)
    conn = connect()
    cleaned.to_sql("traffic", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    set_meta("last_ingest", datetime.now().isoformat(timespec="seconds"))
    return len(cleaned)
def load_frame() -> pd.DataFrame:
    init_db()
    if table_count() == 0:
        csv_path = DATA_PATH / "traffic.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            replace_traffic(df)
    conn = connect()
    df = pd.read_sql_query("SELECT * FROM traffic", conn)
    conn.close()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp")
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    frame = frame[REQUIRED_COLUMNS]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "road_id"])
    frame["road_id"] = frame["road_id"].astype(str).str.upper().str.strip()
    known = set(ROADS.keys())
    frame = frame[frame["road_id"].isin(known) | (~frame["road_id"].isin(known))]
    for col in ["vehicle_count", "average_speed", "road_capacity", "latitude", "longitude"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.loc[frame["average_speed"] < 0, "average_speed"] = np.nan
    frame.loc[frame["average_speed"] > 140, "average_speed"] = np.nan
    frame.loc[frame["vehicle_count"] < 0, "vehicle_count"] = np.nan
    frame.loc[frame["road_capacity"] <= 0, "road_capacity"] = np.nan
    frame = frame.drop_duplicates(subset=["timestamp", "road_id"], keep="last")
    # Fill missing geo/capacity from network catalog
    for rid, meta in ROADS.items():
        mask = frame["road_id"] == rid
        frame.loc[mask & frame["latitude"].isna(), "latitude"] = meta["latitude"]
        frame.loc[mask & frame["longitude"].isna(), "longitude"] = meta["longitude"]
        frame.loc[mask & frame["road_capacity"].isna(), "road_capacity"] = meta["capacity"]
    frame["weather"] = frame["weather"].fillna("clear").astype(str).str.lower()
    for col in ["incident", "event", "road_work"]:
        frame[col] = frame[col].fillna("").astype(str)
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return frame
def data_quality_report(df: Optional[pd.DataFrame] = None) -> dict:
    if df is None:
        df = load_frame()
    if df.empty:
        return {
            "data_quality": 0,
            "missing_records": 0,
            "anomalies": 0,
            "duplicates": 0,
            "invalid_values": 0,
            "row_count": 0,
            "message": "Traffic data could not be loaded. Please upload a valid CSV file.",
        }
    work = df.copy()
    numeric = ["vehicle_count", "average_speed", "road_capacity"]
    missing = int(work[numeric].isna().sum().sum())