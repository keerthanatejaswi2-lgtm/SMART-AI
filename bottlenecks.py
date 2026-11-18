"""Recurring bottleneck identification from historical traffic."""
from __future__ import annotations
import pandas as pd
from .congestion import enrich
from .graph_network import CHRONIC_BOTTLENECKS, ROADS
from .traffic import current_frame
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
def bottleneck_report() -> dict:
    df = enrich(current_frame())
    if df.empty:
        return {"error": "Traffic data could not be loaded. Please upload a valid CSV file."}
    work = df.copy()
    work["hour"] = pd.to_datetime(work["timestamp"]).dt.hour
    work["dow"] = pd.to_datetime(work["timestamp"]).dt.dayofweek
    work["weekday"] = pd.to_datetime(work["timestamp"]).dt.day_name()
    severe = work[work["congestion"].isin(["HIGH", "SEVERE"])]
    recs = []
    for rid, grp in work.groupby("road_id"):
        sev = severe[severe["road_id"] == rid]
        freq = len(sev) / max(len(grp), 1)
        if freq < 0.12 and rid not in CHRONIC_BOTTLENECKS:
            continue
        peak = (
            sev.groupby("hour").size().idxmax()
            if len(sev)
            else int(grp.groupby("hour")["congestion_score"].mean().idxmax())
        )
        by_day = {}
        for i, name in enumerate(WEEKDAYS):
            day = sev[sev["dow"] == i]
            if day.empty:
                by_day[name] = "LOW"
            else:
                # most common level that day at peak hour
                peak_day = day[day["hour"] == peak]
                src = peak_day if len(peak_day) else day
                by_day[name] = src["congestion"].mode().iloc[0]
        trend = (
            grp.sort_values("timestamp")
            .tail(24)
            .groupby("timestamp")["congestion_score"]
            .mean()
        )
        recs.append(
            {
                "road_id": rid,
                "name": ROADS.get(rid, {}).get("name", rid),
                "banner": "RECURRING BOTTLENECK DETECTED" if freq >= 0.18 or rid in CHRONIC_BOTTLENECKS else "WATCHLIST",
                "frequency": round(freq * 100, 1),
                "average_congestion": round(float(grp["congestion_score"].mean()), 1),
                "peak_time": f"{peak:02d}:00",
                "peak_hour": peak,
                "average_speed": round(float(grp["average_speed"].mean()), 1),
                "weekday_pattern": by_day,
                "trend": [{"timestamp": str(i), "score": round(float(v), 1)} for i, v in trend.items()],
                "sample_count": int(len(grp)),
                "severe_count": int(len(sev)),
            }
        )
    recs.sort(key=lambda r: r["frequency"], reverse=True)
    return {"bottlenecks": recs, "count": len(recs)}