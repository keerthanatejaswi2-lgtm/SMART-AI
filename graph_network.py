meta["latitude"] = round(lat, 6)
meta["longitude"] = round(lon, 6)
meta["length_km"] = _len_km(a, b)
meta["polyline"] = _poly(a, b)
meta["road_id"] = rid
# Adjacency: shared interchange or designed interchange (supports R103→R105→R107).
ADJACENCY: Dict[str, List[str]] = {
    "R101": ["R102", "R105", "R111"],
    "R102": ["R101", "R103", "R105", "R106"],
    "R103": ["R102", "R104", "R105", "R106"],
    "R104": ["R103", "R107", "R110"],
    "R105": ["R101", "R102", "R103", "R108", "R109", "R107"],
    "R106": ["R102", "R103", "R109", "R110"],
    "R107": ["R104", "R105", "R110", "R115"],
    "R108": ["R105", "R109", "R111", "R112"],
    "R109": ["R105", "R106", "R108", "R110", "R114"],
    "R110": ["R104", "R106", "R107", "R109"],
    "R111": ["R101", "R108", "R112"],
    "R112": ["R108", "R111", "R113", "R116"],
    "R113": ["R112", "R114", "R116", "R117"],
    "R114": ["R109", "R113", "R115", "R117", "R118"],
    "R115": ["R107", "R114", "R118"],
    "R116": ["R112", "R113", "R117"],
    "R117": ["R113", "R114", "R116", "R118"],
    "R118": ["R114", "R115", "R117"],
}
CHRONIC_BOTTLENECKS = {"R103", "R107", "R110", "R114"}
def travel_minutes(length_km: float, speed_kmh: float) -> float:
    speed = max(float(speed_kmh or 5.0), 4.0)
    return round((length_km / speed) * 60.0, 1)
def shortest_path(origin: str, dest: str, blocked: Optional[set] = None) -> Optional[List[str]]:
    """Unweighted BFS path over road nodes, skipping blocked segments (except endpoints)."""
    blocked = blocked or set()
    if origin not in ROADS or dest not in ROADS:
        return None
    from collections import deque
    q = deque([[origin]])
    seen = {origin}
    while q:
        path = q.popleft()
        node = path[-1]
        if node == dest:
            return path
        for nxt in ADJACENCY.get(node, []):
            if nxt in seen:
                continue
            if nxt in blocked and nxt not in (origin, dest):
                continue
            seen.add(nxt)
            q.append(path + [nxt])
    return None
def k_shortest_paths(origin: str, dest: str, k: int = 4, blocked: Optional[set] = None) -> List[List[str]]:
    """Yen-style simple alternative paths by successively blocking interior nodes."""
    blocked = set(blocked or [])
    primary = shortest_path(origin, dest, blocked)
    if not primary:
        return []
    paths = [primary]
    for hop in primary[1:-1]:
        alt = shortest_path(origin, dest, blocked | {hop})
        if alt and alt not in paths:
            paths.append(alt)
        if len(paths) >= k:
            break
    # Additional corridors via Civic Bypass / Central Ave
    for via in ("R105", "R109", "R108"):
        if via in (origin, dest) or via in blocked:
            continue
        a = shortest_path(origin, via, blocked)
        b = shortest_path(via, dest, blocked)
        if a and b:
            merged = a[:-1] + b
            if merged not in paths and len(merged) <= 7:
                paths.append(merged)
        if len(paths) >= k:
            break
    return paths[:k]
def path_metrics(path: List[str], snapshot: Dict[str, dict]) -> dict:
    dist = 0.0
    time_min = 0.0
    speeds = []
    loads = []
    caps = []
    for rid in path:
        road = ROADS[rid]
        rec = snapshot.get(rid, {})
        speed = float(rec.get("average_speed") or road["ff_speed"] * 0.6)
        count = float(rec.get("vehicle_count") or road["capacity"] * 0.5)
        cap = float(rec.get("road_capacity") or road["capacity"])
        dist += road["length_km"]
        time_min += travel_minutes(road["length_km"], speed)
        speeds.append(speed)
        loads.append(count / max(cap, 1))
        caps.append(max(cap - count, 0))
    n = max(len(path), 1)
    return {
        "path": path,
        "label": " → ".join(path),
        "distance_km": round(dist, 2),
        "travel_time_min": round(time_min, 1),
        "avg_speed": round(sum(speeds) / n, 1),
        "avg_load": round(sum(loads) / n, 3),
        "spare_capacity": round(sum(caps) / n, 1),
        "congestion": classify_load(sum(loads) / n, sum(speeds) / n),
    }
def classify_load(load: float, speed: float) -> str:
    if load >= 1.15 or speed < 14:
        return "SEVERE"
    if load >= 0.9 or speed < 22:
        return "HIGH"
    if load >= 0.65 or speed < 32:
        return "MODERATE"
    return "LOW"
def graph_payload(snapshot: Dict[str, dict]) -> dict:
    nodes = []
    for rid, meta in ROADS.items():
        rec = snapshot.get(rid, {})
        speed = float(rec.get("average_speed") or meta["ff_speed"])
        count = float(rec.get("vehicle_count") or 0)
        cap = float(rec.get("road_capacity") or meta["capacity"])
        load = count / max(cap, 1)
        nodes.append(
            {
                "id": rid,
                "name": meta["name"],
                "length_km": meta["length_km"],
                "capacity": cap,
                "speed": speed,
                "traffic": count,
                "travel_time": travel_minutes(meta["length_km"], speed),
                "congestion": classify_load(load, speed),
                "load": round(load, 3),
            }
        )
    edges = []
    seen = set()
    for a, nbrs in ADJACENCY.items():
        for b in nbrs:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"from": a, "to": b})
    return {"nodes": nodes, "edges": edges}
def road_list() -> List[dict]:
    return [dict(v) for v in ROADS.values()]