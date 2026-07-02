import math

EARTH_RADIUS_M = 6371000
REF_LAT = 13.7563
REF_LNG = 100.5018
DEG_M_LAT = 111320
DEFAULT_HAZARD_RADIUS_M = 500
DEFAULT_SAFETY_MARGIN_M = 500


def to_meters(lat, lng):
    dx = (lng - REF_LNG) * DEG_M_LAT * math.cos(math.radians(REF_LAT))
    dy = (lat - REF_LAT) * DEG_M_LAT
    return {"x": round(dx), "y": round(dy)}


def to_latlng(x, y):
    lng = REF_LNG + x / (DEG_M_LAT * math.cos(math.radians(REF_LAT)))
    lat = REF_LAT + y / DEG_M_LAT
    return [lat, lng]


def haversine(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def offset_coordinate(lat, lng, distance_m, bearing_deg):
    bearing = math.radians(bearing_deg)
    angular_distance = distance_m / EARTH_RADIUS_M
    lat1 = math.radians(lat)
    lng1 = math.radians(lng)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lng2 = lng1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return {"lat": math.degrees(lat2), "lng": math.degrees(lng2)}


def heuristic_cost(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)


def compute_bounds(hazards, survivors, goal_m=None):
    pts = []
    for h in hazards:
        hm = to_meters(h["lat"], h["lng"])
        r = h.get("radius_m", 200)
        pts.append({"x": hm["x"] - r, "y": hm["y"] - r})
        pts.append({"x": hm["x"] + r, "y": hm["y"] + r})
    for s in survivors:
        sm = to_meters(s["lat"], s["lng"])
        pts.append(sm)
    if goal_m:
        pts.append(goal_m)
    if not pts:
        return {"minX": -500, "maxX": 500, "minY": -500, "maxY": 500}
    margin = 300
    return {
        "minX": min(p["x"] for p in pts) - margin,
        "maxX": max(p["x"] for p in pts) + margin,
        "minY": min(p["y"] for p in pts) - margin,
        "maxY": max(p["y"] for p in pts) + margin,
    }


def exit_is_safe(exit_point, hazards, safety_margin_m=None):
    if safety_margin_m is None:
        safety_margin_m = DEFAULT_SAFETY_MARGIN_M
    return all(
        haversine(exit_point["lat"], exit_point["lng"], h["lat"], h["lng"])
        > (h.get("radius_m") or DEFAULT_HAZARD_RADIUS_M) + safety_margin_m
        for h in hazards
    )


def generate_safe_exit(survivor, hazards, safety_margin_m=None):
    if safety_margin_m is None:
        safety_margin_m = DEFAULT_SAFETY_MARGIN_M
    if not survivor:
        return None
    if not hazards:
        return offset_coordinate(survivor["lat"], survivor["lng"], 5000, 0)
    nearest = min(
        hazards,
        key=lambda h: haversine(survivor["lat"], survivor["lng"], h["lat"], h["lng"]),
    )
    nearest_distance = max(
        haversine(survivor["lat"], survivor["lng"], nearest["lat"], nearest["lng"]), 1.0,
    )
    dlat = math.radians(survivor["lat"] - nearest["lat"])
    dlng = math.radians(survivor["lng"] - nearest["lng"])
    y = math.sin(dlng) * math.cos(math.radians(survivor["lat"]))
    x = (
        math.cos(math.radians(nearest["lat"])) * math.sin(math.radians(survivor["lat"]))
        - math.sin(math.radians(nearest["lat"])) * math.cos(math.radians(survivor["lat"])) * math.cos(dlng)
    )
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360 if abs(dlat) + abs(dlng) > 1e-12 else 0
    minimum_distance = (nearest.get("radius_m") or DEFAULT_HAZARD_RADIUS_M) + safety_margin_m
    candidate_distance = max(nearest_distance + safety_margin_m, minimum_distance)
    for _ in range(60):
        candidate = offset_coordinate(nearest["lat"], nearest["lng"], candidate_distance, bearing)
        if exit_is_safe(candidate, hazards, safety_margin_m):
            return candidate
        candidate_distance += safety_margin_m
    return offset_coordinate(nearest["lat"], nearest["lng"], candidate_distance, bearing)
