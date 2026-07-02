import math

from backend.utils.geo import to_meters, DEFAULT_HAZARD_RADIUS_M


def total_danger(mx, my, hazards):
    total = 0.0
    for h in hazards:
        hm = to_meters(h["lat"], h["lng"])
        dist = math.hypot(mx - hm["x"], my - hm["y"])
        radius = h.get("radius_m", DEFAULT_HAZARD_RADIUS_M)
        if dist < radius:
            total += (h.get("severity", 5) / 10.0) * (1.0 - dist / radius)
    return min(total, 1.0)


def movement_cost(ax, ay, bx, by, hazards):
    dist = math.hypot(bx - ax, by - ay)
    mid_danger = total_danger((ax + bx) / 2, (ay + by) / 2, hazards)
    end_danger = total_danger(bx, by, hazards)
    danger = max(mid_danger, end_danger)
    return dist * math.pow(10, danger * 3)


def compute_path_danger_stats(path_m, hazards):
    dangers = [total_danger(p["x"], p["y"], hazards) for p in path_m]
    return {
        "max": round(max(dangers), 3) if dangers else 0,
        "avg": round(sum(dangers) / len(dangers), 3) if dangers else 0,
        "high_count": sum(1 for d in dangers if d > 0.4),
    }


def compute_danger_points(path_m, hazards, max_points=30):
    dangers = [total_danger(p["x"], p["y"], hazards) for p in path_m]
    step = max(1, len(path_m) // max_points)
    points = []
    for i in range(0, len(path_m), step):
        d = dangers[i]
        if d > 0.25:
            from backend.utils.geo import to_latlng
            latlng = to_latlng(path_m[i]["x"], path_m[i]["y"])
            points.append({"lat": latlng[0], "lng": latlng[1], "danger": round(d, 3)})
    return points
