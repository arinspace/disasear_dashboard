import math, random

from backend.utils.geo import compute_bounds
from backend.services.danger_service import total_danger, movement_cost


def algo_simulated_annealing(start_m, goal_m, hazards, grid_step=None):
    b = compute_bounds(hazards, [], goal_m)
    def obj(x, y):
        return movement_cost(x, y, goal_m["x"], goal_m["y"], hazards) + math.pow(10, total_danger(x, y, hazards) * 4)

    cx, cy = start_m["x"], start_m["y"]
    best_x, best_y, best_val = cx, cy, obj(cx, cy)
    path = [{"x": cx, "y": cy}]
    t = 100.0
    cooling = 0.993
    max_iter = 2000
    step_size = 18
    accepted = 0
    rejected = 0

    for _ in range(max_iter):
        if t <= 0.01:
            break
        t *= cooling
        angle = random.uniform(0, 2 * math.pi)
        dist = step_size * (0.3 + random.random() * 0.7)
        nx = max(b["minX"], min(b["maxX"], cx + math.cos(angle) * dist))
        ny = max(b["minY"], min(b["maxY"], cy + math.sin(angle) * dist))

        cur_val = obj(cx, cy)
        new_val = obj(nx, ny)
        delta_e = new_val - cur_val

        if delta_e < 0 or (t > 1e-6 and random.random() < math.exp(-delta_e / t)):
            cx, cy = nx, ny
            path.append({"x": cx, "y": cy})
            accepted += 1
            if new_val < best_val:
                best_x, best_y, best_val = cx, cy, new_val
        else:
            rejected += 1

    path.append({"x": goal_m["x"], "y": goal_m["y"]})
    dedup = [path[0]]
    for p in path[1:]:
        last = dedup[-1]
        if abs(p["x"] - last["x"]) > 0.5 or abs(p["y"] - last["y"]) > 0.5:
            dedup.append(p)

    cost = sum(movement_cost(dedup[i]["x"], dedup[i]["y"], dedup[i+1]["x"], dedup[i+1]["y"], hazards)
               for i in range(len(dedup) - 1))
    return {"path": dedup, "cost": round(cost, 4), "algo_name": "Simulated Annealing",
            "extra": f"Accepted: {accepted} | Rejected: {rejected}"}
