import math, random

from backend.utils.geo import compute_bounds
from backend.services.danger_service import total_danger, movement_cost


def algo_hill_climbing(start_m, goal_m, hazards, variant="steepest", grid_step=None):
    b = compute_bounds(hazards, [], goal_m)
    step = 12
    dirs = [(step,0),(-step,0),(0,step),(0,-step),(step,step),(-step,step),(step,-step),(-step,-step)]
    def obj(x, y):
        return movement_cost(x, y, goal_m["x"], goal_m["y"], hazards) + math.pow(10, total_danger(x, y, hazards) * 4)
    max_iter = 500
    restarts = 10 if variant == "random_restart" else 1
    global_best = None

    for r in range(restarts):
        if r == 0:
            cx, cy = start_m["x"], start_m["y"]
        else:
            cx = b["minX"] + random.random() * (b["maxX"] - b["minX"])
            cy = b["minY"] + random.random() * (b["maxY"] - b["minY"])

        path = [{"x": cx, "y": cy}]
        cur_val = obj(cx, cy)

        for _ in range(max_iter):
            neighbors = []
            for dx, dy in dirs:
                nx = max(b["minX"], min(b["maxX"], cx + dx))
                ny = max(b["minY"], min(b["maxY"], cy + dy))
                neighbors.append({"x": nx, "y": ny})
            best = min(neighbors, key=lambda p: obj(p["x"], p["y"]))
            best_val = obj(best["x"], best["y"])

            if best_val >= cur_val:
                break
            cx, cy = best["x"], best["y"]
            cur_val = best_val
            path.append({"x": cx, "y": cy})

        path.append({"x": goal_m["x"], "y": goal_m["y"]})
        cost = sum(movement_cost(path[i]["x"], path[i]["y"], path[i+1]["x"], path[i+1]["y"], hazards)
                   for i in range(len(path) - 1))
        if global_best is None or cost < global_best["cost"]:
            global_best = {"path": path, "cost": cost}

    name = "Hill Climbing (Random Restart)" if variant == "random_restart" else "Hill Climbing (Steepest)"
    return {"path": global_best["path"], "cost": round(global_best["cost"], 4), "algo_name": name, "extra": f"Restarts: {restarts}"}
