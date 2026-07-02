import time
from typing import Optional

from backend.utils.geo import to_meters, to_latlng
from backend.services.danger_service import (
    total_danger,
    compute_path_danger_stats,
    compute_danger_points,
)
from backend.algorithms.astar import algo_astar
from backend.algorithms.bfs import algo_bfs
from backend.algorithms.greedy import algo_greedy_bfs
from backend.algorithms.simulated_annealing import algo_simulated_annealing
from backend.algorithms.hill_climbing import algo_hill_climbing
from backend.algorithms.genetic import algo_genetic
from backend.algorithms.backtracking import algo_backtracking
from backend.algorithms.ac3 import algo_ac3

ALGORITHM_MAP = {
    "astar": algo_astar,
    "bfs": algo_bfs,
    "greedy": algo_greedy_bfs,
    "sa": algo_simulated_annealing,
    "hc": algo_hill_climbing,
    "hc_restart": lambda sm, gm, h, gs: algo_hill_climbing(sm, gm, h, "random_restart", gs),
    "ga": algo_genetic,
    "backtrack": algo_backtracking,
    "ac3": algo_ac3,
}


def run_optimization(algo_name: str, start_m: dict, goal_m: dict,
                     hazards: list, grid_step: int = 40):
    t0 = time.perf_counter()

    algo_fn = ALGORITHM_MAP.get(algo_name)
    if algo_name == "hc_restart":
        result = algo_fn(start_m, goal_m, hazards)
    elif algo_name in ("hc",):
        result = algo_fn(start_m, goal_m, hazards, "steepest", grid_step)
    elif algo_fn:
        result = algo_fn(start_m, goal_m, hazards, grid_step)
    else:
        result = algo_astar(start_m, goal_m, hazards, grid_step)
        result["algo_name"] = "A* Search (default)"

    elapsed = time.perf_counter() - t0

    if not result:
        return None, elapsed

    path_latlng = [
        [to_latlng(p["x"], p["y"])[0], to_latlng(p["x"], p["y"])[1]]
        for p in result["path"]
    ]

    danger_stats = compute_path_danger_stats(result["path"], hazards)
    danger_points = compute_danger_points(result["path"], hazards)

    response = {
        "algorithm": result["algo_name"],
        "elapsed_sec": round(elapsed, 4),
        "route": path_latlng,
        "path_cost": result["cost"],
        "waypoints": len(result["path"]),
        "danger_stats": danger_stats,
        "danger_points": danger_points,
        "extra": result.get("extra", ""),
    }

    return response, elapsed
