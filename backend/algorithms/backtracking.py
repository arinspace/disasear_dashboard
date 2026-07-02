from backend.utils.geo import compute_bounds
from backend.services.danger_service import total_danger, movement_cost


def algo_backtracking(start_m, goal_m, hazards, grid_step=40):
    b = compute_bounds(hazards, [], goal_m)
    snap = lambda v: round(v / grid_step)
    unsnap = lambda g: g * grid_step
    key_fn = lambda x, y: f"{x},{y}"

    sg = (snap(start_m["x"]), snap(start_m["y"]))
    gg = (snap(goal_m["x"]), snap(goal_m["y"]))
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
    visited = set()
    found_path = None
    step_count = 0
    backtrack_count = 0
    depth_limit = 600

    def dfs(gx, gy, path, depth):
        nonlocal found_path, step_count, backtrack_count
        if found_path or depth > depth_limit:
            return
        mx, my = unsnap(gx), unsnap(gy)
        if mx < b["minX"] or mx > b["maxX"] or my < b["minY"] or my > b["maxY"]:
            return
        k = key_fn(gx, gy)
        if k in visited:
            return
        if total_danger(mx, my, hazards) > 0.6:
            return

        visited.add(k)
        path.append({"x": mx, "y": my})
        step_count += 1

        if gx == gg[0] and gy == gg[1]:
            found_path = list(path)
            return

        sorted_dirs = sorted(dirs, key=lambda d: abs(gx + d[0] - gg[0]) + abs(gy + d[1] - gg[1]))
        for dx, dy in sorted_dirs:
            dfs(gx + dx, gy + dy, path, depth + 1)
            if found_path:
                return

        path.pop()
        backtrack_count += 1

    dfs(sg[0], sg[1], [], 0)

    if not found_path:
        found_path = [{"x": start_m["x"], "y": start_m["y"]}]

    final_path = [{"x": start_m["x"], "y": start_m["y"]}]
    for p in found_path:
        last = final_path[-1]
        if abs(p["x"] - last["x"]) > 0.01 or abs(p["y"] - last["y"]) > 0.01:
            final_path.append(p)
    final_path.append({"x": goal_m["x"], "y": goal_m["y"]})

    cost = sum(movement_cost(final_path[i]["x"], final_path[i]["y"], final_path[i+1]["x"], final_path[i+1]["y"], hazards)
               for i in range(len(final_path) - 1))
    return {"path": final_path, "cost": round(cost, 4), "algo_name": "Backtracking Search",
            "extra": f"Steps: {step_count} | Backtracks: {backtrack_count}"}
