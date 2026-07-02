from backend.utils.geo import compute_bounds
from backend.services.danger_service import total_danger, movement_cost


def algo_ac3(start_m, goal_m, hazards, grid_step=40):
    b = compute_bounds(hazards, [], goal_m)
    snap = lambda v: round(v / grid_step)
    unsnap = lambda g: g * grid_step
    key_fn = lambda x, y: f"{x},{y}"

    sg = (snap(start_m["x"]), snap(start_m["y"]))
    gg = (snap(goal_m["x"]), snap(goal_m["y"]))
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]
    g_min_x, g_max_x = snap(b["minX"]), snap(b["maxX"])
    g_min_y, g_max_y = snap(b["minY"]), snap(b["maxY"])

    # Phase 1: BFS backward from goal
    reachable = set()
    queue = [(gg[0], gg[1])]
    reachable.add(key_fn(gg[0], gg[1]))

    while queue:
        cx, cy = queue.pop(0)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if nx < g_min_x or nx > g_max_x or ny < g_min_y or ny > g_max_y:
                continue
            k = key_fn(nx, ny)
            if k in reachable:
                continue
            mx, my = unsnap(nx), unsnap(ny)
            if total_danger(mx, my, hazards) > 0.6:
                continue
            reachable.add(k)
            queue.append((nx, ny))

    total_cells = 0
    for gx in range(g_min_x, g_max_x + 1):
        for gy in range(g_min_y, g_max_y + 1):
            mx, my = unsnap(gx), unsnap(gy)
            if total_danger(mx, my, hazards) <= 0.6:
                total_cells += 1
    pruned_count = total_cells - len(reachable)

    # Phase 2: DFS on reachable domain
    visited = set()
    found_path = None
    step_count = 0
    depth_limit = 600

    def dfs(gx, gy, path, depth):
        nonlocal found_path, step_count
        if found_path or depth > depth_limit:
            return
        k = key_fn(gx, gy)
        if k not in reachable or k in visited:
            return
        visited.add(k)
        mx, my = unsnap(gx), unsnap(gy)
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
    return {"path": final_path, "cost": round(cost, 4), "algo_name": "Arc Consistency (AC-3)",
            "extra": f"Pruned: {pruned_count} | Reachable: {len(reachable)} | DFS Steps: {step_count}"}
