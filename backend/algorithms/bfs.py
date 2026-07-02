from backend.utils.geo import compute_bounds
from backend.services.danger_service import total_danger, movement_cost


def algo_bfs(start_m, goal_m, hazards, grid_step=40):
    b = compute_bounds(hazards, [], goal_m)
    snap = lambda v: round(v / grid_step)
    unsnap = lambda g: g * grid_step
    key_fn = lambda x, y: f"{x},{y}"

    sg = (snap(start_m["x"]), snap(start_m["y"]))
    gg = (snap(goal_m["x"]), snap(goal_m["y"]))
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

    queue = [(sg[0], sg[1])]
    came_from = {key_fn(sg[0], sg[1]): None}
    visited = {key_fn(sg[0], sg[1])}
    node_count = 0
    found = False

    while queue:
        cx, cy = queue.pop(0)
        node_count += 1

        if cx == gg[0] and cy == gg[1]:
            found = True
            break

        cmx, cmy = unsnap(cx), unsnap(cy)
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            nmx, nmy = unsnap(nx), unsnap(ny)
            if nmx < b["minX"] or nmx > b["maxX"] or nmy < b["minY"] or nmy > b["maxY"]:
                continue
            nk = key_fn(nx, ny)
            if nk in visited:
                continue
            if total_danger(nmx, nmy, hazards) > 0.6:
                continue
            visited.add(nk)
            came_from[nk] = key_fn(cx, cy)
            queue.append((nx, ny))

    path_grid = []
    if found:
        node = gg
        while node is not None:
            path_grid.append(node)
            nk = key_fn(node[0], node[1])
            prev = came_from.get(nk)
            if prev is None:
                break
            parts = prev.split(",")
            node = (int(parts[0]), int(parts[1]))
        path_grid.reverse()
    else:
        path_grid = [sg, gg]

    path_m = [{"x": start_m["x"], "y": start_m["y"]}]
    for gx, gy in path_grid:
        path_m.append({"x": unsnap(gx), "y": unsnap(gy)})

    dedup = [path_m[0]]
    for p in path_m[1:]:
        last = dedup[-1]
        if abs(p["x"] - last["x"]) > 0.01 or abs(p["y"] - last["y"]) > 0.01:
            dedup.append(p)

    cost = sum(movement_cost(dedup[i]["x"], dedup[i]["y"], dedup[i+1]["x"], dedup[i+1]["y"], hazards)
               for i in range(len(dedup) - 1))
    return {"path": dedup, "cost": round(cost, 4), "algo_name": "Breadth-First Search (BFS)",
            "extra": f"Nodes explored: {node_count} | Path found: {found}"}
