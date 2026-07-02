from backend.utils.geo import compute_bounds, heuristic_cost
from backend.utils.math_utils import MinHeap
from backend.services.danger_service import movement_cost


def algo_astar(start_m, goal_m, hazards, grid_step=40):
    b = compute_bounds(hazards, [], goal_m)
    snap = lambda v: round(v / grid_step)
    unsnap = lambda g: g * grid_step
    key_fn = lambda x, y: f"{x},{y}"

    sg = (snap(start_m["x"]), snap(start_m["y"]))
    gg = (snap(goal_m["x"]), snap(goal_m["y"]))
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

    open_set = MinHeap()
    h0 = heuristic_cost(start_m["x"], start_m["y"], goal_m["x"], goal_m["y"])
    open_set.push({"f": h0, "g": 0.0, "px": sg[0], "py": sg[1]})
    came_from = {}
    g_score = {key_fn(sg[0], sg[1]): 0.0}
    visited = set()
    node_count = 0

    while len(open_set) > 0:
        cur = open_set.pop()
        ck = key_fn(cur["px"], cur["py"])
        if ck in visited:
            continue
        visited.add(ck)
        node_count += 1

        if cur["px"] == gg[0] and cur["py"] == gg[1]:
            break

        cx = unsnap(cur["px"])
        cy = unsnap(cur["py"])

        for dx, dy in dirs:
            npx = cur["px"] + dx
            npy = cur["py"] + dy
            nx = unsnap(npx)
            ny = unsnap(npy)
            if nx < b["minX"] or nx > b["maxX"] or ny < b["minY"] or ny > b["maxY"]:
                continue
            nk = key_fn(npx, npy)
            move = movement_cost(cx, cy, nx, ny, hazards)
            ng = cur["g"] + move
            if nk not in g_score or ng < g_score[nk]:
                g_score[nk] = ng
                came_from[nk] = ck
                h = heuristic_cost(nx, ny, goal_m["x"], goal_m["y"])
                open_set.push({"f": ng + h, "g": ng, "px": npx, "py": npy})

    path_grid = []
    node = gg
    while key_fn(node[0], node[1]) in came_from:
        path_grid.append(node)
        prev = came_from[key_fn(node[0], node[1])]
        parts = prev.split(",")
        node = (int(parts[0]), int(parts[1]))
    path_grid.append(sg)
    path_grid.reverse()

    path_m = [{"x": start_m["x"], "y": start_m["y"]}]
    for gx, gy in path_grid:
        path_m.append({"x": unsnap(gx), "y": unsnap(gy)})
    path_m.append({"x": goal_m["x"], "y": goal_m["y"]})

    dedup = [path_m[0]]
    for p in path_m[1:]:
        last = dedup[-1]
        if abs(p["x"] - last["x"]) > 0.01 or abs(p["y"] - last["y"]) > 0.01:
            dedup.append(p)

    cost = sum(movement_cost(dedup[i]["x"], dedup[i]["y"], dedup[i+1]["x"], dedup[i+1]["y"], hazards)
               for i in range(len(dedup) - 1))
    return {"path": dedup, "cost": round(cost, 4), "algo_name": "A* Search", "extra": f"Nodes explored: {node_count}"}
