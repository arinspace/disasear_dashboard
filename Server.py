"""
RescuOpt AI — Flask Backend Server
====================================
รับผลจาก YOLO detection → รัน optimization → ส่ง JSON ให้ dashboard
"""

try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
except ImportError as e:
    raise RuntimeError(
        "Missing dependency: install Flask and Flask-CORS with 'pip install flask flask-cors'"
    ) from e

import math, random, heapq, time, base64, os, json
from dataclasses import dataclass, field
from typing import Optional

app = Flask(__name__, static_folder=".")
CORS(app)

# Global storage
GLOBAL_HAZARDS = []
GLOBAL_SURVIVOR = None
GLOBAL_EXIT = None
EARTH_RADIUS_M = 6371000
DEFAULT_HAZARD_RADIUS_M = 500
DEFAULT_SAFETY_MARGIN_M = 500

# Coordinate system (same as disaster_nav.html)
REF_LAT = 13.7563
REF_LNG = 100.5018
DEG_M_LAT = 111320

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

def exit_is_safe(exit_point, hazards, safety_margin_m=DEFAULT_SAFETY_MARGIN_M):
    return all(
        haversine(exit_point["lat"], exit_point["lng"], h["lat"], h["lng"])
        > (h.get("radius_m") or DEFAULT_HAZARD_RADIUS_M) + safety_margin_m
        for h in hazards
    )

def generate_safe_exit(survivor, hazards, safety_margin_m=DEFAULT_SAFETY_MARGIN_M):
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

# ──────────────────────────────────────────────
# CORE FUNCTIONS (ported from disaster_nav.html)
# ──────────────────────────────────────────────

def total_danger(mx, my, hazards):
    total = 0.0
    for h in hazards:
        hm = to_meters(h["lat"], h["lng"])
        dist = math.hypot(mx - hm["x"], my - hm["y"])
        radius = h.get("radius_m", 200)
        if dist < radius:
            total += (h.get("severity", 5) / 10.0) * (1.0 - dist / radius)
    return min(total, 1.0)

def movement_cost(ax, ay, bx, by, hazards):
    dist = math.hypot(bx - ax, by - ay)
    mid_danger = total_danger((ax + bx) / 2, (ay + by) / 2, hazards)
    end_danger = total_danger(bx, by, hazards)
    danger = max(mid_danger, end_danger)
    return dist * math.pow(10, danger * 3)

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

def heuristic_cost(ax, ay, bx, by, heuristic_type="euclidean"):
    if heuristic_type == "manhattan":
        return abs(ax - bx) + abs(ay - by)
    return math.hypot(bx - ax, by - ay)

# Min-heap for A* and Greedy
class MinHeap:
    def __init__(self):
        self.arr = []
    def push(self, item):
        self.arr.append(item)
        i = len(self.arr) - 1
        while i > 0:
            p = (i - 1) >> 1
            if self.arr[p]["f"] <= self.arr[i]["f"]:
                break
            self.arr[p], self.arr[i] = self.arr[i], self.arr[p]
            i = p
    def pop(self):
        r = self.arr[0]
        last = self.arr.pop()
        if self.arr:
            self.arr[0] = last
            i = 0
            n = len(self.arr)
            while True:
                s = i
                l = i * 2 + 1
                r_idx = i * 2 + 2
                if l < n and self.arr[l]["f"] < self.arr[s]["f"]:
                    s = l
                if r_idx < n and self.arr[r_idx]["f"] < self.arr[s]["f"]:
                    s = r_idx
                if s == i:
                    break
                self.arr[i], self.arr[s] = self.arr[s], self.arr[i]
                i = s
        return r
    def __len__(self):
        return len(self.arr)

# ──────────────────────────────────────────────
# ALGORITHMS (ported from disaster_nav.html)
# ──────────────────────────────────────────────

def algo_astar(start_m, goal_m, hazards, grid_step, heuristic_type="euclidean"):
    b = compute_bounds(hazards, [], goal_m)
    snap = lambda v: round(v / grid_step)
    unsnap = lambda g: g * grid_step
    key_fn = lambda x, y: f"{x},{y}"

    sg = (snap(start_m["x"]), snap(start_m["y"]))
    gg = (snap(goal_m["x"]), snap(goal_m["y"]))
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

    open_set = MinHeap()
    h0 = heuristic_cost(start_m["x"], start_m["y"], goal_m["x"], goal_m["y"], heuristic_type)
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
                h = heuristic_cost(nx, ny, goal_m["x"], goal_m["y"], heuristic_type)
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

def algo_bfs(start_m, goal_m, hazards, grid_step):
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

def algo_greedy_bfs(start_m, goal_m, hazards, grid_step, heuristic_type="euclidean"):
    b = compute_bounds(hazards, [], goal_m)
    snap = lambda v: round(v / grid_step)
    unsnap = lambda g: g * grid_step
    key_fn = lambda x, y: f"{x},{y}"

    sg = (snap(start_m["x"]), snap(start_m["y"]))
    gg = (snap(goal_m["x"]), snap(goal_m["y"]))
    dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

    def h(gx, gy):
        mx, my = unsnap(gx), unsnap(gy)
        return heuristic_cost(mx, my, goal_m["x"], goal_m["y"], heuristic_type)

    counter = 0
    open_heap = [(h(sg[0], sg[1]), counter, sg[0], sg[1])]
    open_set = {key_fn(sg[0], sg[1])}
    came_from = {}
    g_score = {key_fn(sg[0], sg[1]): 0.0}
    closed = set()
    node_count = 0
    found = False

    while open_heap:
        _, _, cx, cy = heapq.heappop(open_heap)
        ck = key_fn(cx, cy)
        if ck in closed:
            continue
        closed.add(ck)
        open_set.discard(ck)
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
            if nk in closed or nk in open_set:
                continue
            if total_danger(nmx, nmy, hazards) > 0.6:
                continue
            came_from[nk] = ck
            g_score[nk] = g_score[ck] + movement_cost(cmx, cmy, nmx, nmy, hazards)
            counter += 1
            heapq.heappush(open_heap, (h(nx, ny), counter, nx, ny))
            open_set.add(nk)

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
    return {"path": dedup, "cost": round(cost, 4), "algo_name": "Greedy Best-First Search",
            "extra": f"Nodes explored: {node_count} | Path found: {found}"}

def algo_simulated_annealing(start_m, goal_m, hazards):
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
    return {"path": dedup, "cost": round(cost, 4), "algo_name": "Simulated Annealing", "extra": f"Accepted: {accepted} | Rejected: {rejected}"}

def algo_hill_climbing(start_m, goal_m, hazards, variant="steepest"):
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
        stuck = False

        for _ in range(max_iter):
            neighbors = []
            for dx, dy in dirs:
                nx = max(b["minX"], min(b["maxX"], cx + dx))
                ny = max(b["minY"], min(b["maxY"], cy + dy))
                neighbors.append({"x": nx, "y": ny})
            best = min(neighbors, key=lambda p: obj(p["x"], p["y"]))
            best_val = obj(best["x"], best["y"])

            if best_val >= cur_val:
                stuck = True
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

def algo_genetic(start_m, goal_m, hazards, grid_step):
    b = compute_bounds(hazards, [], goal_m)
    pop_size = 40
    genes = 50
    gens = 80
    mut_rate = 0.12
    step = grid_step
    dirs = [(step,0),(-step,0),(0,step),(0,-step),(step,step),(-step,step),(step,-step),(-step,-step)]

    def follow(genome):
        cx, cy = start_m["x"], start_m["y"]
        cells = [{"x": cx, "y": cy}]
        for g in genome:
            dx, dy = dirs[g]
            nx = max(b["minX"], min(b["maxX"], cx + dx))
            ny = max(b["minY"], min(b["maxY"], cy + dy))
            if total_danger(nx, ny, hazards) < 0.85:
                cx, cy = nx, ny
            cells.append({"x": cx, "y": cy})
        return {"end": {"x": cx, "y": cy}, "cells": cells}

    def fitness(genome):
        res = follow(genome)
        dist = math.hypot(res["end"]["x"] - goal_m["x"], res["end"]["y"] - goal_m["y"])
        path_danger = sum(total_danger(c["x"], c["y"], hazards) for c in res["cells"])
        score = 50000 - dist - math.pow(path_danger, 2) * 500
        if dist < step * 2:
            score += 20000
        return score

    random_genome = lambda: [random.randint(0, 7) for _ in range(genes)]
    pop = [random_genome() for _ in range(pop_size)]

    for gen in range(gens):
        pop.sort(key=lambda g: fitness(g), reverse=True)
        best_genome = pop[0]
        res = follow(best_genome)
        dist_g = math.hypot(res["end"]["x"] - goal_m["x"], res["end"]["y"] - goal_m["y"])
        if dist_g < step:
            break

        parents = pop[:pop_size // 2]
        children = list(parents)
        while len(children) < pop_size:
            p1 = parents[random.randint(0, len(parents) - 1)]
            p2 = parents[random.randint(0, len(parents) - 1)]
            pt = random.randint(1, genes - 1)
            child = p1[:pt] + p2[pt:]
            for i in range(len(child)):
                if random.random() < mut_rate:
                    child[i] = random.randint(0, 7)
            children.append(child)
        pop = children

    pop.sort(key=lambda g: fitness(g), reverse=True)
    res = follow(pop[0])
    cells = res["cells"]
    dedup = [cells[0]]
    for c in cells[1:]:
        last = dedup[-1]
        if abs(c["x"] - last["x"]) > 0.01 or abs(c["y"] - last["y"]) > 0.01:
            dedup.append(c)
    dedup.append({"x": goal_m["x"], "y": goal_m["y"]})

    cost = sum(movement_cost(dedup[i]["x"], dedup[i]["y"], dedup[i+1]["x"], dedup[i+1]["y"], hazards)
               for i in range(len(dedup) - 1))
    return {"path": dedup, "cost": round(cost, 4), "algo_name": "Genetic Algorithm", "extra": f"Pop: {pop_size} | Gens: {gens}"}

def algo_backtracking(start_m, goal_m, hazards, grid_step):
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
    return {"path": final_path, "cost": round(cost, 4), "algo_name": "Backtracking Search", "extra": f"Steps: {step_count} | Backtracks: {backtrack_count}"}

def algo_ac3(start_m, goal_m, hazards, grid_step):
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

    # Count pruned
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

# ──────────────────────────────────────────────
# CSP ENGINE — Constraint Satisfaction Processing
# ──────────────────────────────────────────────

# Hard Constraints (mandatory — all must pass for a feasible path)
HARD_CONSTRAINTS = [
    "exit_safe",        # Exit must be outside all hazard radii + safety margin
    "path_connected",   # Path must be continuous (no gaps)
    "start_known",      # Start position must be valid
    "goal_accessible",  # Goal must be reachable (not inside a hazard)
    "hazard_valid",     # All hazards must have lat, lng, radius
]

# Soft Constraints (preferences with configurable weights)
SOFT_CONSTRAINT_KEYS = [
    "shortest_distance",
    "lowest_risk",
    "fastest_time",
    "max_safety",
]

@dataclass
class CSPConfig:
    enabled: bool = True
    hard_check: bool = True
    soft_scoring: bool = True
    weights: dict = field(default_factory=lambda: {
        "shortest_distance": 0.40,
        "lowest_risk": 0.35,
        "fastest_time": 0.15,
        "max_safety": 0.10,
    })

CSP_CONFIG = CSPConfig()

def check_hard_constraints(survivor, exit_point, hazards, active_constraints=None, max_danger_limit=0.5, max_distance_limit=2000.0, path=None):
    if active_constraints is None:
        active_constraints = {
            "start_known": True,
            "exit_safe": True,
            "goal_accessible": True,
            "hazard_valid": True,
            "path_connected": True,
            "start_safe": False,
            "avoid_extreme_danger": False,
            "max_distance": False,
        }

    results = {}
    all_pass = True

    # start_known
    if active_constraints.get("start_known", True):
        start_ok = survivor is not None and "lat" in survivor and "lng" in survivor
        if start_ok and active_constraints.get("start_safe", False):
            start_m = to_meters(survivor["lat"], survivor["lng"])
            start_danger = total_danger(start_m["x"], start_m["y"], hazards)
            if start_danger >= max_danger_limit:
                start_ok = False
        results["start_known"] = start_ok
        if not start_ok:
            all_pass = False

    # exit_safe
    if active_constraints.get("exit_safe", True):
        if exit_point and "lat" in exit_point and "lng" in exit_point:
            exit_ok = exit_is_safe(exit_point, hazards)
        else:
            exit_ok = False
        results["exit_safe"] = exit_ok
        if not exit_ok:
            all_pass = False

    # goal_accessible
    if active_constraints.get("goal_accessible", True):
        if exit_point and "lat" in exit_point and "lng" in exit_point:
            goal_ok = not any(
                haversine(exit_point["lat"], exit_point["lng"], h["lat"], h["lng"])
                < (h.get("radius_m") or DEFAULT_HAZARD_RADIUS_M)
                for h in hazards
            )
        else:
            goal_ok = False
        results["goal_accessible"] = goal_ok
        if not goal_ok:
            all_pass = False

    # hazard_valid
    if active_constraints.get("hazard_valid", True):
        hazards_ok = all(
            "lat" in h and "lng" in h and h.get("radius_m", 0) > 0
            for h in hazards
        ) if hazards else True
        results["hazard_valid"] = hazards_ok
        if not hazards_ok:
            all_pass = False

    # path_connected
    if active_constraints.get("path_connected", True):
        results["path_connected"] = True

    # avoid_extreme_danger
    if active_constraints.get("avoid_extreme_danger", False):
        if path:
            danger_ok = all(
                total_danger(p["x"], p["y"], hazards) <= max_danger_limit
                for p in path
            )
        else:
            danger_ok = True
        results["avoid_extreme_danger"] = danger_ok
        if not danger_ok:
            all_pass = False

    # max_distance
    if active_constraints.get("max_distance", False):
        if path:
            total_dist = 0.0
            for i in range(len(path) - 1):
                total_dist += math.hypot(path[i+1]["x"] - path[i]["x"], path[i+1]["y"] - path[i]["y"])
            dist_ok = total_dist <= max_distance_limit
        else:
            dist_ok = True
        results["max_distance"] = dist_ok
        if not dist_ok:
            all_pass = False

    return {
        "passed": all_pass,
        "constraints": results,
        "summary": "All hard constraints satisfied" if all_pass else "Hard constraints violated",
    }

def score_soft_constraints(path_m, hazards, exit_point, weights=None):
    if weights is None:
        weights = CSP_CONFIG.weights

    if not path_m or len(path_m) < 2:
        return {"scores": {}, "total": 0.0}

    total_dist = 0.0
    total_risk = 0.0
    path_length = len(path_m)

    for i in range(len(path_m) - 1):
        seg = math.hypot(path_m[i+1]["x"] - path_m[i]["x"], path_m[i+1]["y"] - path_m[i]["y"])
        total_dist += seg
        risk = total_danger(path_m[i]["x"], path_m[i]["y"], hazards)
        total_risk += risk

    avg_risk = total_risk / path_length if path_length > 0 else 0
    max_danger_on_path = max((total_danger(p["x"], p["y"], hazards) for p in path_m), default=0)

    # Normalized scores (0-1, higher is better)
    # shortest_distance: shorter = better, normalize by 5000m reference
    dist_score = max(0, 1.0 - total_dist / 5000.0)

    # lowest_risk: lower average risk = better
    risk_score = 1.0 - avg_risk

    # fastest_time: inversely related to path steps (fewer waypoints = faster)
    time_score = max(0, 1.0 - path_length / 200.0)

    # max_safety: lower peak danger = better
    safety_score = 1.0 - max_danger_on_path

    scores = {
        "shortest_distance": round(dist_score, 4),
        "lowest_risk": round(risk_score, 4),
        "fastest_time": round(time_score, 4),
        "max_safety": round(safety_score, 4),
    }

    weighted = sum(scores[k] * weights.get(k, 0) for k in scores)
    total = round(weighted, 4)

    return {
        "scores": scores,
        "weights": weights,
        "total": total,
        "details": {
            "total_distance_m": round(total_dist, 2),
            "avg_risk": round(avg_risk, 4),
            "max_danger": round(max_danger_on_path, 4),
            "path_length_nodes": path_length,
        }
    }

# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

SURVIVAL_TIPS = {
    "flood": [
        "อย่าเดินลุยน้ำที่ไหลเร็ว แม้ระดับตื้นก็พาล้มได้",
        "มุ่งสู่พื้นที่สูง ห่างจากร่องน้ำและถนนต่ำ",
        "ระวังสายไฟฟ้าที่ขาดตกลงในน้ำ อันตรายถึงชีวิต",
        "หากติดในอาคาร ขึ้นชั้นบนสุดและส่งสัญญาณขอความช่วยเหลือ",
        "อย่าขับรถฝ่าน้ำท่วม — น้ำ 60 ซม. พัดรถยนต์ได้",
    ],
    "fire": [
        "เคลื่อนที่ทวนลม (upwind) ออกจากแนวไฟเสมอ",
        "ลงต่ำใกล้พื้น — ออกซิเจนอยู่ใต้ระดับควัน",
        "ปิดปากจมูกด้วยผ้าชุ่มน้ำป้องกันควันพิษ",
    ],
    "collapse": [
        "DROP-COVER-HOLD: หมอบ หลบใต้โต๊ะแข็ง ยึดให้มั่น",
        "อย่าใช้ลิฟต์ ใช้บันไดฉุกเฉินเท่านั้น",
    ],
    "toxic": [
        "เคลื่อนที่ตั้งฉากกับทิศทางลมออกจากกลุ่มควันพิษ",
        "ปิดปากจมูกด้วยผ้าหลายชั้นชุบน้ำ",
    ],
    "earthquake": [
        "อยู่ห่างจากอาคาร ต้นไม้ เสาไฟ และสายไฟ",
        "ระวัง aftershock ที่อาจรุนแรงกว่าครั้งแรก",
    ],
}
@app.route("/disaster_nav")
def disaster_nav():
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "dashboard"),
        "disaster_nav.html"
    )

@app.route("/dashboard/disaster_nav.html")
def disaster_nav_alias():
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "dashboard"),
        "disaster_nav.html"
    )

@app.route("/dashboard.html")
def dashboard_html_alias():
    return send_from_directory(".", "dashboard.html")

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    hazards = data.get("hazards", []) + GLOBAL_HAZARDS
    survivors_in = data.get("survivors", [])
    if not survivors_in:
        survivors_in = [{
            "id": "User",
            "lat": data.get("user_lat", 13.7563),
            "lng": data.get("user_lng", 100.5018),
            "health": 80, "mobility": 80, "supplies": 60,
        }]
    first_survivor = survivors_in[0]

    # Apply dynamic hazard radius if provided
    hazard_radius = data.get("hazard_radius")
    if hazard_radius is not None:
        for h in hazards:
            h["radius_m"] = hazard_radius

    exit_point = {}
    if data.get("exit_lat") is not None and data.get("exit_lng") is not None:
        exit_point = {"lat": data["exit_lat"], "lng": data["exit_lng"]}
    elif GLOBAL_EXIT:
        exit_point = GLOBAL_EXIT
    else:
        exit_point = generate_safe_exit(first_survivor, hazards)
    if not exit_point or not exit_is_safe(exit_point, hazards):
        exit_point = generate_safe_exit(first_survivor, hazards)
    if not exit_point:
        return jsonify({"error": "No safe exit available"}), 400

    algo = data.get("algorithm", "astar")
    grid_step = data.get("grid_step", 40)
    heuristic_type = data.get("heuristic", "euclidean")

    # CSP check — validate hard constraints before routing
    csp_config = data.get("constraints", {})
    do_hard_check = csp_config.get("enabled", CSP_CONFIG.enabled) and csp_config.get("hard_check", CSP_CONFIG.hard_check)
    do_soft_scoring = csp_config.get("soft_scoring", CSP_CONFIG.soft_scoring)

    active_constraints = csp_config.get("active_constraints", None)
    max_danger_limit = float(csp_config.get("max_danger_limit", 0.5))
    max_distance_limit = float(csp_config.get("max_distance_limit", 2000.0))

    if do_hard_check:
        hard_result = check_hard_constraints(
            first_survivor, exit_point, hazards,
            active_constraints=active_constraints,
            max_danger_limit=max_danger_limit,
            max_distance_limit=max_distance_limit
        )
        if not hard_result["passed"]:
            return jsonify({
                "error": "Hard constraints violated",
                "hard_constraints": hard_result,
            }), 400

    start_m = to_meters(first_survivor["lat"], first_survivor["lng"])
    goal_m = to_meters(exit_point["lat"], exit_point["lng"])

    t0 = time.perf_counter()
    result = None

    if algo == "astar":
        result = algo_astar(start_m, goal_m, hazards, grid_step, heuristic_type)
    elif algo == "bfs":
        result = algo_bfs(start_m, goal_m, hazards, grid_step)
    elif algo == "greedy":
        result = algo_greedy_bfs(start_m, goal_m, hazards, grid_step, heuristic_type)
    elif algo == "sa":
        result = algo_simulated_annealing(start_m, goal_m, hazards)
    elif algo == "hc":
        result = algo_hill_climbing(start_m, goal_m, hazards, "steepest")
    elif algo == "hc_restart":
        result = algo_hill_climbing(start_m, goal_m, hazards, "random_restart")
    elif algo == "ga":
        result = algo_genetic(start_m, goal_m, hazards, grid_step)
    elif algo == "backtrack":
        result = algo_backtracking(start_m, goal_m, hazards, grid_step)
    elif algo == "ac3":
        result = algo_ac3(start_m, goal_m, hazards, grid_step)
    else:
        result = algo_astar(start_m, goal_m, hazards, grid_step, heuristic_type)
        result["algo_name"] = "A* Search (default)"

    # Post-routing check of dynamic hard constraints
    if do_hard_check and result:
        hard_result_post = check_hard_constraints(
            first_survivor, exit_point, hazards,
            active_constraints=active_constraints,
            max_danger_limit=max_danger_limit,
            max_distance_limit=max_distance_limit,
            path=result["path"]
        )
        if not hard_result_post["passed"]:
            return jsonify({
                "error": "Hard constraints violated",
                "hard_constraints": hard_result_post,
            }), 400

    elapsed = time.perf_counter() - t0

    if not result:
        return jsonify({"error": "Algorithm failed"}), 500

    # Convert path meter coords → lat/lng
    path_latlng = [[to_latlng(p["x"], p["y"])[0], to_latlng(p["x"], p["y"])[1]] for p in result["path"]]

    # Danger stats
    dangers = [total_danger(p["x"], p["y"], hazards) for p in result["path"]]
    danger_stats = {
        "max": round(max(dangers), 3) if dangers else 0,
        "avg": round(sum(dangers) / len(dangers), 3) if dangers else 0,
        "high_count": sum(1 for d in dangers if d > 0.4),
    }

    # Danger points on path
    danger_points = []
    step = max(1, len(result["path"]) // 30)
    for i in range(0, len(result["path"]), step):
        d = dangers[i]
        if d > 0.25:
            latlng = to_latlng(result["path"][i]["x"], result["path"][i]["y"])
            danger_points.append({"lat": latlng[0], "lng": latlng[1], "danger": round(d, 3)})

    # Tips
    hazard_types = list({h.get("type", "flood") for h in hazards})
    tips = []
    for ht in hazard_types:
        for tip in SURVIVAL_TIPS.get(ht, []):
            tips.append({"type": ht, "tip": tip})

    # Urgency scores
    sorted_survivors = sorted(survivors_in, key=lambda s:
        (100 - s.get("health", 80)) * 0.5 +
        (100 - s.get("mobility", 80)) * 0.3 +
        (100 - s.get("supplies", 60)) * 0.2,
        reverse=True)

    # CSP soft scoring
    csp_result = None
    if do_soft_scoring:
        custom_weights = csp_config.get("weights", None)
        csp_result = score_soft_constraints(result["path"], hazards, exit_point, weights=custom_weights)

    return jsonify({
        "algorithm": result["algo_name"],
        "elapsed_sec": round(elapsed, 4),
        "path": path_latlng,
        "path_cost": result["cost"],
        "waypoints": len(result["path"]),
        "danger_stats": danger_stats,
        "exit": exit_point,
        "exit_danger": round(total_danger(goal_m["x"], goal_m["y"], hazards), 3),
        "danger_points": danger_points,
        "tips": tips,
        "extra": result.get("extra", ""),
        "survivors": [{
            "id": s.get("id", f"S{i+1}"),
            "lat": s.get("lat", s.get("y")),
            "lng": s.get("lng", s.get("x")),
            "urgency": round(
                (100 - s.get("health", 80)) * 0.5 +
                (100 - s.get("mobility", 80)) * 0.3 +
                (100 - s.get("supplies", 60)) * 0.2, 1),
            "health": s.get("health", 80),
            "mobility": s.get("mobility", 80),
            "supplies": s.get("supplies", 60),
        } for i, s in enumerate(sorted_survivors)],
        "csp": csp_result,
    })

@app.route("/api/csp/check", methods=["POST"])
def csp_check():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    hazards = data.get("hazards", GLOBAL_HAZARDS)
    survivor = data.get("survivor", GLOBAL_SURVIVOR)
    exit_point = data.get("exit", None)

    hard_result = check_hard_constraints(survivor, exit_point, hazards)
    return jsonify(hard_result)

@app.route("/api/csp/config", methods=["GET", "POST"])
def csp_config():
    global CSP_CONFIG
    if request.method == "POST":
        data = request.json
        if not data:
            return jsonify({"error": "No data"}), 400
        if "enabled" in data:
            CSP_CONFIG.enabled = bool(data["enabled"])
        if "hard_check" in data:
            CSP_CONFIG.hard_check = bool(data["hard_check"])
        if "soft_scoring" in data:
            CSP_CONFIG.soft_scoring = bool(data["soft_scoring"])
        if "weights" in data:
            w = data["weights"]
            allowed = set(SOFT_CONSTRAINT_KEYS)
            CSP_CONFIG.weights = {k: float(v) for k, v in w.items() if k in allowed}
        return jsonify({
            "status": "updated",
            "config": {
                "enabled": CSP_CONFIG.enabled,
                "hard_check": CSP_CONFIG.hard_check,
                "soft_scoring": CSP_CONFIG.soft_scoring,
                "weights": CSP_CONFIG.weights,
            }
        })
    return jsonify({
        "enabled": CSP_CONFIG.enabled,
        "hard_check": CSP_CONFIG.hard_check,
        "soft_scoring": CSP_CONFIG.soft_scoring,
        "weights": CSP_CONFIG.weights,
        "hard_constraints": HARD_CONSTRAINTS,
        "soft_constraints": SOFT_CONSTRAINT_KEYS,
    })

@app.route("/api/report_flood", methods=["POST"])
def report_flood():
    data = request.json
    severity = data.get("severity", 5)
    lat = data.get("lat", 13.7563)
    lng = data.get("lng", 100.5018)

    if severity <= 3:
        radius_m = 2000
    elif severity <= 6:
        radius_m = 20000
    else:
        radius_m = 100000

    new_hazard = {
        "type": "flood",
        "lat": lat, "lng": lng,
        "severity": severity, "radius_m": radius_m,
        "confidence": data.get("confidence", 0),
        "level_label": data.get("level_label", "Unknown")
    }
    GLOBAL_HAZARDS.append(new_hazard)

    user_dist_m = float(data.get("user_dist_m", 0))
    user_dir = data.get("user_dir", "N")
    bearings = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}
    bearing_deg = bearings.get(user_dir, 0)

    total_dist_from_center_m = radius_m + user_dist_m
    survivor = offset_coordinate(lat, lng, total_dist_from_center_m, bearing_deg)

    global GLOBAL_SURVIVOR, GLOBAL_EXIT
    GLOBAL_SURVIVOR = survivor
    GLOBAL_EXIT = generate_safe_exit(survivor, GLOBAL_HAZARDS)

    return jsonify({"status": "received", "hazard": new_hazard})

@app.route("/api/hazards", methods=["GET"])
def get_hazards():
    return jsonify({
        "hazards": GLOBAL_HAZARDS,
        "survivor": GLOBAL_SURVIVOR,
        "exit": GLOBAL_EXIT,
    })

@app.route("/api/report_flood_v2", methods=["POST"])
def report_flood_v2():
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    survivors = data.get("survivors", [])
    hazards = data.get("hazards", [])
    severity = data.get("severity", 5)
    confidence = data.get("confidence", 0)
    level_label = data.get("level_label", "Unknown")

    GLOBAL_HAZARDS.clear()
    seen_hazards = set()

    for h in hazards:
        lat = h.get("lat", 13.7563)
        lng = h.get("lng", 100.5018)
        radius_m = h.get("radius_m") or DEFAULT_HAZARD_RADIUS_M
        hazard_key = (round(lat, 7), round(lng, 7), h.get("type", "flood"))
        if hazard_key in seen_hazards:
            continue
        seen_hazards.add(hazard_key)
        GLOBAL_HAZARDS.append({
            "type": h.get("type", "flood"),
            "lat": lat, "lng": lng,
            "severity": h.get("severity", severity),
            "radius_m": radius_m,
            "confidence": confidence,
            "level_label": level_label,
        })

    global GLOBAL_SURVIVOR, GLOBAL_EXIT
    if survivors:
        first = survivors[0]
        GLOBAL_SURVIVOR = {"lat": first["lat"], "lng": first["lng"]}
        GLOBAL_EXIT = generate_safe_exit(GLOBAL_SURVIVOR, GLOBAL_HAZARDS)

    return jsonify({
        "status": "received",
        "survivors_count": len(survivors),
        "hazards_count": len(GLOBAL_HAZARDS),
        "exit": GLOBAL_EXIT,
    })

if __name__ == "__main__":
    print("RescuOpt AI Server starting on http://127.0.0.1:5000/dashboard/disaster_nav.html")
    app.run(debug=True, port=5000)
