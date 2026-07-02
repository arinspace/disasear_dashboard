import math, random

from backend.utils.geo import compute_bounds
from backend.services.danger_service import total_danger, movement_cost


def algo_genetic(start_m, goal_m, hazards, grid_step=40):
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
        res = follow(pop[0])
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
