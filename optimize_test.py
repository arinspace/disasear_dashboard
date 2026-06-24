"""
CS50 AI Lecture 3 - Optimization & Pathfinding Visualizer
==========================================================
Demonstrates: Hill Climbing, Simulated Annealing, Genetic Algorithm,
              Backtracking Search, Arc Consistency (AC-3)
"""

import tkinter as tk
from tkinter import ttk, font
import random
import math
import collections
import time


# ──────────────────────────────────────────────
#  CONSTANTS & COLOUR PALETTE
# ──────────────────────────────────────────────
GRID_SIZE   = 15          # 15×15 grid
CELL_PX     = 42          # pixels per cell
PAD         = 4           # cell padding for inner rect

# Colours
C_BG        = "#1a1a2e"   # deep navy background
C_PANEL     = "#16213e"   # slightly lighter for panel
C_ACCENT    = "#0f3460"   # accent panels
C_GRID_LINE = "#0d2137"

C_EMPTY     = "#0d2137"
C_WALL      = "#e94560"
C_START     = "#4ecca3"
C_GOAL      = "#f9ca24"
C_VISITED   = "#1a4a6e"
C_FRONTIER  = "#2980b9"
C_PATH      = "#a29bfe"
C_CURRENT   = "#fd79a8"
C_PRUNED    = "#2d1b2e"
C_GA_POP    = "#fdcb6e"
C_GA_BEST   = "#00b894"
C_BACKTRACK = "#e17055"
C_TEXT_MAIN = "#eaeaea"
C_TEXT_DIM  = "#74b9ff"


# ──────────────────────────────────────────────
#  CELL STATE ENUM
# ──────────────────────────────────────────────
EMPTY = 0; WALL = 1; START = 2; GOAL = 3

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES  = ["UP", "DOWN", "LEFT", "RIGHT"]


# ──────────────────────────────────────────────
#  MAIN APPLICATION
# ──────────────────────────────────────────────
class PathfindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CS50 AI — Optimization & Pathfinding Visualizer")
        self.root.configure(bg=C_BG)
        self.root.resizable(False, False)

        # Grid state
        self.grid = [[EMPTY]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.start = None
        self.goal  = None

        # Interaction mode: "wall" | "start" | "goal"
        self.place_mode = tk.StringVar(value="wall")

        # Animation state
        self.running      = False
        self.after_id     = None
        self.anim_steps   = []   # list of (grid_overlay_dict, status_text) snapshots
        self.anim_index   = 0
        self.overlay      = {}   # {(r,c): colour}

        self._build_ui()
        self._draw_grid()

    # ── UI CONSTRUCTION ──────────────────────────────────
    def _build_ui(self):
        # ─ Left: canvas ─
        canvas_frame = tk.Frame(self.root, bg=C_BG)
        canvas_frame.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="n")

        canvas_w = GRID_SIZE * CELL_PX + 2
        canvas_h = GRID_SIZE * CELL_PX + 2
        self.canvas = tk.Canvas(canvas_frame, width=canvas_w, height=canvas_h,
                                bg=C_GRID_LINE, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self._on_left_click)
        self.canvas.bind("<B1-Motion>",       self._on_left_drag)
        self.canvas.bind("<Button-3>",        self._on_right_click)

        # ─ Right: controls ─
        ctrl = tk.Frame(self.root, bg=C_PANEL, width=300)
        ctrl.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="ns")
        ctrl.grid_propagate(False)

        pad = dict(padx=12, pady=5)

        # Title
        tk.Label(ctrl, text="CS50 AI", bg=C_PANEL, fg=C_ACCENT,
                 font=("Helvetica", 11, "bold")).pack(**pad, anchor="w")
        tk.Label(ctrl, text="Optimization Visualizer", bg=C_PANEL, fg=C_TEXT_MAIN,
                 font=("Helvetica", 14, "bold")).pack(padx=12, pady=(0,2), anchor="w")
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Place Mode ──
        tk.Label(ctrl, text="PLACE MODE", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")

        mode_frame = tk.Frame(ctrl, bg=C_PANEL)
        mode_frame.pack(padx=12, pady=2, fill="x")
        for text, val, color in [("Start (S)", "start", C_START),
                                   ("Goal (G)",  "goal",  C_GOAL),
                                   ("Wall (X)",  "wall",  C_WALL)]:
            rb = tk.Radiobutton(mode_frame, text=text, variable=self.place_mode,
                                value=val, bg=C_PANEL, fg=color, selectcolor=C_ACCENT,
                                activebackground=C_PANEL, activeforeground=color,
                                font=("Helvetica", 10, "bold"), cursor="hand2")
            rb.pack(anchor="w", pady=1)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Algorithm ──
        tk.Label(ctrl, text="ALGORITHM", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")

        self.algo_var = tk.StringVar(value="Hill Climbing")
        algo_menu = ttk.Combobox(ctrl, textvariable=self.algo_var, state="readonly",
                                 font=("Helvetica", 10),
                                 values=[
                                     "Hill Climbing",
                                     "Simulated Annealing",
                                     "Genetic Algorithm",
                                     "Backtracking Search",
                                     "Arc Consistency (AC-3)"
                                 ])
        algo_menu.pack(padx=12, pady=4, fill="x")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Speed ──
        tk.Label(ctrl, text="ANIMATION SPEED", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")

        spd_frame = tk.Frame(ctrl, bg=C_PANEL)
        spd_frame.pack(padx=12, pady=2, fill="x")
        tk.Label(spd_frame, text="Fast", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")
        self.speed_var = tk.IntVar(value=120)
        tk.Scale(spd_frame, variable=self.speed_var, from_=10, to=600,
                 orient="horizontal", bg=C_PANEL, fg=C_TEXT_MAIN,
                 troughcolor=C_ACCENT, highlightthickness=0,
                 showvalue=False).pack(side="left", fill="x", expand=True)
        tk.Label(spd_frame, text="Slow", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 8)).pack(side="left")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Buttons ──
        btn_cfg = dict(font=("Helvetica", 11, "bold"), bd=0, cursor="hand2",
                       pady=7, relief="flat")
        self.run_btn = tk.Button(ctrl, text="▶  RUN", bg="#00b894", fg="white",
                                 command=self._run, **btn_cfg)
        self.run_btn.pack(padx=12, pady=4, fill="x")

        self.stop_btn = tk.Button(ctrl, text="■  STOP", bg="#d63031", fg="white",
                                  command=self._stop, state="disabled", **btn_cfg)
        self.stop_btn.pack(padx=12, pady=4, fill="x")

        tk.Button(ctrl, text="↺  RESET GRID", bg=C_ACCENT, fg=C_TEXT_MAIN,
                  command=self._reset, **btn_cfg).pack(padx=12, pady=4, fill="x")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Legend ──
        tk.Label(ctrl, text="LEGEND", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        legend_items = [
            (C_START,    "Start Point"),
            (C_GOAL,     "Goal Point"),
            (C_WALL,     "Wall / Barrier"),
            (C_VISITED,  "Visited Cell"),
            (C_CURRENT,  "Current Position"),
            (C_PATH,     "Final / Best Path"),
            (C_FRONTIER, "Frontier / Candidate"),
            (C_BACKTRACK,"Backtracking"),
            (C_GA_POP,   "GA Population"),
            (C_GA_BEST,  "GA Best Path"),
            (C_PRUNED,   "Pruned (AC-3)"),
        ]
        leg_frame = tk.Frame(ctrl, bg=C_PANEL)
        leg_frame.pack(padx=12, pady=2, fill="x")
        for i, (color, label) in enumerate(legend_items):
            row_f = tk.Frame(leg_frame, bg=C_PANEL)
            row_f.grid(row=i//2, column=i%2, sticky="w", padx=2, pady=1)
            tk.Label(row_f, bg=color, width=2, height=1).pack(side="left", padx=(0,4))
            tk.Label(row_f, text=label, bg=C_PANEL, fg=C_TEXT_MAIN,
                     font=("Helvetica", 8)).pack(side="left")

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", padx=12, pady=6)

        # ── Status ──
        tk.Label(ctrl, text="STATUS", bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9, "bold")).pack(**pad, anchor="w")
        self.status_var = tk.StringVar(value="Ready. Place Start (S) and Goal (G) on the grid, add walls, then click RUN.")
        status_box = tk.Label(ctrl, textvariable=self.status_var,
                              bg=C_ACCENT, fg=C_TEXT_MAIN,
                              font=("Helvetica", 9), wraplength=260,
                              justify="left", anchor="nw", padx=8, pady=8)
        status_box.pack(padx=12, pady=4, fill="x")

        # ── Step counter ──
        self.step_var = tk.StringVar(value="Step: —")
        tk.Label(ctrl, textvariable=self.step_var, bg=C_PANEL, fg=C_TEXT_DIM,
                 font=("Helvetica", 9)).pack(padx=12, anchor="w")

    # ── GRID DRAWING ─────────────────────────────────────
    def _cell_coords(self, r, c):
        x0 = c * CELL_PX + PAD
        y0 = r * CELL_PX + PAD
        x1 = x0 + CELL_PX - PAD * 2
        y1 = y0 + CELL_PX - PAD * 2
        return x0, y0, x1, y1

    def _draw_grid(self):
        self.canvas.delete("all")
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._draw_cell(r, c)

    def _cell_color(self, r, c):
        if (r, c) in self.overlay:
            return self.overlay[(r, c)]
        state = self.grid[r][c]
        if state == WALL:  return C_WALL
        if state == START: return C_START
        if state == GOAL:  return C_GOAL
        return C_EMPTY

    def _draw_cell(self, r, c):
        x0, y0, x1, y1 = self._cell_coords(r, c)
        color = self._cell_color(r, c)
        tag = f"cell_{r}_{c}"
        self.canvas.delete(tag)
        # Rounded-rect effect via slightly smaller rect
        self.canvas.create_rectangle(x0, y0, x1, y1,
                                     fill=color, outline=C_GRID_LINE,
                                     width=1, tags=tag)
        state = self.grid[r][c]
        lbl = ""
        if state == START: lbl = "S"
        elif state == GOAL: lbl = "G"
        elif state == WALL: lbl = "✕"
        if lbl:
            self.canvas.create_text((x0+x1)//2, (y0+y1)//2,
                                    text=lbl,
                                    fill="white" if state == WALL else C_BG,
                                    font=("Helvetica", int(CELL_PX*0.38), "bold"),
                                    tags=tag)

    def _redraw_overlay(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                self._draw_cell(r, c)

    # ── MOUSE INTERACTION ────────────────────────────────
    def _rc_from_event(self, event):
        c = event.x // CELL_PX
        r = event.y // CELL_PX
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            return r, c
        return None, None

    def _place(self, r, c):
        mode = self.place_mode.get()
        if mode == "start":
            if self.start:
                sr, sc = self.start
                self.grid[sr][sc] = EMPTY
                self._draw_cell(sr, sc)
            self.start = (r, c)
            self.grid[r][c] = START
        elif mode == "goal":
            if self.goal:
                gr, gc = self.goal
                self.grid[gr][gc] = EMPTY
                self._draw_cell(gr, gc)
            self.goal = (r, c)
            self.grid[r][c] = GOAL
        else:  # wall toggle
            if self.grid[r][c] == WALL:
                self.grid[r][c] = EMPTY
            elif self.grid[r][c] == EMPTY:
                self.grid[r][c] = WALL
        self._draw_cell(r, c)

    def _on_left_click(self, event):
        if self.running: return
        r, c = self._rc_from_event(event)
        if r is not None:
            self._place(r, c)

    def _on_left_drag(self, event):
        if self.running: return
        r, c = self._rc_from_event(event)
        if r is not None and self.place_mode.get() == "wall":
            if self.grid[r][c] == EMPTY:
                self.grid[r][c] = WALL
                self._draw_cell(r, c)

    def _on_right_click(self, event):
        if self.running: return
        r, c = self._rc_from_event(event)
        if r is not None:
            if self.grid[r][c] == WALL:
                self.grid[r][c] = EMPTY
                self._draw_cell(r, c)

    # ── CONTROLS ─────────────────────────────────────────
    def _run(self):
        if not self.start or not self.goal:
            self.status_var.set("⚠  Please place both a Start (S) and Goal (G) on the grid first.")
            return
        self.running = True
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.overlay = {}

        algo = self.algo_var.get()
        if algo == "Hill Climbing":
            self.anim_steps = HillClimbing(self.grid, self.start, self.goal).solve()
        elif algo == "Simulated Annealing":
            self.anim_steps = SimulatedAnnealing(self.grid, self.start, self.goal).solve()
        elif algo == "Genetic Algorithm":
            self.anim_steps = GeneticAlgorithm(self.grid, self.start, self.goal).solve()
        elif algo == "Backtracking Search":
            self.anim_steps = BacktrackingSearch(self.grid, self.start, self.goal).solve()
        elif algo == "Arc Consistency (AC-3)":
            self.anim_steps = ArcConsistency(self.grid, self.start, self.goal).solve()

        self.anim_index = 0
        self._animate()

    def _animate(self):
        if not self.running:
            return
        if self.anim_index >= len(self.anim_steps):
            self.running = False
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            return

        overlay, status = self.anim_steps[self.anim_index]
        self.overlay = overlay
        self._redraw_overlay()
        self.status_var.set(status)
        self.step_var.set(f"Step: {self.anim_index + 1} / {len(self.anim_steps)}")
        self.anim_index += 1
        self.after_id = self.root.after(self.speed_var.get(), self._animate)

    def _stop(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("⏹  Stopped.")

    def _reset(self):
        self._stop()
        self.grid  = [[EMPTY]*GRID_SIZE for _ in range(GRID_SIZE)]
        self.start = None
        self.goal  = None
        self.overlay = {}
        self.anim_steps = []
        self.anim_index = 0
        self._draw_grid()
        self.status_var.set("Grid cleared. Place Start, Goal, and Walls.")
        self.step_var.set("Step: —")


# ──────────────────────────────────────────────
#  HELPER
# ──────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors(grid, r, c):
    result = []
    for dr, dc in DIRECTIONS:
        nr, nc = r+dr, c+dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr][nc] != WALL:
            result.append((nr, nc))
    return result

def in_bounds(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

def is_passable(grid, r, c):
    return in_bounds(r, c) and grid[r][c] != WALL


# ──────────────────────────────────────────────
#  ALGORITHM BASE
# ──────────────────────────────────────────────
class BaseAlgo:
    def __init__(self, grid, start, goal):
        # Deep copy grid so we don't mutate
        self.grid  = [row[:] for row in grid]
        self.start = start
        self.goal  = goal
        self.steps = []   # list of (overlay_dict, status_str)

    def snap(self, overlay, status):
        self.steps.append((dict(overlay), status))

    def solve(self):
        raise NotImplementedError

    def _mark_path(self, path, overlay):
        for cell in path:
            if cell != self.start and cell != self.goal:
                overlay[cell] = C_PATH


# ──────────────────────────────────────────────
#  1. HILL CLIMBING
# ──────────────────────────────────────────────
class HillClimbing(BaseAlgo):
    def solve(self):
        current   = self.start
        visited   = set()
        path      = [current]
        overlay   = {}

        self.snap(overlay, "Hill Climbing started. Moving toward the goal greedily.")

        for _ in range(GRID_SIZE * GRID_SIZE * 2):
            visited.add(current)
            nbrs = neighbors(self.grid, *current)
            if not nbrs:
                overlay[current] = C_CURRENT
                self.snap(overlay, "⛔ No passable neighbors! Hill Climbing is STUCK — no escape from local minimum.")
                break

            # Sort by Manhattan distance to goal
            nbrs.sort(key=lambda x: manhattan(x, self.goal))
            best_nbr  = nbrs[0]
            best_dist = manhattan(best_nbr, self.goal)
            cur_dist  = manhattan(current, self.goal)

            # Mark visited
            for v in visited:
                if v != self.start and v != self.goal:
                    overlay[v] = C_VISITED
            # Mark frontier
            for n in nbrs:
                if n not in visited and n != self.goal:
                    overlay[n] = C_FRONTIER

            overlay[current] = C_CURRENT

            if current == self.goal:
                self._mark_path(path, overlay)
                self.snap(overlay, "🎉 Goal reached! Hill Climbing succeeded.")
                return self.steps

            if best_dist >= cur_dist:
                # Stuck — local minimum
                overlay[current] = C_BACKTRACK
                self.snap(overlay,
                          f"⚠  LOCAL MINIMUM detected at {current}! "
                          f"All neighbors (dist={best_dist}) are ≥ current (dist={cur_dist}). "
                          "Hill Climbing CANNOT escape without restart.")
                break

            self.snap(overlay,
                      f"Moving {current} → {best_nbr}. "
                      f"Distance to goal: {cur_dist} → {best_dist}.")
            current = best_nbr
            path.append(current)

        self._mark_path(path, overlay)
        return self.steps


# ──────────────────────────────────────────────
#  2. SIMULATED ANNEALING
# ──────────────────────────────────────────────
class SimulatedAnnealing(BaseAlgo):
    def solve(self):
        current   = self.start
        visited   = set([current])
        path      = [current]
        overlay   = {}

        T         = 1.0    # initial temperature
        T_min     = 0.01
        alpha     = 0.92   # cooling rate
        step      = 0

        self.snap(overlay, f"Simulated Annealing started. Temperature T={T:.3f}.")

        while current != self.goal and T > T_min:
            nbrs = [n for n in neighbors(self.grid, *current)]
            if not nbrs:
                self.snap(overlay, "No neighbors — search ended.")
                break

            # Draw visited / frontier
            for v in visited:
                if v != self.start and v != self.goal:
                    overlay[v] = C_VISITED
            for n in nbrs:
                if n != self.goal:
                    overlay[n] = C_FRONTIER
            overlay[current] = C_CURRENT

            candidate = random.choice(nbrs)
            delta = manhattan(current, self.goal) - manhattan(candidate, self.goal)
            # delta > 0 means candidate is better (closer)

            if delta > 0:
                # Better move — always accept
                msg = (f"T={T:.3f} | Better move! {current}→{candidate} "
                       f"(Δdist={delta:+d}). Accepted.")
                accepted = True
            else:
                prob = math.exp(delta / T) if T > 0 else 0
                accepted = random.random() < prob
                if accepted:
                    msg = (f"T={T:.3f} | ⚡ WORSE move accepted! {current}→{candidate} "
                           f"(Δdist={delta:+d}, p={prob:.2f}). Escaping local min!")
                else:
                    msg = (f"T={T:.3f} | Worse move REJECTED. {current}→{candidate} "
                           f"(Δdist={delta:+d}, p={prob:.2f}).")

            self.snap(overlay, msg)

            if accepted:
                current = candidate
                path.append(current)
                visited.add(current)

            T    *= alpha
            step += 1

        # Final
        for v in visited:
            if v != self.start and v != self.goal:
                overlay[v] = C_VISITED
        overlay[current] = C_CURRENT

        if current == self.goal:
            self._mark_path(path, overlay)
            self.snap(overlay, f"🎉 Goal reached after {step} steps! T={T:.4f}.")
        else:
            self._mark_path(path, overlay)
            self.snap(overlay, f"Temperature cooled to {T:.4f}. Search ended at {current}.")
        return self.steps


# ──────────────────────────────────────────────
#  3. GENETIC ALGORITHM
# ──────────────────────────────────────────────
class GeneticAlgorithm(BaseAlgo):
    POP_SIZE   = 20
    PATH_LEN   = GRID_SIZE * 2
    GENERATIONS= 40
    MUTATION_R = 0.15

    def _random_path(self):
        return [random.randint(0,3) for _ in range(self.PATH_LEN)]

    def _follow(self, genome):
        """Walk genome from start, return final cell and cells visited."""
        r, c   = self.start
        cells  = [(r, c)]
        for gene in genome:
            dr, dc = DIRECTIONS[gene]
            nr, nc = r+dr, c+dc
            if is_passable(self.grid, nr, nc):
                r, c = nr, nc
            cells.append((r, c))
        return (r, c), cells

    def _fitness(self, genome):
        (r, c), cells = self._follow(genome)
        dist = manhattan((r,c), self.goal)
        # Penalise hitting walls (already handled by _follow skipping)
        # Reward getting close or reaching goal
        score = GRID_SIZE*3 - dist
        if (r,c) == self.goal:
            score += 100
        return score

    def solve(self):
        pop = [self._random_path() for _ in range(self.POP_SIZE)]
        overlay = {}

        for gen in range(self.GENERATIONS):
            # Evaluate
            scored = sorted(pop, key=self._fitness, reverse=True)
            best   = scored[0]
            _, best_cells = self._follow(best)
            bf = self._fitness(best)

            # Visualise population paths (pick 5 random)
            overlay = {}
            sample = random.sample(scored, min(5, len(scored)))
            for ind in sample:
                _, cells = self._follow(ind)
                for cell in cells:
                    if cell != self.start and cell != self.goal:
                        overlay[cell] = C_GA_POP
            # Draw best on top
            for cell in best_cells:
                if cell != self.start and cell != self.goal:
                    overlay[cell] = C_GA_BEST

            best_end, _ = self._follow(best)
            status = (f"Generation {gen+1}/{self.GENERATIONS} | "
                      f"Best fitness: {bf} | "
                      f"Best end: {best_end} | "
                      f"Dist to goal: {manhattan(best_end, self.goal)}")
            if best_end == self.goal:
                status = f"🎉 Generation {gen+1}: GOAL REACHED! Fitness={bf}"
            self.snap(overlay, status)

            if best_end == self.goal:
                break

            # Selection: top half
            parents = scored[:self.POP_SIZE//2]

            # Crossover + mutation
            children = list(parents)
            while len(children) < self.POP_SIZE:
                p1, p2 = random.sample(parents, 2)
                pt = random.randint(1, self.PATH_LEN-1)
                child = p1[:pt] + p2[pt:]
                # Mutation
                for i in range(len(child)):
                    if random.random() < self.MUTATION_R:
                        child[i] = random.randint(0,3)
                children.append(child)
            pop = children

        # Final best path
        scored = sorted(pop, key=self._fitness, reverse=True)
        _, best_cells = self._follow(scored[0])
        overlay = {}
        for cell in best_cells:
            if cell != self.start and cell != self.goal:
                overlay[cell] = C_PATH
        dist = manhattan(self._follow(scored[0])[0], self.goal)
        self.snap(overlay, f"GA complete. Final best path shown. Distance to goal: {dist}.")
        return self.steps


# ──────────────────────────────────────────────
#  4. BACKTRACKING SEARCH (DFS with backtracking)
# ──────────────────────────────────────────────
class BacktrackingSearch(BaseAlgo):
    def solve(self):
        overlay  = {}
        visited  = set()
        path     = []
        found    = [False]

        def dfs(cell):
            if found[0]:
                return
            r, c = cell
            if not in_bounds(r,c) or self.grid[r][c] == WALL or cell in visited:
                return

            visited.add(cell)
            path.append(cell)

            # Draw state
            ov = {}
            for v in visited:
                if v != self.start and v != self.goal:
                    ov[v] = C_VISITED
            for p in path:
                if p != self.start and p != self.goal:
                    ov[p] = C_FRONTIER
            ov[cell] = C_CURRENT

            if cell == self.goal:
                found[0] = True
                self._mark_path(path, ov)
                self.steps.append((dict(ov), f"🎉 GOAL FOUND at {cell}! Path length: {len(path)}."))
                return

            self.steps.append((dict(ov), f"Exploring {cell}. Path depth: {len(path)}."))

            nbrs = neighbors(self.grid, r, c)
            random.shuffle(nbrs)  # randomise direction order
            expanded = False
            for nbr in nbrs:
                if nbr not in visited:
                    expanded = True
                    dfs(nbr)
                    if found[0]:
                        return

            # Backtrack
            if not found[0]:
                path.pop()
                ov2 = {}
                for v in visited:
                    if v != self.start and v != self.goal:
                        ov2[v] = C_VISITED
                for p in path:
                    if p != self.start and p != self.goal:
                        ov2[p] = C_FRONTIER
                if path:
                    ov2[path[-1]] = C_CURRENT
                ov2[cell] = C_BACKTRACK
                self.steps.append((dict(ov2),
                                   f"⬅  BACKTRACKING from {cell}. Dead end — returning to {path[-1] if path else self.start}."))

        random.seed(42)
        dfs(self.start)
        if not found[0]:
            self.steps.append(({}, "No path found between Start and Goal."))
        return self.steps


# ──────────────────────────────────────────────
#  5. ARC CONSISTENCY (AC-3 inspired CSP pathfinding)
# ──────────────────────────────────────────────
class ArcConsistency(BaseAlgo):
    """
    Simulates AC-3-inspired constraint propagation:
    1. BFS from Goal backwards to compute reachability domain.
    2. Cells from which Goal is unreachable are PRUNED (domain wiped).
    3. Then run DFS only on unpruned cells, showing inference.
    """
    def solve(self):
        overlay = {}

        # ─ Phase 1: backward BFS from goal to find reachable cells ─
        reachable = set()
        queue = collections.deque([self.goal])
        reachable.add(self.goal)
        while queue:
            cell = queue.popleft()
            for nbr in neighbors(self.grid, *cell):
                if nbr not in reachable:
                    reachable.add(nbr)
                    queue.append(nbr)

        # Cells not in reachable = pruned
        pruned = set()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r][c] != WALL and (r,c) not in reachable:
                    pruned.add((r,c))

        # Animate pruning phase
        ov = {}
        for cell in pruned:
            ov[cell] = C_PRUNED
        self.snap(ov, "AC-3 Phase 1: Backward BFS from Goal. Identifying cells that CANNOT reach the Goal.")

        for cell in pruned:
            ov2 = dict(ov)
            ov2[cell] = C_PRUNED
            self.steps.append((dict(ov2),
                               f"🔴 Pruning {cell} — domain reduced: this cell cannot contribute to any valid path."))

        # Final prune snapshot
        ov_pruned = {cell: C_PRUNED for cell in pruned}
        self.snap(ov_pruned,
                  f"AC-3 propagation complete. {len(pruned)} cells pruned. "
                  f"Now running Backtracking only on consistent domain ({len(reachable)} cells).")

        # ─ Phase 2: DFS on unpruned domain ─
        visited = set()
        path    = []
        found   = [False]

        def dfs(cell):
            if found[0]: return
            if cell in pruned or cell in visited: return
            r, c = cell
            if not in_bounds(r,c) or self.grid[r][c] == WALL: return

            visited.add(cell)
            path.append(cell)

            ov = dict(ov_pruned)
            for v in visited:
                if v != self.start and v != self.goal and v not in pruned:
                    ov[v] = C_VISITED
            for p in path:
                if p != self.start and p != self.goal:
                    ov[p] = C_FRONTIER
            ov[cell] = C_CURRENT

            if cell == self.goal:
                found[0] = True
                self._mark_path(path, ov)
                self.steps.append((dict(ov), f"🎉 GOAL FOUND! AC-3 path length: {len(path)}. Much more direct than raw backtracking!"))
                return

            self.steps.append((dict(ov),
                               f"AC-3 DFS exploring {cell} (pruned cells skipped automatically). Depth: {len(path)}."))

            for nbr in neighbors(self.grid, r, c):
                if nbr not in pruned and nbr not in visited:
                    dfs(nbr)
                    if found[0]: return

            # Backtrack
            if not found[0]:
                path.pop()
                ov2 = dict(ov_pruned)
                for v in visited:
                    if v != self.start and v != self.goal and v not in pruned:
                        ov2[v] = C_VISITED
                for p in path:
                    if p != self.start and p != self.goal:
                        ov2[p] = C_FRONTIER
                if path: ov2[path[-1]] = C_CURRENT
                ov2[cell] = C_BACKTRACK
                self.steps.append((dict(ov2),
                                   f"⬅  AC-3 backtrack from {cell} — but pruned cells already eliminated bad branches!"))

        dfs(self.start)
        if not found[0]:
            self.steps.append(({}, "No path found (start/goal may be isolated by walls)."))
        return self.steps


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()

    # Style ttk
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground=C_ACCENT,
                    background=C_ACCENT,
                    foreground=C_TEXT_MAIN,
                    selectbackground=C_ACCENT,
                    selectforeground=C_TEXT_MAIN)
    style.configure("TSeparator", background=C_ACCENT)

    app = PathfindingApp(root)
    root.mainloop()