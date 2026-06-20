"""
Disaster Survival AI Optimization System
=========================================
ระบบ AI สำหรับการวางแผนเส้นทางเอาชีวิตรอดจากภัยพิบัติ
โดยใช้หลัก Optimization:
  - Hill Climbing (Local Search)
  - Simulated Annealing
  - Linear Programming (Resource Allocation)

อ้างอิงจาก: lecture3_optimization.pdf
"""

import math
import random
import heapq
from dataclasses import dataclass, field
from typing import Optional
import time


# ──────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────

@dataclass
class Coordinate:
    x: float
    y: float
    name: str = ""

    def distance_to(self, other: "Coordinate") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"{self.name}({self.x:.1f},{self.y:.1f})"


@dataclass
class Survivor:
    id: str
    position: Coordinate
    health: int           # 0-100
    mobility: int         # 0-100 (100 = เดินปกติ)
    supplies: int         # 0-100 (เสบียง/น้ำ)

    @property
    def urgency_score(self) -> float:
        """คะแนนความเร่งด่วน: ยิ่งสูงยิ่งต้องช่วยก่อน"""
        return (100 - self.health) * 0.5 + (100 - self.mobility) * 0.3 + (100 - self.supplies) * 0.2


@dataclass
class Hazard:
    position: Coordinate
    type: str          # "fire", "flood", "collapse", "toxic"
    severity: int      # 1-10
    radius: float      # รัศมีอันตราย (หน่วยเดียวกับแผนที่)

    def danger_at(self, pos: Coordinate) -> float:
        """ระดับอันตราย ณ ตำแหน่งนั้น (0.0 - 1.0)"""
        d = self.position.distance_to(pos)
        if d >= self.radius:
            return 0.0
        return (self.severity / 10.0) * (1 - d / self.radius)


@dataclass(order=True)
class _PQItem:
    priority: float
    item: object = field(compare=False)


# ──────────────────────────────────────────────
# TERRAIN / MAP
# ──────────────────────────────────────────────

class DisasterMap:
    """ตัวแทนพื้นที่ภัยพิบัติ พร้อม hazard overlay"""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.hazards: list[Hazard] = []

    def add_hazard(self, hazard: Hazard):
        self.hazards.append(hazard)

    def total_danger(self, pos: Coordinate) -> float:
        """รวมอันตรายทุกแหล่ง (capped at 1.0)"""
        total = sum(h.danger_at(pos) for h in self.hazards)
        return min(total, 1.0)

    def movement_cost(self, a: Coordinate, b: Coordinate) -> float:
        """ต้นทุนการเคลื่อนที่ = ระยะทาง × (1 + danger_factor)"""
        dist = a.distance_to(b)
        danger_mid = self.total_danger(
            Coordinate((a.x + b.x) / 2, (a.y + b.y) / 2, "mid")
        )
        return dist * (1 + 3 * danger_mid)   # อันตรายทำให้เส้นทางแพงขึ้น 3×


# ──────────────────────────────────────────────
# 1. HILL CLIMBING — ค้นหาจุดปลอดภัยที่ดีที่สุดในพื้นที่
# ──────────────────────────────────────────────

class HillClimbSafeZoneFinder:
    """
    ใช้ Hill Climbing หาจุดรวมพล (safe zone) ที่เหมาะสมที่สุด
    Objective: minimize total_danger + maximize accessibility
    """

    def __init__(self, disaster_map: DisasterMap, step_size: float = 1.0):
        self.map = disaster_map
        self.step_size = step_size

    def _objective(self, pos: Coordinate) -> float:
        """
        ฟังก์ชัน objective (ต้องการ minimize)
        ยิ่งน้อยยิ่งดี = จุดปลอดภัย
        """
        danger = self.map.total_danger(pos)
        # เพิ่ม penalty ถ้าอยู่ใกล้ขอบแผนที่มากเกิน (ออกไม่ได้)
        edge_penalty = 0.0
        margin = 2.0
        if pos.x < margin or pos.x > self.map.width - margin:
            edge_penalty += 0.3
        if pos.y < margin or pos.y > self.map.height - margin:
            edge_penalty += 0.3
        return danger + edge_penalty

    def _neighbors(self, pos: Coordinate) -> list[Coordinate]:
        """8 ทิศทาง"""
        s = self.step_size
        offsets = [
            (s, 0), (-s, 0), (0, s), (0, -s),
            (s, s), (-s, s), (s, -s), (-s, -s),
        ]
        neighbors = []
        for dx, dy in offsets:
            nx, ny = pos.x + dx, pos.y + dy
            if 0 <= nx <= self.map.width and 0 <= ny <= self.map.height:
                neighbors.append(Coordinate(nx, ny, "candidate"))
        return neighbors

    def find_safe_zone(
        self,
        start: Coordinate,
        max_iter: int = 500,
        variant: str = "steepest"
    ) -> tuple[Coordinate, float, list[Coordinate]]:
        """
        Hill Climbing variants:
          steepest  — เลือก neighbor ที่ดีที่สุด
          stochastic — สุ่มจาก neighbor ที่ดีกว่า
          random_restart — ทำซ้ำหลายรอบ
        """
        best_pos, best_score = start, self._objective(start)
        path = [start]

        if variant == "random_restart":
            for _ in range(5):
                rand_start = Coordinate(
                    random.uniform(0, self.map.width),
                    random.uniform(0, self.map.height),
                    "rand_start",
                )
                pos, score, sub_path = self.find_safe_zone(rand_start, max_iter, "steepest")
                if score < best_score:
                    best_pos, best_score, path = pos, score, sub_path
            return best_pos, best_score, path

        current = start
        current_score = self._objective(current)

        for _ in range(max_iter):
            neighbors = self._neighbors(current)
            if variant == "stochastic":
                better = [n for n in neighbors if self._objective(n) < current_score]
                if not better:
                    break
                candidate = random.choice(better)
            else:  # steepest
                candidate = min(neighbors, key=self._objective)

            candidate_score = self._objective(candidate)
            if candidate_score >= current_score:
                break  # local minimum
            current = candidate
            current_score = candidate_score
            path.append(current)

        if current_score < best_score:
            best_pos, best_score = current, current_score

        return best_pos, best_score, path


# ──────────────────────────────────────────────
# 2. SIMULATED ANNEALING — หาเส้นทางอพยพที่ดีที่สุด
# ──────────────────────────────────────────────

class SimulatedAnnealingEvacPlanner:
    """
    ใช้ Simulated Annealing หาเส้นทาง waypoint ที่หลบภัยและสั้นที่สุด
    แก้ปัญหาคล้าย Traveling Salesman Problem (ต้องผ่านจุด checkpoint ให้ครบ)
    """

    def __init__(self, disaster_map: DisasterMap):
        self.map = disaster_map

    def _route_cost(self, start: Coordinate, waypoints: list[Coordinate], goal: Coordinate) -> float:
        """ต้นทุนรวมของเส้นทาง"""
        route = [start] + waypoints + [goal]
        return sum(
            self.map.movement_cost(route[i], route[i + 1])
            for i in range(len(route) - 1)
        )

    def plan_evacuation(
        self,
        start: Coordinate,
        checkpoints: list[Coordinate],
        goal: Coordinate,
        max_iter: int = 2000,
        initial_temp: float = 100.0,
    ) -> tuple[list[Coordinate], float, list[float]]:
        """
        หาลำดับ checkpoint ที่ดีที่สุดโดยใช้ SA
        Returns: (best_route, best_cost, cost_history)
        """
        if not checkpoints:
            cost = self.map.movement_cost(start, goal)
            return [start, goal], cost, [cost]

        current_order = list(range(len(checkpoints)))
        random.shuffle(current_order)
        current_cost = self._route_cost(
            start, [checkpoints[i] for i in current_order], goal
        )
        best_order = current_order[:]
        best_cost = current_cost
        cost_history = [current_cost]

        def temperature(t: int) -> float:
            return initial_temp * (0.995 ** t)

        for t in range(1, max_iter + 1):
            T = temperature(t)
            # สลับ 2 จุดแบบสุ่ม
            if len(current_order) < 2:
                break
            i, j = random.sample(range(len(current_order)), 2)
            neighbor_order = current_order[:]
            neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]
            neighbor_cost = self._route_cost(
                start, [checkpoints[k] for k in neighbor_order], goal
            )
            delta_e = neighbor_cost - current_cost   # ΔE = how much WORSE (we minimize)
            if delta_e < 0:
                # ดีกว่า — รับทันที
                current_order, current_cost = neighbor_order, neighbor_cost
            else:
                # แย่กว่า — รับด้วยความน่าจะเป็น e^(-ΔE/T)
                if T > 0 and random.random() < math.exp(-delta_e / T):
                    current_order, current_cost = neighbor_order, neighbor_cost

            if current_cost < best_cost:
                best_order, best_cost = current_order[:], current_cost

            cost_history.append(current_cost)

        best_route = [start] + [checkpoints[i] for i in best_order] + [goal]
        return best_route, best_cost, cost_history


# ──────────────────────────────────────────────
# 3. A* PATHFINDING — หาเส้นทางผ่านพื้นที่อันตราย
# ──────────────────────────────────────────────

class AStarPathfinder:
    """
    A* Search สำหรับนำทางในพื้นที่ที่มีอันตราย
    heuristic = Euclidean distance to goal
    """

    def __init__(self, disaster_map: DisasterMap, grid_size: float = 2.0):
        self.map = disaster_map
        self.grid_size = grid_size

    def _snap(self, pos: Coordinate) -> tuple[int, int]:
        g = self.grid_size
        return (round(pos.x / g), round(pos.y / g))

    def _unsnap(self, gx: int, gy: int, name: str = "") -> Coordinate:
        return Coordinate(gx * self.grid_size, gy * self.grid_size, name)

    def find_path(
        self, start: Coordinate, goal: Coordinate
    ) -> tuple[list[Coordinate], float]:
        g_start = self._snap(start)
        g_goal = self._snap(goal)

        open_set: list = []
        heapq.heappush(open_set, _PQItem(0.0, g_start))
        came_from: dict[tuple, tuple] = {}
        g_score: dict[tuple, float] = {g_start: 0.0}

        def h(gpos):
            p = self._unsnap(*gpos)
            return p.distance_to(goal)

        visited = set()

        while open_set:
            item = heapq.heappop(open_set)
            current = item.item
            if current in visited:
                continue
            visited.add(current)

            if current == g_goal:
                break

            cx, cy = current
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nx, ny = cx + dx, cy + dy
                if nx < 0 or ny < 0:
                    continue
                if nx * self.grid_size > self.map.width or ny * self.grid_size > self.map.height:
                    continue
                neighbor = (nx, ny)
                pos_a = self._unsnap(cx, cy)
                pos_b = self._unsnap(nx, ny)
                move_cost = self.map.movement_cost(pos_a, pos_b)
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + h(neighbor)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, _PQItem(f, neighbor))

        # Reconstruct path
        path_grid = []
        node = g_goal
        while node in came_from:
            path_grid.append(node)
            node = came_from[node]
        path_grid.append(g_start)
        path_grid.reverse()

        path = [self._unsnap(gx, gy, f"wp{i}") for i, (gx, gy) in enumerate(path_grid)]
        total_cost = g_score.get(g_goal, float("inf"))
        return path, total_cost


# ──────────────────────────────────────────────
# 4. LINEAR PROGRAMMING (GREEDY) — จัดสรรทรัพยากร
# ──────────────────────────────────────────────

class ResourceAllocator:
    """
    จัดสรรทรัพยากรช่วยเหลือโดยอิงหลัก Linear Programming
    Minimize: weighted urgency unmet
    Constraints: ทรัพยากรที่มีจำกัด
    """

    def allocate(
        self,
        survivors: list[Survivor],
        available_medkits: int,
        available_food: int,
        available_rescue_teams: int,
    ) -> dict[str, dict]:
        """
        จัดลำดับและจัดสรรทรัพยากรให้ผู้รอดชีวิต
        Returns: dict[survivor_id -> allocated_resources]
        """
        # เรียงตาม urgency (สูงสุดก่อน)
        sorted_survivors = sorted(survivors, key=lambda s: s.urgency_score, reverse=True)

        allocation: dict[str, dict] = {}
        remaining_medkits = available_medkits
        remaining_food = available_food
        remaining_teams = available_rescue_teams

        for s in sorted_survivors:
            alloc = {"medkit": 0, "food": 0, "rescue_team": 0}

            # ต้องการยา (health < 50)
            if s.health < 50 and remaining_medkits > 0:
                alloc["medkit"] = 1
                remaining_medkits -= 1

            # ต้องการอาหาร (supplies < 40)
            if s.supplies < 40 and remaining_food > 0:
                alloc["food"] = 1
                remaining_food -= 1

            # ต้องการทีมช่วยเหลือ (mobility < 30 หรือ health < 30)
            if (s.mobility < 30 or s.health < 30) and remaining_teams > 0:
                alloc["rescue_team"] = 1
                remaining_teams -= 1

            allocation[s.id] = alloc

        return allocation


# ──────────────────────────────────────────────
# 5. SURVIVAL AI ADVISOR — ระบบรวม
# ──────────────────────────────────────────────

class SurvivalAIAdvisor:
    """ระบบ AI หลักที่รวมทุก optimization algorithms"""

    DISASTER_TIPS = {
        "fire": [
            "🔥 เคลื่อนที่ทวนลม (upwind) ออกจากแนวไฟ",
            "🔥 หลีกเลี่ยงอาคารที่อาจพังถล่ม",
            "🔥 ปิดปากจมูกด้วยผ้าชุ่มน้ำกันควัน",
            "🔥 เดินต่ำกว่าระดับควัน (คลาน)",
        ],
        "flood": [
            "🌊 อย่าเดินลุยน้ำที่ไหลเร็ว แม้ตื้น",
            "🌊 มุ่งสู่พื้นที่สูง ห่างจากแหล่งน้ำ",
            "🌊 ระวังสายไฟฟ้าที่อาจตกน้ำ",
            "🌊 หากติดในอาคาร ขึ้นชั้นบนสุดและส่งสัญญาณขอความช่วยเหลือ",
        ],
        "collapse": [
            "🧱 DROP-COVER-HOLD: หมอบ-หลบ-ยึด",
            "🧱 ห่างจากกระจก ของหนัก ชั้นวางของ",
            "🧱 อย่าใช้ลิฟต์",
            "🧱 หากติดใต้ซากปรักหักพัง ส่งเสียงเคาะเป็นระยะ",
        ],
        "toxic": [
            "☣️ ปิดปากจมูก ทำหน้ากากจากเสื้อผ้า",
            "☣️ เคลื่อนที่ตั้งฉากกับทิศลม",
            "☣️ อย่าสัมผัสสิ่งที่อาจปนเปื้อน",
            "☣️ ล้างร่างกายด้วยน้ำสะอาดหากสัมผัสสาร",
        ],
    }

    def __init__(self, disaster_map: DisasterMap):
        self.map = disaster_map
        self.hill_climber = HillClimbSafeZoneFinder(disaster_map)
        self.sa_planner = SimulatedAnnealingEvacPlanner(disaster_map)
        self.pathfinder = AStarPathfinder(disaster_map)
        self.resource_allocator = ResourceAllocator()

    def analyze(
        self,
        survivors: list[Survivor],
        exit_point: Coordinate,
        checkpoints: list[Coordinate],
        resources: dict,
    ) -> dict:
        """
        วิเคราะห์สถานการณ์และออกคำแนะนำแบบ Optimization
        """
        results = {}

        # 1. หาจุดรวมพลที่ปลอดภัย
        centroid = Coordinate(
            sum(s.position.x for s in survivors) / len(survivors),
            sum(s.position.y for s in survivors) / len(survivors),
            "centroid",
        )
        safe_zone, danger_level, hc_path = self.hill_climber.find_safe_zone(
            centroid, variant="random_restart"
        )
        results["safe_zone"] = {
            "position": safe_zone,
            "danger_level": round(danger_level, 3),
            "path_length": len(hc_path),
            "algorithm": "Hill Climbing (Random Restart)",
        }

        # 2. วางแผนเส้นทางอพยพ
        evac_route, evac_cost, _ = self.sa_planner.plan_evacuation(
            centroid, checkpoints, exit_point
        )
        results["evacuation_route"] = {
            "route": evac_route,
            "total_cost": round(evac_cost, 2),
            "waypoints": len(evac_route),
            "algorithm": "Simulated Annealing",
        }

        # 3. หาเส้นทาง A* ไปยังทางออก
        astar_path, astar_cost = self.pathfinder.find_path(centroid, exit_point)
        results["astar_path"] = {
            "path": astar_path,
            "cost": round(astar_cost, 2),
            "steps": len(astar_path),
            "algorithm": "A* Search",
        }

        # 4. จัดสรรทรัพยากร
        allocation = self.resource_allocator.allocate(
            survivors,
            resources.get("medkits", 0),
            resources.get("food", 0),
            resources.get("rescue_teams", 0),
        )
        results["resource_allocation"] = {
            "allocation": allocation,
            "algorithm": "Greedy (Urgency-Priority LP)",
        }

        # 5. ระดับอันตรายรวม & คำแนะนำ
        hazard_types = {h.type for h in self.map.hazards}
        tips = []
        for ht in hazard_types:
            tips.extend(self.DISASTER_TIPS.get(ht, []))

        results["hazard_summary"] = {
            "types": list(hazard_types),
            "max_danger_at_centroid": round(self.map.total_danger(centroid), 3),
            "survival_tips": tips,
        }

        # 6. ลำดับความสำคัญผู้รอดชีวิต
        results["survivor_priority"] = sorted(
            [{"id": s.id, "urgency": round(s.urgency_score, 1), "position": s.position}
             for s in survivors],
            key=lambda x: x["urgency"],
            reverse=True,
        )

        return results

    def print_report(self, results: dict):
        sep = "═" * 60
        print(f"\n{sep}")
        print("  🚨 DISASTER SURVIVAL AI — OPTIMIZATION REPORT")
        print(sep)

        # Safe Zone
        sz = results["safe_zone"]
        print(f"\n📍 จุดรวมพลที่ปลอดภัย  [{sz['algorithm']}]")
        print(f"   ตำแหน่ง : {sz['position']}")
        print(f"   ระดับอันตราย : {sz['danger_level']} (0=ปลอดภัย, 1=อันตรายมาก)")

        # Evacuation
        ev = results["evacuation_route"]
        print(f"\n🗺️  เส้นทางอพยพ  [{ev['algorithm']}]")
        route_str = " → ".join(str(p) for p in ev["route"])
        print(f"   เส้นทาง : {route_str}")
        print(f"   ต้นทุนรวม : {ev['total_cost']}")

        # A*
        ap = results["astar_path"]
        print(f"\n🔍 เส้นทาง A*  [{ap['algorithm']}]")
        print(f"   จำนวนขั้น : {ap['steps']}  ต้นทุน : {ap['cost']}")

        # Resources
        ra = results["resource_allocation"]
        print(f"\n📦 การจัดสรรทรัพยากร  [{ra['algorithm']}]")
        for sid, alloc in ra["allocation"].items():
            parts = [f"{k}={v}" for k, v in alloc.items() if v > 0]
            print(f"   {sid}: {', '.join(parts) if parts else 'ไม่ได้รับทรัพยากร'}")

        # Survivors Priority
        print(f"\n👥 ลำดับความเร่งด่วน (urgency score)")
        for sp in results["survivor_priority"]:
            print(f"   [{sp['urgency']:5.1f}] {sp['id']} @ {sp['position']}")

        # Hazards & Tips
        hs = results["hazard_summary"]
        print(f"\n⚠️  ภัยพิบัติที่ตรวจพบ: {', '.join(hs['types'])}")
        print(f"   อันตราย ณ จุดศูนย์กลาง: {hs['max_danger_at_centroid']}")
        print(f"\n💡 คำแนะนำการเอาชีวิตรอด:")
        for tip in hs["survival_tips"]:
            print(f"   {tip}")

        print(f"\n{sep}\n")


# ──────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────

def main():
    random.seed(42)

    # ─── สร้างแผนที่ 50×50 ───
    dmap = DisasterMap(width=50, height=50)

    # เพิ่มภัยพิบัติ
    dmap.add_hazard(Hazard(Coordinate(20, 20, "fire_A"), "fire",     severity=8, radius=8))
    dmap.add_hazard(Hazard(Coordinate(35, 10, "flood_B"), "flood",   severity=6, radius=10))
    dmap.add_hazard(Hazard(Coordinate(10, 40, "collapse_C"), "collapse", severity=7, radius=6))
    dmap.add_hazard(Hazard(Coordinate(40, 35, "toxic_D"), "toxic",   severity=5, radius=7))

    # ─── ผู้รอดชีวิต ───
    survivors = [
        Survivor("Survivor-A", Coordinate(15, 25, "S_A"), health=70, mobility=90, supplies=50),
        Survivor("Survivor-B", Coordinate(30, 15, "S_B"), health=30, mobility=20, supplies=10),
        Survivor("Survivor-C", Coordinate(8,  38, "S_C"), health=55, mobility=60, supplies=30),
        Survivor("Survivor-D", Coordinate(42, 30, "S_D"), health=80, mobility=80, supplies=80),
        Survivor("Survivor-E", Coordinate(25, 45, "S_E"), health=10, mobility=15, supplies=5),
    ]

    # ─── จุด checkpoint และทางออก ───
    checkpoints = [
        Coordinate(12, 12, "CP-1_Supply"),
        Coordinate(45, 5,  "CP-2_HighGround"),
        Coordinate(28, 28, "CP-3_Medical"),
    ]
    exit_point = Coordinate(48, 48, "EXIT")

    # ─── ทรัพยากร ───
    resources = {"medkits": 3, "food": 2, "rescue_teams": 2}

    # ─── รัน AI ───
    ai = SurvivalAIAdvisor(dmap)
    print("⏳ กำลังคำนวณ optimization...")
    t0 = time.time()
    results = ai.analyze(survivors, exit_point, checkpoints, resources)
    elapsed = time.time() - t0
    print(f"✅ คำนวณเสร็จใน {elapsed:.2f} วินาที")

    ai.print_report(results)

    return results


if __name__ == "__main__":
    main()
