"""
RescuOpt AI — Flask Backend Server
====================================
รับผลจาก YOLO detection → รัน optimization → ส่ง JSON ให้ dashboard
"""

try:
    from flask import Flask, request, jsonify, send_from_directory  # type: ignore[import]
    from flask_cors import CORS  # type: ignore[import]
except ImportError as e:
    raise RuntimeError(
        "Missing dependency: install Flask and Flask-CORS with 'pip install flask flask-cors'"
    ) from e

import math, random, heapq, time, base64, os, json
from dataclasses import dataclass, field
from typing import Optional

app = Flask(__name__, static_folder=".")
CORS(app)

# ──────────────────────────────────────────────
# OPTIMIZATION CORE  (ย่อจาก disaster_ai.py)
# ──────────────────────────────────────────────

@dataclass
class Hazard:
    hazard_type: str
    x: float
    y: float
    severity: int
    radius: float

    def danger_at(self, px, py):
        dist = math.hypot(px - self.x, py - self.y)
        if dist >= self.radius:
            return 0.0
        return (self.severity / 10.0) * (1.0 - dist / self.radius)


@dataclass
class Survivor:
    survivor_id: str
    x: float
    y: float
    health: int
    mobility: int
    supplies: int

    @property
    def urgency_score(self):
        return ((100-self.health)*0.5 + (100-self.mobility)*0.3 + (100-self.supplies)*0.2)

    @property
    def status_label(self):
        u = self.urgency_score
        if u >= 60: return "CRITICAL"
        if u >= 35: return "URGENT"
        return "STABLE"


class DisasterMap:
    def __init__(self):
        self.hazards: list[Hazard] = []

    def add_hazard(self, h: Hazard):
        self.hazards.append(h)

    def total_danger(self, px, py):
        return min(sum(h.danger_at(px, py) for h in self.hazards), 1.0)

    def movement_cost(self, ax, ay, bx, by):
        dist = math.hypot(bx-ax, by-ay)
        mid  = self.total_danger((ax+bx)/2, (ay+by)/2)
        return dist * (1.0 + 4.0 * mid)


class AStarSearch:
    DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]

    def __init__(self, dmap: DisasterMap, grid_step=0.001):
        self.map = dmap
        self.gs  = grid_step   # ใช้ lat/lng โดยตรง

    def find_path(self, sx, sy, ex, ey):
        def snap(v, g): return round(v/g)
        def unsnap(g, s): return g * s
        g = self.gs
        sg = (snap(sx,g), snap(sy,g))
        eg = (snap(ex,g), snap(ey,g))

        open_h = []
        h0 = math.hypot(sx-ex, sy-ey)
        heapq.heappush(open_h, (h0, sg))
        came  = {}
        gsco  = {sg: 0.0}
        vis   = set()

        iters = 0
        while open_h and iters < 8000:
            iters += 1
            _, cur = heapq.heappop(open_h)
            if cur in vis: continue
            vis.add(cur)
            if cur == eg: break
            cx, cy = unsnap(cur[0],g), unsnap(cur[1],g)
            for dx,dy in self.DIRS:
                nb = (cur[0]+dx, cur[1]+dy)
                nx, ny = unsnap(nb[0],g), unsnap(nb[1],g)
                move  = self.map.movement_cost(cx,cy,nx,ny)
                ng    = gsco[cur] + move
                if nb not in gsco or ng < gsco[nb]:
                    gsco[nb] = ng
                    came[nb] = cur
                    h        = math.hypot(nx-ex, ny-ey)
                    heapq.heappush(open_h, (ng+h, nb))

        path_g = []
        node = eg
        while node in came:
            path_g.append(node)
            node = came[node]
        path_g.append(sg)
        path_g.reverse()

        path = [(sx,sy)] + [(unsnap(n[0],g), unsnap(n[1],g)) for n in path_g[1:]] + [(ex,ey)]
        dedup = [path[0]]
        for p in path[1:]:
            if abs(p[0]-dedup[-1][0])>1e-6 or abs(p[1]-dedup[-1][1])>1e-6:
                dedup.append(p)

        total = sum(self.map.movement_cost(dedup[i][0],dedup[i][1],
                                            dedup[i+1][0],dedup[i+1][1])
                    for i in range(len(dedup)-1))
        return dedup, round(total,4)


class SimulatedAnnealing:
    def __init__(self, dmap: DisasterMap):
        self.map = dmap

    def find_path(self, sx, sy, ex, ey, max_iter=1500, T0=0.01, cooling=0.995):
        cx, cy = sx, sy
        path   = [(cx,cy)]
        T      = T0
        step   = 0.003

        for _ in range(max_iter):
            T *= cooling
            angle = random.uniform(0, 2*math.pi)
            dist  = random.uniform(step*0.5, step)
            nx = cx + math.cos(angle)*dist
            ny = cy + math.sin(angle)*dist

            def obj(x,y): return self.map.movement_cost(x,y,ex,ey) + self.map.total_danger(x,y)*20
            dE = obj(nx,ny) - obj(cx,cy)
            if dE < 0 or (T > 1e-9 and random.random() < math.exp(-dE/T)):
                cx, cy = nx, ny
                path.append((cx,cy))

        path.append((ex,ey))
        dedup = [path[0]]
        for p in path[1:]:
            if abs(p[0]-dedup[-1][0])>1e-6 or abs(p[1]-dedup[-1][1])>1e-6:
                dedup.append(p)
        total = sum(self.map.movement_cost(dedup[i][0],dedup[i][1],dedup[i+1][0],dedup[i+1][1])
                    for i in range(len(dedup)-1))
        return dedup, round(total,4)


class ResourceAllocator:
    def allocate(self, survivors, medkits, food, rescue_teams):
        sorted_s = sorted(survivors, key=lambda s: s.urgency_score, reverse=True)
        result = {}
        for s in sorted_s:
            alloc = {"medkit":0,"food":0,"rescue_team":0}
            if s.health < 50 and medkits > 0:
                alloc["medkit"] = 1; medkits -= 1
            if s.supplies < 40 and food > 0:
                alloc["food"] = 1; food -= 1
            if (s.mobility < 30 or s.health < 30) and rescue_teams > 0:
                alloc["rescue_team"] = 1; rescue_teams -= 1
            result[s.survivor_id] = alloc
        return result


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

# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/api/optimize", methods=["POST"])
def optimize():
    """
    รับ JSON:
    {
      "user_lat": float, "user_lng": float,
      "exit_lat": float, "exit_lng": float,
      "hazards": [{"type","lat","lng","severity","radius_m"}, ...],
      "survivors": [{"id","lat","lng","health","mobility","supplies"},...],
      "algorithm": "astar"|"sa",
      "medkits": int, "food": int, "rescue_teams": int
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    dmap = DisasterMap()

    # แปลง lat/lng → coordinate scale (ใช้ lat/lng โดยตรง)
    for h in data.get("hazards", []):
        # แปลง radius จากเมตร → องศาประมาณ
        radius_deg = h.get("radius_m", 200) / 111320
        dmap.add_hazard(Hazard(
            hazard_type = h.get("type","flood"),
            x           = h["lng"],
            y           = h["lat"],
            severity    = h.get("severity", 7),
            radius      = radius_deg,
        ))

    survivors_in = data.get("survivors", [])
    if not survivors_in:
        # ใช้ user เป็น survivor เดียว
        survivors_in = [{
            "id": "User",
            "lat": data["user_lat"],
            "lng": data["user_lng"],
            "health": 80, "mobility": 80, "supplies": 60,
        }]

    survivors = [
        Survivor(s["id"], s["lng"], s["lat"],
                 s.get("health",80), s.get("mobility",80), s.get("supplies",60))
        for s in survivors_in
    ]

    exit_x = data["exit_lng"]
    exit_y = data["exit_lat"]
    algo   = data.get("algorithm", "astar")

    t0 = time.perf_counter()
    if algo == "sa":
        sa = SimulatedAnnealing(dmap)
        path, cost = sa.find_path(
            survivors[0].x, survivors[0].y, exit_x, exit_y
        )
        algo_name = "Simulated Annealing"
    else:
        astar = AStarSearch(dmap, grid_step=0.0008)
        path, cost = astar.find_path(
            survivors[0].x, survivors[0].y, exit_x, exit_y
        )
        algo_name = "A* Search"
    elapsed = time.perf_counter() - t0

    # danger stats
    dangers = [dmap.total_danger(p[0],p[1]) for p in path]
    danger_stats = {
        "max":  round(max(dangers),3) if dangers else 0,
        "avg":  round(sum(dangers)/len(dangers),3) if dangers else 0,
        "high_count": sum(1 for d in dangers if d > 0.4),
    }

    # resources
    alloc = ResourceAllocator().allocate(
        survivors,
        data.get("medkits",3),
        data.get("food",3),
        data.get("rescue_teams",2),
    )

    # tips
    hazard_types = list({h.hazard_type for h in dmap.hazards})
    tips = []
    for ht in hazard_types:
        for tip in SURVIVAL_TIPS.get(ht,[]):
            tips.append({"type": ht, "tip": tip})

    # path → lat/lng list for Leaflet
    path_latlng = [[p[1], p[0]] for p in path]

    # danger points บนเส้นทาง
    danger_points = []
    step = max(1, len(path)//30)
    for i in range(0, len(path), step):
        d = dangers[i]
        if d > 0.25:
            danger_points.append({
                "lat": path[i][1], "lng": path[i][0], "danger": round(d,3)
            })

    return jsonify({
        "algorithm"    : algo_name,
        "elapsed_sec"  : round(elapsed, 4),
        "path"         : path_latlng,
        "path_cost"    : cost,
        "waypoints"    : len(path),
        "danger_stats" : danger_stats,
        "exit_danger"  : round(dmap.total_danger(exit_x, exit_y), 3),
        "danger_points": danger_points,
        "tips"         : tips,
        "survivors"    : [
            {
                "id"       : s.survivor_id,
                "lat"      : s.y,
                "lng"      : s.x,
                "urgency"  : round(s.urgency_score,1),
                "status"   : s.status_label,
                "health"   : s.health,
                "mobility" : s.mobility,
                "supplies" : s.supplies,
                "resources": alloc.get(s.survivor_id,{}),
            }
            for s in sorted(survivors, key=lambda s: s.urgency_score, reverse=True)
        ],
    })


@app.route("/api/report_flood", methods=["POST"])
def report_flood():
    """
    รับ: {"lat", "lng", "severity", "confidence", "level_label"}
    เก็บ hazard ใหม่และส่งกลับ
    """
    data = request.json
    return jsonify({
        "status"   : "received",
        "hazard"   : {
            "type"    : "flood",
            "lat"     : data["lat"],
            "lng"     : data["lng"],
            "severity": data.get("severity", 5),
            "radius_m": int(200 + data.get("severity",5) * 30),
        }
    })


if __name__ == "__main__":
    print("RescuOpt AI Server starting on http://localhost:5000")
    app.run(debug=True, port=5000)