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

# Global storage for hazards
GLOBAL_HAZARDS = []
GLOBAL_SURVIVOR = None
GLOBAL_EXIT = None

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


class HillClimbing:
    def __init__(self, dmap: DisasterMap):
        self.map = dmap

    def find_path(self, sx, sy, ex, ey, max_iter=2000):
        cx, cy = sx, sy
        path   = [(cx,cy)]
        step   = 0.003

        def obj(x,y): return self.map.movement_cost(x,y,ex,ey) + self.map.total_danger(x,y)*20

        for _ in range(max_iter):
            best_nx, best_ny = cx, cy
            best_val = obj(cx, cy)
            
            # 8 directions for neighbors
            improved = False
            for angle in [i * (math.pi / 4) for i in range(8)]:
                nx = cx + math.cos(angle)*step
                ny = cy + math.sin(angle)*step
                
                val = obj(nx, ny)
                if val < best_val:
                    best_nx, best_ny = nx, ny
                    best_val = val
                    improved = True
                    
            if not improved:
                break # Local minimum reached
                
            cx, cy = best_nx, best_ny
            path.append((cx,cy))
            
            # Check if reached exit
            if math.hypot(ex-cx, ey-cy) <= step:
                break

        path.append((ex,ey))
        dedup = [path[0]]
        for p in path[1:]:
            if abs(p[0]-dedup[-1][0])>1e-6 or abs(p[1]-dedup[-1][1])>1e-6:
                dedup.append(p)
        total = sum(self.map.movement_cost(dedup[i][0],dedup[i][1],dedup[i+1][0],dedup[i+1][1])
                    for i in range(len(dedup)-1))
        return dedup, round(total,4)


class LocalBeamSearch:
    def __init__(self, dmap: DisasterMap):
        self.map = dmap

    def find_path(self, sx, sy, ex, ey, k=3, max_iter=1500):
        step = 0.003
        def obj(x,y): return self.map.movement_cost(x,y,ex,ey) + self.map.total_danger(x,y)*20

        # Each state is a tuple: (cost, x, y, path)
        states = [ (obj(sx, sy), sx, sy, [(sx,sy)]) for _ in range(k) ]
        
        for _ in range(max_iter):
            all_neighbors = []
            for cost, cx, cy, path in states:
                if math.hypot(ex-cx, ey-cy) <= step:
                    all_neighbors.append((cost, cx, cy, path))
                    continue
                    
                for angle in [i * (math.pi / 4) for i in range(8)]:
                    nx = cx + math.cos(angle)*step
                    ny = cy + math.sin(angle)*step
                    n_cost = obj(nx, ny)
                    all_neighbors.append((n_cost, nx, ny, path + [(nx,ny)]))
            
            all_neighbors.sort(key=lambda x: x[0])
            next_states = []
            
            for state in all_neighbors:
                if len(next_states) >= k:
                    break
                too_close = False
                for ns in next_states:
                    if math.hypot(state[1]-ns[1], state[2]-ns[2]) < step*0.1:
                        too_close = True
                        break
                if not too_close:
                    next_states.append(state)
            
            idx = 0
            while len(next_states) < k and idx < len(all_neighbors):
                if all_neighbors[idx] not in next_states:
                    next_states.append(all_neighbors[idx])
                idx += 1
                
            states = next_states
            
            if math.hypot(ex-states[0][1], ey-states[0][2]) <= step:
                break

        best_path = states[0][3] + [(ex,ey)]
        dedup = [best_path[0]]
        for p in best_path[1:]:
            if abs(p[0]-dedup[-1][0])>1e-6 or abs(p[1]-dedup[-1][1])>1e-6:
                dedup.append(p)
        total = sum(self.map.movement_cost(dedup[i][0],dedup[i][1],dedup[i+1][0],dedup[i+1][1])
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
    all_hazards = data.get("hazards", []) + GLOBAL_HAZARDS
    for h in all_hazards:
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
    if algo == "hc":
        hc = HillClimbing(dmap)
        path, cost = hc.find_path(
            survivors[0].x, survivors[0].y, exit_x, exit_y
        )
        algo_name = "Hill Climbing (Local Search)"
    elif algo == "sa":
        sa = SimulatedAnnealing(dmap)
        path, cost = sa.find_path(
            survivors[0].x, survivors[0].y, exit_x, exit_y
        )
        algo_name = "Simulated Annealing (Local Search)"
    elif algo == "beam":
        beam = LocalBeamSearch(dmap)
        path, cost = beam.find_path(
            survivors[0].x, survivors[0].y, exit_x, exit_y
        )
        algo_name = "Local Beam Search"
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
    severity = data.get("severity", 5)
    lat = data.get("lat", 13.7563)
    lng = data.get("lng", 100.5018)

    if severity <= 3:
        radius_m = 2000
        offset_m = 5
    elif severity <= 6:
        radius_m = 20000
        offset_m = 2000
    else:
        radius_m = 100000
        offset_m = 5000

    new_hazard = {
        "type": "flood",
        "lat": lat,
        "lng": lng,
        "severity": severity,
        "radius_m": radius_m,
        "confidence": data.get("confidence", 0),
        "level_label": data.get("level_label", "Unknown")
    }
    GLOBAL_HAZARDS.append(new_hazard)
    
    user_dist_m = float(data.get("user_dist_m", 0))
    user_dir = data.get("user_dir", "N")

    bearings = {
        "N": 0, "NE": 45, "E": 90, "SE": 135,
        "S": 180, "SW": 225, "W": 270, "NW": 315
    }
    bearing_deg = bearings.get(user_dir, 0)
    bearing_rad = math.radians(bearing_deg)

    total_dist_from_center_m = radius_m + user_dist_m
    
    # Calculate survivor coordinates
    lat_offset = (total_dist_from_center_m * math.cos(bearing_rad)) / 111320.0
    lng_offset = (total_dist_from_center_m * math.sin(bearing_rad)) / (111320.0 * math.cos(math.radians(lat)))
    survivor_lat = lat + lat_offset
    survivor_lng = lng + lng_offset

    # Calculate exit coordinates (further away in the same direction)
    exit_dist_from_center_m = total_dist_from_center_m + offset_m
    exit_lat_offset = (exit_dist_from_center_m * math.cos(bearing_rad)) / 111320.0
    exit_lng_offset = (exit_dist_from_center_m * math.sin(bearing_rad)) / (111320.0 * math.cos(math.radians(lat)))
    exit_lat = lat + exit_lat_offset
    exit_lng = lng + exit_lng_offset

    global GLOBAL_SURVIVOR, GLOBAL_EXIT
    GLOBAL_SURVIVOR = {"lat": survivor_lat, "lng": survivor_lng}
    GLOBAL_EXIT = {"lat": exit_lat, "lng": exit_lng}

    return jsonify({
        "status"   : "received",
        "hazard"   : new_hazard
    })

@app.route("/api/hazards", methods=["GET"])
def get_hazards():
    return jsonify({
        "hazards": GLOBAL_HAZARDS,
        "survivor": GLOBAL_SURVIVOR,
        "exit": GLOBAL_EXIT
    })


@app.route("/api/report_flood_v2", methods=["POST"])
def report_flood_v2():
    """
    รับข้อมูลจาก Map Picker:
    {
      "survivors": [{"id","lat","lng"}, ...],
      "hazards": [{"type","lat","lng","severity","radius_m"}, ...],
      "severity": int,
      "confidence": int,
      "level_label": str
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    survivors = data.get("survivors", [])
    hazards = data.get("hazards", [])
    severity = data.get("severity", 5)
    confidence = data.get("confidence", 0)
    level_label = data.get("level_label", "Unknown")

    # Clear existing hazards to avoid duplicate stacking
    GLOBAL_HAZARDS.clear()
    
    # Add hazards from map picker to global storage
    for h in hazards:
        new_hazard = {
            "type": h.get("type", "flood"),
            "lat": h.get("lat", 13.7563),
            "lng": h.get("lng", 100.5018),
            "severity": h.get("severity", severity),
            "radius_m": h.get("radius_m", 500),
            "confidence": confidence,
            "level_label": level_label
        }
        GLOBAL_HAZARDS.append(new_hazard)

    # Set first survivor as the global survivor
    global GLOBAL_SURVIVOR, GLOBAL_EXIT
    if survivors:
        first = survivors[0]
        GLOBAL_SURVIVOR = {"lat": first["lat"], "lng": first["lng"]}

        # Calculate exit point: ensure it is safely outside the hazard radius
        if hazards:
            nearest_h = min(hazards, key=lambda h:
                math.hypot(first["lat"] - h["lat"], first["lng"] - h["lng"]))
            
            h_lat = nearest_h.get("lat", 13.7563)
            h_lng = nearest_h.get("lng", 100.5018)
            h_rad_m = nearest_h.get("radius_m", 2000)
            sev = nearest_h.get("severity", 5)
            
            if sev <= 3:
                offset_m = 500
            elif sev <= 6:
                offset_m = 2000
            else:
                offset_m = 5000
                
            total_safe_dist_m = h_rad_m + offset_m
            
            # Approximations for lat/lng distance
            dlat = first["lat"] - h_lat
            # Adjust dlng by cos(lat) to make distance isotropic
            cos_lat = math.cos(math.radians(h_lat))
            dlng = (first["lng"] - h_lng) * cos_lat
            dist_deg = math.hypot(dlat, dlng)
            
            safe_dist_deg = total_safe_dist_m / 111320.0
            
            if dist_deg > 1e-9:
                dir_lat = dlat / dist_deg
                dir_lng = dlng / dist_deg
                
                # Make sure the exit is at least safe_dist_deg from center,
                # and if survivor is already safe, place exit further by offset
                final_dist_deg = max(dist_deg + (offset_m / 111320.0), safe_dist_deg)
                
                GLOBAL_EXIT = {
                    "lat": h_lat + dir_lat * final_dist_deg,
                    "lng": h_lng + (dir_lng * final_dist_deg) / cos_lat
                }
            else:
                # Survivor is exactly at hazard center, go North
                GLOBAL_EXIT = {
                    "lat": h_lat + safe_dist_deg,
                    "lng": h_lng
                }
        else:
            # No hazards, set exit 2km north
            GLOBAL_EXIT = {
                "lat": first["lat"] + (2000 / 111320.0),
                "lng": first["lng"]
            }

    return jsonify({
        "status": "received",
        "survivors_count": len(survivors),
        "hazards_count": len(hazards)
    })


if __name__ == "__main__":
    print("RescuOpt AI Server starting on http://localhost:5000")
    app.run(debug=True, port=5000)