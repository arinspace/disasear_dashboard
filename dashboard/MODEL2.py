"""
Disaster Survival AI - Route Optimization Model
===============================================
AI system for computing optimal evacuation routes and survival strategies.
Uses optimization techniques including Simulated Annealing and Linear Programming.

Classes:
    DisasterSurvivalAI: Main class for route optimization and survival recommendations
"""

import math
import random
from scipy.optimize import linprog


class DisasterSurvivalAI:
    """Optimization system for disaster evacuation route planning.
    
    Uses cost-based optimization to find the safest evacuation zone considering:
    - Distance to safety
    - Proximity to disaster
    - Severity level of disaster
    
    Then uses Linear Programming to allocate optimal time and resources for survival activities.
    
    Attributes:
        user_loc (tuple): Current user position (x, y)
        disaster_loc (tuple): Disaster center position (x, y)
        severity (int): Disaster severity level 1-5
        safe_zones (list): Available evacuation zones [(x1, y1), (x2, y2), ...]
    """

    def __init__(self, user_loc, disaster_loc, severity, safe_zones):
        """Initialize Disaster Survival AI system.
        
        Args:
            user_loc (tuple): User coordinates (x, y)
            disaster_loc (tuple): Disaster center coordinates (x, y)
            severity (int): Disaster severity 1-5 (5 = most severe)
            safe_zones (list): Available safe zone coordinates
        """

    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points.
        
        Args:
            p1 (tuple): First point (x, y)
            p2 (tuple): Second point (x, y)
            
        Returns:
            float: Euclidean distance
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def optimize_route(self):
        """Find optimal evacuation route using cost optimization.
        
        Cost function considers:
        - Distance to safe zone
        - Danger penalty based on proximity to disaster and severity
        
        Returns:
            tuple: (best_zone, best_cost)
                - best_zone: Coordinates of recommended safe zone
                - best_cost: Computed safety score (lower is better)
        """
        best_zone = None
        best_cost = float("inf")

        # ค้นหาโหนด/ศูนย์อพยพรอบๆ ตัว (State-space Landscape)
        for zone in self.safe_zones:
            dist_to_safe = self._calculate_distance(self.user_loc, zone)
            dist_to_disaster = self._calculate_distance(
                zone, self.disaster_loc
            )

            # Cost Function: ยิ่งใกล้ภัยพิบัติยิ่ง Cost สูง, ยิ่งใกล้จุดปลอดภัยยิ่ง Cost ต่ำ
            # หากความรุนแรง (severity) สูง จะเพิ่มน้ำหนักความตระหนักต่อภัยพิบัติมากขึ้น
            danger_penalty = (self.severity * 10) / (dist_to_disaster + 0.1)
            total_cost = dist_to_safe + danger_penalty

            if total_cost < best_cost:
                best_cost = total_cost
                best_zone = zone

        return best_zone, best_cost

    def optimize_survival_advice(self):
        """Allocate survival resources using Linear Programming.
        
        Optimizes time allocation among three survival activities:
        1. Evacuation (10 minutes)
        2. Preparation of supplies (20 minutes)
        3. Temporary shelter finding (5 minutes)
        
        Resources and priorities adjust based on disaster severity.
        
        Returns:
            list: Selected survival advice strings based on LP optimization
        
        Algorithm:
            Uses scipy.optimize.linprog to maximize utility function subject to time constraints
        """
        # สมมติมีคำแนะนำ 3 แบบ: [1. อพยพทันที, 2. จัดเตรียมสิ่งของ, 3. หาสถานที่หลบซ่อนชั่วคราว]
        # คะแนนความสำคัญ (Utility) ของแต่ละคำแนะนำตามระดับความรุนแรง
        if self.severity >= 4:
            c = [-10, -2, -1]  # ต้องการ Maximize (ใส่ลบเพราะ linprog ทำ Minimize) อพยพสำคัญสุด
            time_limit = 15  # มีเวลาจำกัดแค่ 15 นาทีในการเตรียมตัว
        else:
            c = [-5, -8, -4]  # ความรุนแรงต่ำ การจัดเตรียมสิ่งของสำคัญกว่า
            time_limit = 45  # มีเวลา 45 นาที

        # ข้อจำกัดเรื่องเวลาที่ใช้ในแต่ละกิจกรรม (นาที): อพยพ(10นาที), จัดของ(20นาที), หลบซ่อน(5นาที)
        A = [[10, 20, 5]]
        b = [time_limit]

        # ขอบเขต (เลือกทำได้ตั้งแต่ 0 ถึง 1 ส่วนความทุ่มเท)
        bounds = [(0, 1), (0, 1), (0, 1)]

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        # แปลงผลลัพธ์ LP ออกมาเป็นคำอธิบายคำแนะนำ
        advice_pool = [
            "⚠️ [อพยพทันที] มุ่งหน้าไปตามเส้นทางที่ระบบคำนวณห้ามรีรอ",
            "🎒 [เตรียมเสบียง] หยิบเฉพาะน้ำดื่ม ยารักษาโรค และเอกสารสำคัญเท่านั้น",
            "🧱 [หาที่กำบัง] หากขยับตัวไม่ได้ ให้หลบในส่วนโครงสร้างตึกที่แข็งแรงที่สุด",
        ]

        selected_advice = []
        for idx, value in enumerate(res.x):
            if value > 0.3:  # ถ้าโมเดล Optimization เลือกกิจกรรมนั้นมากกว่า 30%
                selected_advice.append(advice_pool[idx])

        return selected_advice


# --- การทดสอบระบบ (Execution Example) ---
# Input: พิกัดผู้ใช้, พิกัดภัยพิบัติ, ระดับความรุนแรง, พิกัดศูนย์อพยพที่มีให้เลือก
ai_system = DisasterSurvivalAI(
    user_loc=(0, 0),
    disaster_loc=(3, 3),
    severity=5,  # รุนแรงมากระดับ 5
    safe_zones=[(1, 5), (4, 4), (10, 2)],  # (4,4) อยู่ใกล้ภัยพิบัติเกินไปเสี่ยงอันตราย
)

best_destination, safety_score = ai_system.optimize_route()
optimal_advices = ai_system.optimize_survival_advice()

# --- แสดงผลลัพธ์ (Output) ---
print(f"=== ผลลัพธ์จาก RescuOpt AI (Severity Level: {ai_system.severity}) ===")
print(
    f"📍 เส้นทางเอาตัวรอดที่ดีที่สุด: จากจุดปัจจุบัน (0,0) มุ่งหน้าไปยัง พิกัด {best_destination}"
)
print(
    f"🛡️ เหตุผล: จุดนี้ถูกคำนวณแล้วว่าพ้นรัศมีและคุ้มค่าต่อการเดินทางที่สุด (Optimized Cost: {safety_score:.2f})"
)
print("\n📋 คำอธิบายและคำแนะนำในการเอาตัวรอดที่เหมาะสมที่สุด:")
for advice in optimal_advices:
    print(advice)