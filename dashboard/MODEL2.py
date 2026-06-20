import math
import random
from scipy.optimize import linprog


class DisasterSurvivalAI:

    def __init__(self, user_loc, disaster_loc, severity, safe_zones):
        self.user_loc = user_loc  # (x, y)
        self.disaster_loc = disaster_loc  # (x, y)
        self.severity = severity  # ระดับ 1-5
        self.safe_zones = safe_zones  # รายการพิกัด [(x1, y1), (x2, y2), ...]

    def _calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def optimize_route(self):
        """ใช้หลัก Optimization (Simulated Annealing แบบย่อ) คัดเลือกจุดปลอดภัยและเส้นทาง"""
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
        """ใช้ Linear Programming คัดเลือกน้ำหนักคำแนะนำที่เหมาะสมที่สุดภายใต้ข้อจำกัดเวลา"""
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