import os
from ultralytics import YOLO
import cv2
import json

# --- CONFIGURATION (ปรับเปลี่ยนพาร์ทตามการใช้งานจริง) ---
# 1. พาร์ทโมเดลที่คุณเทรนเสร็จแล้วจากโค้ดบน (เช่น 'best.pt')
# สมมติว่าย้ายไฟล์ best.pt มาไว้โฟลเดอร์เดียวกับโค้ด
BEST_MODEL_PATH = "runs/detect/train/weights/best.pt" # <--- แก้ไขพาร์ทให้ถูกต้อง

# 2. พาร์ทรูปภาพถ่ายที่ User ส่งเข้ามาทดสอบ
# สามารถแทนที่ด้วยพาร์ทรูปภาพถ่ายหน้างานจริง
TEST_IMAGE_PATH = "flood_user_sample.jpg" 

# 3. การตั้งค่าการตรวจจับ
CONFIDENCE_THRESHOLD = 0.5 # ระดับความมั่นใจขั้นต่ำที่จะยอมรับผลการตรวจจับ (50%)
IMAGE_SIZE = 640           # ขนาดรูปภาพที่จะนำเข้าโมเดล (มาตรฐาน)

def run_inference():
    print(f"--- 🚀 Running Inference ---")
    print(f"Loading Best Model: {BEST_MODEL_PATH}")
    
    # --- 🛠️ ขั้นตอนที่ 1: โหลดโมเดลตัวจริง ---
    model = YOLO(BEST_MODEL_PATH)
    
    # --- 🛠️ ขั้นตอนที่ 2: สั่งตรวจจับ (Prediction Loop) ---
    print(f"Testing Image: {TEST_IMAGE_PATH}")
    
    # ประมวลผลภาพเดี่ยว (Single Inference)
    results = model(TEST_IMAGE_PATH, conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE)
    
    # --- 🛠️ ขั้นตอนที่ 3: ดึงผลลัพธ์และจัดรูปแบบ ---
    flood_objects = []
    
    # วนลูปดึงข้อมูลจากผลลัพธ์
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # ดึงข้อมูลขอบเขตวัตถุ (Bounding Box coords: xyxy)
            coords = box.xyxy[0].tolist()
            # ดึงค่าความมั่นใจ (Confidence)
            confidence = float(box.conf[0])
            # ดึงชื่อคลาส (Class Name - สมมติ NC=1 คลาสคือ 'flood')
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # บันทึกผลลัพธ์ลงในลิสต์แบบ Structured Data
            flood_data = {
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [round(c, 2) for c in coords] # ปัดเศษทศนิยม
            }
            flood_objects.append(flood_data)
            
            print(f"Detected: {class_name} with confidence {confidence:.2f}")

    # --- 🛠️ ขั้นตอนที่ 4: แสดงผลลัพธ์ (Output) ---
    # แสดงผลเป็น JSON เผื่อนำไปเชื่อมต่อกับ Pipeline อื่น
    print("\n--- ✅ Inference Complete (JSON Output) ---")
    print(json.dumps(flood_objects, indent=2))
    
    # --- ✅ วาดกรอบสี่เหลี่ยมทับลงบนภาพและบันทึกผล ---
    results[0].save(filename="inference_result.jpg")
    print("\nบันทึกภาพผลลัพธ์พร้อมกรอบสี่เหลี่ยมแล้วในชื่อ: inference_result.jpg")

if __name__ == "__main__":
    run_inference()