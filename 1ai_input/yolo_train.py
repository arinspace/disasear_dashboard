import os
from ultralytics import YOLO

# --- CONFIGURATION (ปรับเปลี่ยนพาร์ทตามชุดข้อมูลของคุณ) ---
# 1. พาร์ทโมเดลตั้งต้นที่คุณเตรียมไว้ (เช็กให้ตรงกับชื่อไฟล์จริง)
PRETRAINED_MODEL_PATH = "yolov8n(geo).pt" 

# 2. พาร์ทไฟล์ data.yaml ที่กำหนดชุดข้อมูล (เช็กให้ตรงกับโฟลเดอร์ data)
# ไฟล์ data.yaml ควรมีเนื้อหาดังนี้ (ตัวอย่าง):
# train: /path/to/my_data_folder/images/train
# val: /path/to/my_data_folder/images/val
# nc: 1  (จำนวนคลาส)
# names: ['flood'] (ชื่อคลาส)
DATA_YAML_PATH = "path/to/data.yaml" # <--- แก้ไขพาร์ทให้ถูกต้อง

# 3. การตั้งค่าการเทรน (ปรับได้ตามความแรงของ GPU)
EPOCHS = 100       # จำนวนรอบการเทรน (แนะนำ 50-100 รอบสำหรับ Fine-tune)
BATCH_SIZE = 16    # จำนวนภาพต่อรอบ (ถ้า GPU แรมเต็มให้ลดลงเหลือ 8 หรือ 4)
IMAGE_SIZE = 640   # ขนาดรูปภาพที่จะนำเข้าโมเดล (มาตรฐาน YOLOv8 คือ 640)
DEVICE = 0         # อุปกรณ์ที่จะใช้เทรน (ถ้ามี GPU เลือก 0, ถ้าใช้ CPU พิมพ์ 'cpu')

def train_custom_model():
    print(f"--- 🚀 Starting Custom Training ---")
    print(f"Loading Pretrained Model: {PRETRAINED_MODEL_PATH}")
    
    # --- 🛠️ ขั้นตอนที่ 1: โหลดโมเดลตั้งต้น ---
    # โหลดโมเดลที่มีโครงสร้างพร้อมเทรนต่อ
    model = YOLO(PRETRAINED_MODEL_PATH)
    
    # --- 🛠️ ขั้นตอนที่ 2: เริ่มสั่งเทรน (Training Loop) ---
    print(f"Training Data Config: {DATA_YAML_PATH}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Device: {DEVICE}")
    
    # บันทึก log การเทรนไว้ในโฟลเดอร์ runs/detect/train
    results = model.train(
        data=DATA_YAML_PATH, 
        epochs=EPOCHS, 
        batch=BATCH_SIZE, 
        imgsz=IMAGE_SIZE, 
        device=DEVICE,
        workers=8,     # จำนวน CPU threads สำหรับโหลดข้อมูล (ถ้าใช้ CPU เทรนให้ลดลง)
        patience=20    # ถ้าค่า Loss ไม่ดีขึ้นใน 20 รอบ ให้หยุดเทรนเพื่อป้องกัน Overfitting
    )
    
    print(f"--- ✅ Custom Training Complete ---")
    # ไฟล์โมเดลตัวเก่งที่สุดจะถูกบันทึกไว้ในพาร์ทนี้
    print(f"Best model weights saved at: {os.path.join(results.save_dir, 'weights/best.pt')}")

if __name__ == "__main__":
    train_custom_model()