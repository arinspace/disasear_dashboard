import segmentation_models_pytorch as smp

# คำสั่งนี้คือการสร้างโมเดล U-Net ขึ้นมาใหม่
model = smp.Unet(
    encoder_name="resnet34",        # เลือกโครงสร้างดวงตา AI (Backbone)
    encoder_weights="imagenet",     # ให้มันโหลดน้ำหนักที่ฉลาดมาแล้วจาก ImageNet
    in_channels=6,                  # เราใส่ 6 เพราะเป็นภาพ Pre + Post รวมกัน
    classes=1                       # ผลลัพธ์สุดท้ายคือ "น้ำท่วม" (1 คลาส)
)