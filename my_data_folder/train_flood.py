import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import rasterio
import os
import numpy as np

# 1. นิยาม Dataset: ตัวอ่านภาพจากโฟลเดอร์ train ของคุณ
class FloodDataset(Dataset):
    def __init__(self, pre_dir, post_dir, mask_dir):
        self.images = sorted(os.listdir(pre_dir))
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.mask_dir = mask_dir

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        with rasterio.open(os.path.join(self.pre_dir, img_name)) as src: pre = src.read().astype(np.float32) / 255.0
        with rasterio.open(os.path.join(self.post_dir, img_name)) as src: post = src.read().astype(np.float32) / 255.0
        with rasterio.open(os.path.join(self.mask_dir, img_name)) as src: mask = src.read().astype(np.float32)
        
        # รวม Pre และ Post เป็น 6 Channel
        combined = np.concatenate([pre, post], axis=0)
        return torch.tensor(combined), torch.tensor(mask).unsqueeze(0)

# 2. ตั้งค่าการเทรน
dataset = FloodDataset('train/images_pre', 'train/images_post', 'train/masks')
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 3. สร้างโมเดล U-Net
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=6, classes=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = smp.losses.DiceLoss(mode='binary')

# 4. Loop การเรียนรู้
print("เริ่มกระบวนการสอน AI...")
model.train()
for epoch in range(10): # เริ่มต้นที่ 10 รอบ
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} จบแล้ว! ค่าความผิดพลาด (Loss): {loss.item():.4f}")

torch.save(model.state_dict(), "flood_model.pth")
print("✅ เทรนเสร็จสิ้น! บันทึกโมเดลไว้ใน flood_model.pth แล้ว")