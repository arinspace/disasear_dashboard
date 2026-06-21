"""
Flood Detection Model Training Script
======================================
Trains U-Net segmentation model on flood detection dataset.
Combines pre-flood and post-flood satellite imagery for binary flood/no-flood classification.

Configuration:
    - Model: U-Net with ResNet34 encoder
    - Input: 6-channel satellite data (pre + post)
    - Loss: Dice Loss (good for binary segmentation with class imbalance)
    - Optimizer: Adam
    - Training epochs: 10 (initial, can be increased)
    - Batch size: 4
    - Output model: flood_model.pth
"""

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import rasterio
import os
import numpy as np

class FloodDataset(Dataset):
    """Custom Dataset for flood detection with pre/post event satellite imagery.
    
    Combines pre-event and post-event satellite images into 6-channel tensors.
    Loads corresponding ground truth flood masks for supervised learning.
    
    Directory Structure:
        - pre_dir: Pre-event RGB satellite images
        - post_dir: Post-event RGB satellite images  
        - mask_dir: Binary flood masks (0=no flood, 1=flood)
    """
    
    def __init__(self, pre_dir, post_dir, mask_dir):
        """Initialize flood detection dataset.
        
        Args:
            pre_dir (str): Directory containing pre-event satellite images
            post_dir (str): Directory containing post-event satellite images
            mask_dir (str): Directory containing flood ground truth masks
        """
        self.images = sorted(os.listdir(pre_dir))
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.mask_dir = mask_dir

    def __len__(self):
        """Return total number of samples."""
        return len(self.images)

    def __getitem__(self, idx):
        """Load pre-event, post-event images and corresponding mask.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (combined_image, mask) where
                - combined_image: torch.Tensor of shape (6, H, W) [0-1] normalized
                - mask: torch.Tensor of shape (1, H, W) with flood labels
        """
        img_name = self.images[idx]
        with rasterio.open(os.path.join(self.pre_dir, img_name)) as src: pre = src.read().astype(np.float32) / 255.0
        with rasterio.open(os.path.join(self.post_dir, img_name)) as src: post = src.read().astype(np.float32) / 255.0
        with rasterio.open(os.path.join(self.mask_dir, img_name)) as src: mask = src.read().astype(np.float32)
        
        # Concatenate pre and post images into 6-channel tensor
        combined = np.concatenate([pre, post], axis=0)
        return torch.tensor(combined), torch.tensor(mask).unsqueeze(0)

# 2. Training configuration
dataset = FloodDataset('train/images_pre', 'train/images_post', 'train/masks')
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 3. Create U-Net model
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=6, classes=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = smp.losses.DiceLoss(mode='binary')

# 4. Training loop
print("Starting model training...")
model.train()
for epoch in range(10):  # 10 epochs initially (can be increased for better performance)
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete! Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "flood_model.pth")
print("✅ Training complete! Model saved to flood_model.pth")