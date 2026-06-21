"""
U-Net Training Script for BDD100K Segmentation
===============================================
Trains U-Net model on BDD100K dataset with early stopping and learning rate scheduling.

Configuration:
    - Epochs: 40
    - Batch Size: 4
    - Learning Rate: 0.001 with step decay
    - Device: GPU (if available) or CPU
    - Early Stopping: Patience = 5 epochs
    
Model Checkpoints:
    - Best model saved as 'unet_bdd100k_best.pth' based on validation loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_model import UNet
from dataset_loader import BDDDataset
from torch.optim.lr_scheduler import StepLR

# ✅ Config
EPOCHS = 40
BATCH_SIZE = 4
LEARNING_RATE = 0.001
PATIENCE = 5   # early stopping patience

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Dataset
train_dataset = BDDDataset(
    root_dir="D:/geoai_train/geoai_train/datasets/bdd100k",
    list_file="D:/geoai_train/geoai_train/datasets/bdd100k/train.txt",
    split="train"
)

val_dataset = BDDDataset(
    root_dir="D:/geoai_train/geoai_train/datasets/bdd100k",
    list_file="D:/geoai_train/geoai_train/datasets/bdd100k/val.txt",
    split="val"
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ✅ Model
model = UNet(n_classes=19).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# ✅ Early Stopping variables
best_val_loss = float("inf")
patience_counter = 0

# ✅ Training Loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Scheduler step
    scheduler.step()

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "unet_bdd100k_best.pth")
        print("✅ Model improved, saved checkpoint.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered.")
            break
