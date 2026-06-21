"""
U-Net Model Initialization with Pre-trained ResNet34 Encoder
=============================================================
Creates U-Net segmentation model for flood detection on multi-channel satellite data.
Uses ResNet34 pre-trained on ImageNet as encoder for better feature learning.

Model Configuration:
    - Encoder: ResNet34 (pre-trained on ImageNet)
    - Input channels: 6 (Pre-flood RGB + Post-flood RGB)
    - Output classes: 1 (Binary flood/no-flood segmentation)
    - Architecture: Fully convolutional U-Net with skip connections
"""

import segmentation_models_pytorch as smp

# Create U-Net model with pre-trained encoder
model = smp.Unet(
    encoder_name="resnet34",        # Backbone architecture (ResNet34)
    encoder_weights="imagenet",     # Load ImageNet pre-trained weights
    in_channels=6,                  # 6 channels: Pre-flood (RGB) + Post-flood (RGB)
    classes=1                       # Binary output: flood (1) vs no-flood (0)
)

# Model is ready for training with training data
# Typically used with:
#   - Loss: DiceLoss or BCEWithLogitsLoss
#   - Optimizer: Adam or SGD
#   - Input shape: (batch, 6, height, width)