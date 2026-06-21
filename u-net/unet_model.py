"""
U-Net Semantic Segmentation Model
==================================
Implementation of U-Net architecture for semantic segmentation tasks.
Features symmetric encoder-decoder architecture with skip connections for feature preservation.

References:
    - Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Two consecutive 2D convolution blocks with batch normalization and ReLU activation.
    
    Architecture:
        Conv2d(in_channels -> out_channels, 3x3, padding=1)
        -> BatchNorm2d
        -> ReLU
        -> Conv2d(out_channels -> out_channels, 3x3, padding=1)
        -> BatchNorm2d
        -> ReLU
    
    This block preserves spatial dimensions and is used throughout the U-Net.
    """
    
    def __init__(self, in_channels, out_channels):
        """Initialize double convolution block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Apply double convolution to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, height, width)
        """
        return self.conv(x)

class UNet(nn.Module):
    """U-Net semantic segmentation model.
    
    Architecture:
        - Encoder: 2 downsampling blocks with MaxPool2d
        - Bridge: Central feature extraction layer
        - Decoder: 2 upsampling blocks with skip connections from encoder
        - Output: Segmentation map with n_classes channels
    
    Channel progression: 3 -> 64 -> 128 -> 256 (bridge) -> 128 -> 64 -> n_classes
    """
    
    def __init__(self, n_classes):
        """Initialize U-Net model.
        
        Args:
            n_classes (int): Number of output segmentation classes
        """
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = DoubleConv(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        """Forward pass through U-Net.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch, 3, height, width)
            
        Returns:
            torch.Tensor: Segmentation logits of shape (batch, n_classes, height, width)
        
        Process:
            1. Encoder: Apply DoubleConv + MaxPool (2x downsampling)
            2. Bridge: Feature extraction at lowest resolution
            3. Decoder: Upsample with ConvTranspose2d + concatenate with skip connections
            4. Output: Final 1x1 convolution to generate class predictions
        """
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bridge(p2)

        u1 = self.up1(b)
        c1 = self.conv1(torch.cat([u1, d2], dim=1))
        u2 = self.up2(c1)
        c2 = self.conv2(torch.cat([u2, d1], dim=1))

        return self.out(c2)
