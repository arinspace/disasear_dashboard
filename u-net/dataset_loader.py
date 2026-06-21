"""
BDD100K Dataset Loader
======================
Dataset class for loading BDD100K segmentation dataset for training U-Net models.
Handles image and mask loading with PyTorch DataLoader compatibility.
"""

import os
import torch
from torch.utils.data import Dataset
import cv2

class BDDDataset(Dataset):
    """PyTorch Dataset for BDD100K semantic segmentation.
    
    Attributes:
        root_dir (str): Root directory containing 'images' and 'labels' subdirectories
        list_file (str): Path to text file with list of sample IDs (one per line)
        split (str): Dataset split - 'train' or 'val'
        transform (callable, optional): Optional transforms to apply to image/mask pairs
    """
    
    def __init__(self, root_dir, list_file, split="train", transform=None):
        """
        Initialize BDD100K dataset loader.
        
        Args:
            root_dir (str): Root dataset directory (e.g. D:/geoai_train/geoai_train/datasets/bdd100k)
            list_file (str): Path to train.txt or val.txt file containing sample IDs
            split (str): Dataset split - 'train' or 'val' to select subdirectory
            transform (callable, optional): Optional image/mask transformation function
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # อ่านไฟล์ list (แต่ละบรรทัดคือ ID)
        with open(list_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.ids)

    def __getitem__(self, idx):
        """Load image and mask for given index.
        
        Args:
            idx (int): Index of sample to load
            
        Returns:
            tuple: (image tensor [0-1], mask tensor) where
                - image: torch.Tensor of shape (3, H, W) normalized to [0, 1]
                - mask: torch.Tensor of shape (H, W) with class indices
                
        Raises:
            FileNotFoundError: If image or mask file not found at expected path
        """
        sample_id = self.ids[idx]

        # สร้าง path ของ image และ mask จาก ID
        img_path = os.path.join(self.root_dir, "images", self.split, sample_id + ".jpg")
        mask_path = os.path.join(self.root_dir, "labels", self.split, sample_id + ".png")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
