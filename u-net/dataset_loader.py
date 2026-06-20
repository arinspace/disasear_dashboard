import os
import torch
from torch.utils.data import Dataset
import cv2

class BDDDataset(Dataset):
    def __init__(self, root_dir, list_file, split="train", transform=None):
        """
        root_dir: โฟลเดอร์หลักของ dataset เช่น D:/geoai_train/geoai_train/datasets/bdd100k
        list_file: path ไปยัง train.txt หรือ val.txt ที่มี list ของ ID
        split: "train" หรือ "val" เพื่อเลือกโฟลเดอร์ย่อย
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # อ่านไฟล์ list (แต่ละบรรทัดคือ ID)
        with open(list_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
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
