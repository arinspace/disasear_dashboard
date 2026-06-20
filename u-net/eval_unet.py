import torch
from torch.utils.data import DataLoader
from unet_model import UNet
from dataset_loader import BDDDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_dataset = BDDDataset("datasets/bdd100k/images/val", "datasets/bdd100k/labels/val")
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = UNet(n_classes=19).to(device)
model.load_state_dict(torch.load("unet_bdd100k.pth"))
model.eval()

with torch.no_grad():
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        # TODO: คำนวณ mIoU หรือ accuracy
        print("Batch evaluated")
