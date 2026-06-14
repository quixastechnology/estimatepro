import torch
from torch.utils.data import DataLoader
from utils import FloorPlanDataset, ConvAutoencoder
import numpy as np
import json

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Model ----------------
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("models/conv_autoencoder.pth", map_location=device))
model.eval()

# ---------------- Load Training Dataset ----------------
train_dataset = FloorPlanDataset(r"datasets/training")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# ---------------- Compute Reconstruction Errors ----------------
errors = []
with torch.no_grad():
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        recon = model(imgs)
        error = torch.mean((imgs - recon)**2, dim=[1,2,3])
        errors.append(error.item())

errors = np.array(errors)

# ---------------- Dynamic Threshold (mean + 3*std) ----------------
threshold = errors.mean() + 3 * errors.std()
print(f"Dynamic threshold: {threshold:.6f}")

# ---------------- Save Threshold ----------------
with open("threshold.json", "w") as f:
    json.dump({"threshold": float(threshold)}, f)
print("Threshold saved to threshold.json")
