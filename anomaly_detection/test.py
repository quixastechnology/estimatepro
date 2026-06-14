import torch
from torch.utils.data import DataLoader
from utils import TestDataset, ConvAutoencoder  # reuse model class
import os
import numpy as np

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Model ----------------
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("models/conv_autoencoder.pth", map_location=device))
model.eval()

# ---------------- Testing Folders ----------------
test_folders = {
    "House Plans": r"datasets\testing\house_images_testing",
    "Building Plans": r"datasets\testing\building_images_testing"
}

# ---------------- Threshold ----------------
threshold = 0.01  # tune based on training loss

# ---------------- Run Predictions ----------------
for category, folder in test_folders.items():
    print(f"\n📂 Testing {category} ({folder})")
    
    test_dataset = TestDataset(folder)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    for imgs, filenames in test_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            recon = model(imgs)
        error = torch.mean((imgs - recon) ** 2).item()
        status = "Anomaly" if error > threshold else "Normal"
        print(f"{filenames[0]} → {status} (error={error:.6f})")
