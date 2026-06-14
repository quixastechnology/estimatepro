import csv
from torch.utils.data import DataLoader
from utils import TestDataset, ConvAutoencoder
import torch

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Model ----------------
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("models/conv_autoencoder.pth", map_location=device))
model.eval()

# ---------------- Test Folders ----------------
test_folders = {
    "House Plans": r"datasets/testing/house_images_testing",
    "Building Plans": r"datasets/testing/building_images_testing"
}

# ---------------- Threshold ----------------
threshold = 0.01  # replace with your dynamic threshold if needed

# ---------------- Save Results to CSV ----------------
results_file = "anomaly_results.csv"

with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Category", "Filename", "Status", "Error"])

    for category, folder in test_folders.items():
        test_dataset = TestDataset(folder)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        for imgs, filenames in test_loader:
            imgs = imgs.to(device)
            with torch.no_grad():
                recon = model(imgs)
            error = torch.mean((imgs - recon) ** 2).item()
            status = "Anomaly" if error > threshold else "Normal"
            writer.writerow([category, filenames[0], status, f"{error:.6f}"])
