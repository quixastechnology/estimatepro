import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import TestDataset, ConvAutoencoder
import torch

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Model ----------------
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("models/conv_autoencoder.pth", map_location=device))
model.eval()

# ---------------- Test Folder ----------------
folder = r"datasets/testing/house_images_testing"
test_dataset = TestDataset(folder)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------------- Visualization Function ----------------
def visualize(img, recon, filename):
    img = img.squeeze().cpu().numpy()
    recon = recon.squeeze().cpu().numpy()

    plt.figure(figsize=(6,3))
    plt.suptitle(f"{filename}")
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Reconstruction")
    plt.imshow(recon, cmap='gray')
    plt.axis('off')

    plt.show()

# ---------------- Example Usage ----------------
imgs, filenames = next(iter(test_loader))
with torch.no_grad():
    recon = model(imgs.to(device))
visualize(imgs[0], recon[0], filenames[0])
