import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch.nn as nn

# ---------------- Training Dataset ----------------
class FloorPlanDataset(Dataset):
    """
    Dataset class for training floor plan images.
    Assumes all images are in one folder (unlabeled) and converts them to tensors.
    """
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.files[idx])
        img = Image.open(img_path).convert('L')  # convert to grayscale
        img = img.resize((128,128))               # resize to match model input
        img = np.array(img)/255.0                 # normalize to [0,1]
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        if self.transform:
            img = self.transform(img)
        return img, self.files[idx]  # return image tensor and filename

# ---------------- Testing Dataset ----------------
class TestDataset(Dataset):
    """
    Dataset class for testing floor plan images.
    Returns image tensor and filename.
    """
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.files[idx])
        img = Image.open(img_path).convert('L')  # convert to grayscale
        img = img.resize((128,128))               # resize to model input
        img = np.array(img)/255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        return img, self.files[idx]

# ---------------- Convolutional Autoencoder ----------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # (128 -> 64)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # (64 -> 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # (32 -> 16)
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # (16 -> 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # (32 -> 64)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # (64 -> 128)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
