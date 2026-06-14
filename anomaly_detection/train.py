import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import FloorPlanDataset  # make sure this returns tensors in [0,1]
import os

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Conv Autoencoder ----------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1), nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,stride=2,padding=1,output_padding=1), nn.Sigmoid()
        )

    def forward(self,x):
        return self.decoder(self.encoder(x))

# ---------------- Dataset & DataLoader ----------------
train_dataset = FloorPlanDataset(r"datasets\training")
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ---------------- Model, Loss, Optimizer ----------------
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------- Training Info ----------------
epochs = 10
total_batches = len(train_loader)
total_updates = total_batches * epochs

print(f"📊 Dataset size = {len(train_dataset)} images")
print(f"Batch size = {batch_size} → {total_batches} batches per epoch")
print(f"Epochs = {epochs} (model sees all {len(train_dataset)} images {epochs} times)")
print(f"That means:\n{total_batches} updates per epoch × {epochs} epochs = {total_updates} total weight updates.\n")

# ---------------- Training Loop ----------------
for epoch in range(epochs):
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_dataset):.6f}")

# ---------------- Save Model ----------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/conv_autoencoder.pth")
print("✅ Model saved to models/conv_autoencoder.pth")
