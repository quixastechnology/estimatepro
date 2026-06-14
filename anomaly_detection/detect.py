import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import TestDataset, FloorPlanDataset, ConvAutoencoder

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Model ----------------
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("models/conv_autoencoder.pth", map_location=device))
model.eval()

# ---------------- Compute threshold from training House plans ----------------
train_dataset = FloorPlanDataset(r"datasets/training")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
train_errors = []

with torch.no_grad():
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        recon = model(imgs)
        error = torch.mean((imgs - recon)**2, dim=[1,2,3])
        train_errors.append(error.item())

threshold = np.percentile(train_errors, 99)
print(f"Dynamic threshold (99th percentile): {threshold:.6f}")

# ---------------- Test folders ----------------
test_folders = {
    "House Plans": r"datasets/testing/house_images_testing",
    "Building Plans": r"datasets/testing/building_images_testing"
}

# ---------------- Load labels ----------------
labels_df = pd.read_csv("datasets/testing/labels.csv")
labels_df['filename'] = labels_df['filename'].str.strip()  # remove whitespace

# ---------------- Collect predictions ----------------
predictions = []

for domain_name, folder_path in test_folders.items():
    test_dataset = TestDataset(folder_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for imgs, filenames in test_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            recon = model(imgs)
        error = torch.mean((imgs - recon)**2).item()
        status = "Anomaly" if error > threshold else "Normal"
        predictions.append({
            "Domain": domain_name,
            "Filename": filenames[0],
            "Status": status,
            "Error": error
        })

pred_df = pd.DataFrame(predictions)
pred_df.to_csv("anomaly_results.csv", index=False)

# ---------------- Compute metrics for any domain ----------------
def compute_metrics(domain_name):
    domain_pred = pred_df[pred_df['Domain'] == domain_name].copy()
    domain_labels = labels_df[labels_df['filename'].isin(domain_pred['Filename'])].copy()

    merged = pd.merge(domain_pred, domain_labels, left_on='Filename', right_on='filename', how='inner')
    if merged.empty:
        print(f"No data for {domain_name}. Check labels.csv!")
        return

    y_true = merged['label'].map({"Normal":0, "Anomaly":1})
    y_pred = merged['Status'].map({"Normal":0, "Anomaly":1})

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nMetrics for {domain_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal","Anomaly"], yticklabels=["Normal","Anomaly"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({domain_name})")
    plt.show()

# ---------------- Evaluate House and Building ----------------
compute_metrics("House Plans")
compute_metrics("Building Plans")
