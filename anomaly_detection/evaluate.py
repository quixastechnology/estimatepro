import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Load CSVs ----------------
pred_df = pd.read_csv("anomaly_results.csv")
labels_df = pd.read_csv("datasets/testing/labels.csv")

# Normalize filenames and labels
pred_df['Filename'] = pred_df['Filename'].str.strip().str.lower()
pred_df['Status'] = pred_df['Status'].str.strip().str.capitalize()
labels_df['filename'] = labels_df['filename'].str.strip().str.lower()
labels_df['label'] = labels_df['label'].str.strip().str.capitalize()

# ---------------- Helper function ----------------
def compute_metrics(filenames, domain_name):
    # Merge predictions with labels based on filename
    merged = pd.merge(pred_df, labels_df, left_on='Filename', right_on='filename', how='inner')
    merged = merged[merged['Filename'].isin(filenames)]
    
    if merged.empty:
        print(f"No data found for {domain_name}")
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

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
    plt.title(f"Confusion Matrix ({domain_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ---------------- House Plan filenames ----------------
house_files = [f"{i}.jpg" for i in range(1, 21)]
compute_metrics(house_files, "House Plans")

# ---------------- Building Plan filenames ----------------
building_files = [f"{i}.png" for i in range(1, 6)]
compute_metrics(building_files, "Building Plans")
