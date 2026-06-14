from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import json

def evaluate_model(y_true, y_pred, save_path="estimatepro-ai/metrics.json"):
    """
    Evaluate model predictions and save metrics.

    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        save_path (str): Path to save JSON metrics
    """

    # --- Calculate Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # --- Print Report ---
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("Summary Metrics:")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")

    # --- Save Metrics to JSON ---
    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "detailed_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    }

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📁 Metrics saved to {save_path}")
    return metrics


# ---------------- Example Run ----------------
if __name__ == "__main__":
    
    y_true = ["bedroom", "bathroom", "kitchen", "living room", "bedroom"]
    y_pred = ["bedroom", "bathroom", "living room", "living room", "bedroom"]

    evaluate_model(y_true, y_pred)
