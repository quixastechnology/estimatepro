import os
import json

building_test_dir = r"D:\Company Projects\estimatepro-ai\datasets\testing\building_images_testing"

labels = {}

for filename in os.listdir(building_test_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        labels[filename] = 1   # 🚨 building = anomaly

output_path = r"D:\Company Projects\estimatepro-ai\datasets\building_test_labels.json"

with open(output_path, "w") as f:
    json.dump(labels, f, indent=4)

print(f"✅ building_test_labels.json created at: {output_path}")
