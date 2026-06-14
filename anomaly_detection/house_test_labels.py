import os
import json

# Absolute path (works best on Windows)
house_test_dir = r"D:\Company Projects\estimatepro-ai\datasets\testing\house_images_testing"

labels = {}

for filename in os.listdir(house_test_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        labels[filename] = 0   # ✅ house = normal

output_path = r"D:\Company Projects\estimatepro-ai\datasets\house_test_labels.json"

with open(output_path, "w") as f:
    json.dump(labels, f, indent=4)

print(f"✅ house_test_labels.json created at: {output_path}")
