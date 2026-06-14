import os

# Define the folder structure
folders = [
    "estimatepro-ai/ai_jobs/cost_calc",
    "estimatepro-ai/ai_jobs/summarizer",
    "estimatepro-ai/ai_jobs/ocr",
    "estimatepro-ai/ai_jobs/scheduler",
    "estimatepro-ai/ai_jobs/utils",
    "estimatepro-ai/datasets",
    "estimatepro-ai/models",
    "estimatepro-ai/outputs",
    "estimatepro-ai/tests",
    "estimatepro-ai/api"
]

# Create the folders and add __init__.py files
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    init_path = os.path.join(folder, "__init__.py")
    with open(init_path, "w") as f:
        f.write("# Init")

"Folder structure created."
