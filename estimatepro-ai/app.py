import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from flasgger import Swagger


# Import AI job modules
from ai_jobs.ocr import extract_rooms_from_plan
from ai_jobs.scheduler import generate_schedule_from_json
from ai_jobs.summarization import generate_summary_json
from ai_jobs.Estimate_report import generate_estimate_report

app = Flask(__name__)
swagger = Swagger(app)

  

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------
# Helpers
# -----------------------------
def save_uploaded_file(file):
    """Save uploaded file and return path."""
    if not file or file.filename == "":
        return None, "No file selected"
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    return filepath, None


# -----------------------------
# Core Functions
# -----------------------------
def calculate_cost(data: dict) -> dict:
    """Calculate cost based on rooms with area * rate per sqft (+10% contingency)."""
    rooms = data.get("rooms", [])
    rate_per_sqft = data.get("rate_per_sqft", 15) 
    breakdown, grand_total = [], 0

    for room in rooms:
        name = room.get("name", "Unknown")
        area = float(room.get("area", 0))
        cost = round(area * rate_per_sqft * 1.1, 2) 
        breakdown.append({   
            "room": name,
            "area": area,
            "rate_per_sqft": rate_per_sqft,
            "cost": cost
        })
        grand_total += cost

    return {"breakdown": breakdown, "grand_total": round(grand_total, 2)}


def generate_schedule(data: dict) -> list:
    """Generate sequential schedule for rooms."""
    rooms = data.get("rooms", [])
    durations = data.get("durations", {})  
    default_days = data.get("default_days", 5)

    today = datetime.today()
    schedule, current_date = [], today

    for room in rooms:
        name = room.get("name", "Unknown")
        days = int(durations.get(name, default_days))
        end_date = current_date + timedelta(days=days)
        schedule.append({
            "task": name,
            "start_date": current_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "duration_days": days
        })
        current_date = end_date

    return schedule


def generate_summary(data: dict) -> str:
    """Generate a project summary."""
    rooms = data.get("rooms", [])
    cost = data.get("cost_estimate", {}).get("grand_total", 0)
    schedule = data.get("schedule", [])
    return (
        f"Project includes {len(rooms)} rooms. "
        f"Total estimated cost: ${cost}. "
        f"Timeline includes {len(schedule)} tasks."
    )


def run_all_pipeline(image_path: str):
    """Run OCR → Cost → Schedule → Summary → Report."""
    rooms_data = extract_rooms_from_plan(image_path)

    cost_data = calculate_cost({
        "rooms": rooms_data,
        "rate_per_sqft": 15
    })

    schedule_data = generate_schedule({
        "rooms": rooms_data,
        "default_days": 5
    })

    summary = generate_summary({
        "rooms": rooms_data,
        "cost_estimate": cost_data,
        "schedule": schedule_data
    })

    report = generate_estimate_report(rooms_data, cost_data, schedule_data, summary)

    return {
        "rooms": rooms_data,
        "cost_estimate": cost_data,
        "schedule": schedule_data,
        "summary": summary,
        "report": report
    }


# -----------------------------
# Unified Module Endpoint
# -----------------------------
@app.route("/api/module", methods=["POST"])
def run_module():
    """
    Run different AI modules (OCR, Cost, Schedule, Summarize, Report, Run-All)
    ---
    tags:
      - AI Modules
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - module
            - data
          properties:
            module:
              type: string
              enum: ["ocr", "cost", "schedule", "summarize", "report", "runall"]
              example: "cost"
            data:
              type: object
              example:
                rooms:
                  - name: "Living Room"
                    area: 200
                  - name: "Bedroom"
                    area: 150
                rate_per_sqft: 15
    responses:
      200:
        description: Returns result of selected module
    """
    try:
        data = request.get_json(force=True)
        module = data.get("module")
        module_data = data.get("data")

        if not module or module_data is None:
            return jsonify({"error": "Missing 'module' or 'data'"}), 400

        # --- OCR ---
        if module == "ocr":
            image_path = module_data.get("image_path")
            if not isinstance(image_path, str):
                return jsonify({"error": "OCR expects image path as string"}), 400
            return jsonify({"rooms": extract_rooms_from_plan(image_path)})

        # --- Cost Estimator ---
        elif module == "cost":
            return jsonify({"cost_estimate": calculate_cost(module_data)})

        # --- Scheduler ---
        elif module == "schedule":
            return jsonify({"schedule": generate_schedule(module_data)})

        # --- Summarizer ---
        elif module == "summarize":
            return jsonify({"summary": generate_summary(module_data)})

        # --- Report ---
        elif module == "report":
            rooms = module_data.get("rooms", [])
            cost = module_data.get("cost_estimate", {})
            schedule = module_data.get("schedule", [])
            summary = module_data.get("summary", "")
            report = generate_estimate_report(rooms, cost, schedule, summary)
            return jsonify({"report": report})

        # --- Run-All ---
        elif module == "runall":
            image_path = module_data.get("image_path")
            if not isinstance(image_path, str):
                return jsonify({"error": "runall expects image path as string"}), 400
            return jsonify(run_all_pipeline(image_path))

        else:
            return jsonify({"error": f"Unknown module '{module}'"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Home route for quick check
# -----------------------------
@app.route("/")
def home():
    return "Hello, Flask API with Swagger + ngrok is running!"


# -----------------------------
# Run Flask + ngrok
# -----------------------------
if __name__ == "__main__":
    app.run()
