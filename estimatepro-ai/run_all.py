import os
import json
import pandas as pd

# --- Load .env if available ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file.")
except ImportError:
    print("⚠️ python-dotenv not installed. Using system environment variables only.")

# --- Validate API key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
    raise ValueError(
        "❌ OpenAI API key is missing or invalid.\n"
        "Set it in a .env file as:\n"
        "OPENAI_API_KEY=sk-xxxx...\n"
        "or in your system environment variables."
    )

# --- Import OpenAI ---
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Import project modules ---
from ai_jobs.ocr import extract_rooms_from_plan
from ai_jobs.cost_estimator import estimate_costs_from_json
from ai_jobs.scheduler import generate_schedule_from_json
from ai_jobs.summarization import generate_summary_json  
from ai_jobs.Estimate_report import generate_estimate_report  

# --- Define paths ---
IMAGE_PATH = os.path.join("estimatepro-ai", "ai_jobs", "tests", "sample_plan.jpg")
LINEITEMS_PATH = os.path.join("estimatepro-ai", "ai_jobs", "tests", "lineitems.json")

# --- OCR Module ---
def run_ocr():
    print("🔍 Running OCR Module...")
    try:
        result = extract_rooms_from_plan(IMAGE_PATH)

        if "error" in result:
            raise ValueError(result["error"])

        rooms = result.get("rooms", [])
        rooms = [r for r in rooms if "room names" not in r.lower()]  # filter junk
        rooms_unique = list(dict.fromkeys(rooms))  # remove duplicates

        print("✅ OCR Output (Room Areas):")
        print(json.dumps(rooms_unique, indent=2))
        return rooms_unique
    except Exception as e:
        print(f"❌ Error in OCR: {e}")
        return []

# --- Estimator Module ---
def run_estimator():
    print("\n💰 Running Cost Estimator...")
    try:
        result = estimate_costs_from_json(LINEITEMS_PATH)
        print("✅ Cost Estimation Output:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"❌ Cost Estimation Failed: {e}")
        return {}

# --- Scheduler Module ---
def run_scheduler():
    print("\n📅 Running Scheduler...")
    try:
        result = generate_schedule_from_json(LINEITEMS_PATH)
        print("✅ Schedule Output:")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        print(f"❌ Scheduling Failed: {e}")
        return {}

# --- Summarizer Module ---
def run_summarizer():
    print("\n🧾 Running Line-Item Summarizer...")
    try:
        result = generate_summary_json(LINEITEMS_PATH)
        print("✅ Line-Item Summary Output:")
        print(json.dumps(result, indent=2))
        return result
    except json.JSONDecodeError as e:
        print(f"❌ Summarization returned invalid JSON: {e}")
        return {}
    except Exception as e:
        print(f"❌ Summarization Failed: {e}")
        return {}

# --- Report Generator Module ---
def run_report_generator(rooms, cost_estimate, schedule, summary):
    print("\n📄 Generating Final Estimate Report...")
    try:
        result = generate_estimate_report(
            rooms=rooms,
            cost_estimate=cost_estimate,
            schedule=schedule,
            summary=summary
        )
        print("✅ Final Estimate Report Generated:")
        print(result)
        return result
    except Exception as e:
        print(f"❌ Report Generation Failed: {e}")
        return None

# --- Main Entry ---
if __name__ == "__main__":
    rooms = run_ocr()
    cost_estimate = run_estimator()
    schedule = run_scheduler()
    summary = run_summarizer()
    run_report_generator(rooms, cost_estimate, schedule, summary)
