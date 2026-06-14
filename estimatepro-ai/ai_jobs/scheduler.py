import json
from pathlib import Path
from openai import OpenAI
import re
import datetime
import os

# Load API key from environment
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file.")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables only.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API key not found. Please set OPENAI_API_KEY in .env or system env.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def clean_response(content: str) -> str:
    """Remove ```json or ``` markdown formatting from GPT response"""
    if content.startswith("```"):
        content = re.sub(r"```(?:json)?\n?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    return content


def calculate_mae(pred_schedule, true_durations):
    """
    Calculate Mean Absolute Error (MAE) between predicted and true durations.
    """
    errors = []
    for task in pred_schedule:
        task_name = task["task"]
        pred_days = task["duration_days"]
        true_days = true_durations.get(task_name)

        if true_days is not None:
            errors.append(abs(pred_days - true_days))

    if not errors:
        return None

    mae = sum(errors) / len(errors)
    return mae


def generate_schedule_from_json(data, project_name=None):
    """
    Generate project schedule JSON using GPT based on client-style prompt.
    Supports both file path (string) and dict input.
    """
    try:
        # Load file if path given
        if isinstance(data, str):
            json_path = Path(data).resolve()
            with open(json_path, 'r') as f:
                data = json.load(f)
        elif not isinstance(data, dict):
            return "❌ Invalid data format. Must be JSON file path (str) or dict."

        current_year = datetime.datetime.now().year
        today = datetime.date.today().strftime("%Y-%m-%d")

        # Client-style prompt
        prompt = f"""
Timeline Estimation
You are a project scheduler.
Based on {json.dumps(data, indent=2)},  
generate a Gantt schedule JSON mapping tasks to dates.  

Each entry must have: "task", "start_date", "end_date", "duration_days".  
Use year {current_year}, starting from {today}.  
Return only valid JSON, no explanation.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a construction project manager and planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        raw_content = response.choices[0].message.content.strip()
        print("🔍 GPT Raw Response:\n", raw_content)

        cleaned_content = clean_response(raw_content)

        if cleaned_content.startswith("["):
            schedule = json.loads(cleaned_content)
            return schedule
        else:
            return f"❌ GPT did not return valid JSON.\nCleaned content:\n{cleaned_content}"

    except Exception as e:
        return f"❌ Scheduling Failed: {str(e)}"


# Example usage
if __name__ == "__main__":
    durations_path = "estimatepro-ai/ai_jobs/durations.json"

    with open(durations_path, "r") as f:
        true_data = json.load(f)

    result = generate_schedule_from_json(durations_path, project_name="MyProject")

    if isinstance(result, list):
        print("✅ Project Schedule (JSON):")
        print(json.dumps(result, indent=2))

        # Calculate MAE
        true_durations = true_data.get("durations", {})
        mae = calculate_mae(result, true_durations)

        if mae is not None:
            print(f"\n📊 Metric: MAE (Mean Absolute Error) = {mae:.2f} days")
            if mae <= 3:
                print("✅ Interpretation: Small projects OK. Larger multi-trade projects may need schedule tuning.")
            else:
                print("⚠️ Interpretation: High deviation. Schedule tuning strongly recommended.")
        else:
            print("⚠️ Could not calculate MAE (no matching tasks).")

    else:
        print(result)
