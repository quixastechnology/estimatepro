import os
import sys
import json

# Load environment variables 
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

# Validate API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found. Please set it in your environment or .env file.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

def generate_summary_json(file_path: str) -> dict:
    """Reads a JSON file of line items and generates a cost summary using GPT-4o-mini."""
    try:
        with open(file_path, encoding="utf-8") as f:
            lineitems = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read input file: {e}")

    prompt = (
        "You are a construction cost summarizer.\n"
        "Given this JSON list of line items (with qty, rate, description),\n"
        "return ONLY valid JSON, with:\n"
        "1. total_cost: sum of qty * rate\n"
        "2. top_cost_drivers: list of top 3 items (description and cost)\n"
        "3. summary: 2–3 sentence summary\n\n"
        "OUTPUT JSON ONLY, no explanation or markdown.\n\n"
        f"{json.dumps(lineitems)}"
    )

    try:
        response = client.responses.create(
            model="gpt-4o-mini",  
            input=prompt
        )
        summary_text = response.output_text.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}")

    # Clean up potential markdown formatting
    if summary_text.startswith("```"):
        summary_text = summary_text.strip("`")
        summary_text = summary_text.replace("json", "", 1).strip()

    try:
        return json.loads(summary_text)
    except json.JSONDecodeError:
        raise ValueError(f"API returned invalid JSON:\n{summary_text}")

if __name__ == "__main__":
    path = r"estimatepro-ai/ai_jobs/tests/lineitems.json"
    result = generate_summary_json(path)
    print("Summary Output:", json.dumps(result, indent=2))
