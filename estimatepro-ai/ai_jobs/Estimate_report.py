import os
from openai import OpenAI
from dotenv import load_dotenv  

load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_estimate_report(costs, schedule, areas):
    # --- Prompt for Executive Report (Client Instruction) ---
    prompt_report = f"""
    You are an executive summary writer. Compile a 200-word summary highlighting:
    - Total project cost
    - Top 3 cost drivers
    - Timeline overview
    - Recommendations

    Use the following details:
    Costs: {costs}
    Schedule: {schedule}
    Areas: {areas}

    Write as a professional executive report, clear and concise.
    """

    # Get report
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_report}],
    )
    report = response.choices[0].message.content.strip()

    # Print the report
    print("\n📄 Executive Estimate Report:\n")
    print(report)

    # --- Prompt for Evaluation ---
    prompt_eval = f"""
    You are an evaluator. Read the following Executive Estimate Report:

    {report}

    Now rate it on:
    - Usefulness (1–5)
    - Correctness (1–5)
    - Readability (1–5)

    Give results in this JSON format only:
    {{
      "Usefulness": x.x,
      "Correctness": x.x,
      "Readability": x.x,
      "Interpretation": "..."
    }}
    """

    eval_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_eval}],
    )
    evaluation = eval_response.choices[0].message.content.strip()

    # Print Evaluation Block
    print("\n📊 Executive Report Quality – Human Evaluation\n")
    print(evaluation)


# --- Example Run ---
if __name__ == "__main__":
    costs = {"Concrete": 3000, "Steel": 2000, "Paint": 500}
    schedule = {"Start": "2025-08-01", "End": "2025-08-18", "Duration_days": 17}
    areas = {"Living Room": "200 sqft", "Bedroom": "150 sqft"}

    generate_estimate_report(costs, schedule, areas)
