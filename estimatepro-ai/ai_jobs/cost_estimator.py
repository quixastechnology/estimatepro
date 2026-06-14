import json
from pathlib import Path
from openai import OpenAI
import os
import re
import math
from dotenv import load_dotenv  

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_response(content: str) -> str:
    """Remove markdown formatting if GPT wraps response in ```json blocks"""
    if content.startswith("```"):
        content = re.sub(r"```(?:json)?\n?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    return content

def estimate_costs_from_json(json_file_path: str, project_name: str = None):
    """Send JSON to GPT and get itemized cost estimation"""
    try:
        json_path = Path(json_file_path).resolve()
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prompt = f"""
You are a construction cost model. Using items and rates: {json.dumps(data)},  
calculate cost = qty × rate + 10% contingency.  
Return JSON with each line-item cost and grand total.  
Return only valid JSON.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional construction cost calculator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        raw_content = response.choices[0].message.content.strip()
        cleaned_content = clean_response(raw_content)

        if cleaned_content.startswith("{"):
            return json.loads(cleaned_content)
        else:
            return f"❌ Invalid JSON returned:\n{cleaned_content}"

    except Exception as e:
        print(f"❌ Cost Estimation Failed: {e}")
        return None

def calculate_accuracy(predictions, ground_truths):
    """Compute MAPE and RMSE"""
    n = len(predictions)
    if n == 0:
        return None, None

    abs_percentage_errors = []
    squared_errors = []

    for pred, true in zip(predictions, ground_truths):
        abs_error = abs(pred - true)
        abs_percentage_errors.append(abs_error / max(1, true) * 100)  
        squared_errors.append((pred - true) ** 2)

    mape = sum(abs_percentage_errors) / n
    rmse = math.sqrt(sum(squared_errors) / n)
    return mape, rmse

def example_digit_by_digit(true_cost, pred_cost):
    """Show MAPE calculation step by step"""
    error = abs(pred_cost - true_cost)
    mape = (error / true_cost) * 100
    print(f"True Cost = {true_cost}")
    print(f"Predicted Cost = {pred_cost}")
    print(f"Error = |{pred_cost} - {true_cost}| = {error}")
    print(f"MAPE = {error}/{true_cost} = {mape:.2f}%")

if __name__ == "__main__":
    
    result = estimate_costs_from_json("estimatepro-ai/ai_jobs/materials.json", project_name="MyProject")

    if isinstance(result, dict):
        print("✅ Estimated Costs (JSON):")
        print(json.dumps(result, indent=2))

        # Ground truth costs (update to match number of items)
        ground_truth_costs = [550, 2750, 396, 330, 220, 660, 550, 440, 495, 440]  

       
        predicted_costs = [item["cost"] for item in result["items"]]

        mape, rmse = calculate_accuracy(predicted_costs, ground_truth_costs)
        print(f"\n📊 Accuracy Metrics:")
        print(f"MAPE = {mape:.2f}%")
        print(f"RMSE = ${rmse:.2f}")

        
        example_digit_by_digit(ground_truth_costs[0], predicted_costs[0])
    else:
        print(result)
