import cv2
import pytesseract
import re
import json
import os
from openai import OpenAI
from dotenv import load_dotenv  
# Load environment variables from .env file
load_dotenv()

#  Initialize OpenAI client (reads from .env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Configure pytesseract (set path to tesseract.exe if needed on Windows)
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_rooms_from_plan(image_path: str) -> dict:
    """
    Hybrid OCR: Use Tesseract + GPT to extract structured rooms JSON.
    """
    try:
        # --- Step 1: Run Tesseract ---
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"❌ Could not read image at {image_path}"}

        raw_text = pytesseract.image_to_string(img)

        # --- Step 2: Ask GPT to clean + structure ---
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a construction OCR specialist. "
                        "The user gives you messy OCR text from a floor plan. "
                        "Extract room types and areas in square meters. "
                        "If a room has no dimensions, set area_sqm=null. "
                        "Output strictly in JSON format like:\n"
                        '{"rooms":[{"type":"Office","area_sqm":30.5},{"type":"Kitchen","area_sqm":12}]}'
                    )
                },
                {
                    "role": "user",
                    "content": f"OCR text:\n{raw_text}"
                }
            ],
            max_tokens=500
        )

        text_output = response.choices[0].message.content.strip()

        # --- Step 3: Clean response from GPT ---
        cleaned = re.sub(r"^```json|```$", "", text_output, flags=re.MULTILINE).strip()

        # --- Step 4: Return structured JSON ---
        return json.loads(cleaned)

    except Exception as e:
        return {"error": f"❌ Error in OCR: {e}"}


# ✅ Quick test
if __name__ == "__main__":
    test_image = r"D:\Company Projects\estimatepro-ai\datasets\testing\images\1.jpg"
    result = extract_rooms_from_plan(test_image)
    print("\n✅ OCR Result:\n", json.dumps(result, indent=2))
