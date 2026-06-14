import pytesseract
from pytesseract import Output
import cv2
import os
import numpy as np
import Levenshtein
import re

# --- Metrics ---
def cer(gt, pred):
    """Character Error Rate"""
    dist = Levenshtein.distance(gt.lower(), pred.lower())
    return dist / max(1, len(gt))

def word_accuracy(gt, pred):
    """Word-level Accuracy (robust)"""
    gt_words = gt.lower().split()
    pred_words = pred.lower().split()
    dist = Levenshtein.distance(" ".join(gt_words), " ".join(pred_words))
    return max(0, 1 - dist / max(1, len(" ".join(gt_words))))

def char_accuracy(gt, pred):
    """Character Accuracy (robust)"""
    dist = Levenshtein.distance(gt.lower(), pred.lower())
    return max(0, 1 - dist / max(1, len(gt)))

# --- OCR text cleaning ---
def clean_text(text):
    """
    Remove numbers, measurements, symbols, and single letters
    """
    text = re.sub(r"\b[A-Za-z]\b", "", text)  # remove single letters
    text = re.sub(r"[^A-Za-z\s]", " ", text)  # remove numbers and symbols
    text = re.sub(r"\s+", " ", text)          # normalize spaces
    return text.strip()

# --- Keep only valid room names ---
def filter_rooms(pred_text, room_list):
    """
    Keep only room names present in the OCR prediction
    """
    filtered = []
    pred_words = pred_text.upper().split()
    for room in room_list:
        # Check if all words in room appear in OCR output
        if all(word.upper() in pred_words for word in room.split()):
            filtered.append(room.upper())
    return " | ".join(filtered)

# --- OCR Evaluation ---
def evaluate_ocr(image_path, gt_text="", room_list=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    # OCR with confidence scores
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    
    # Combine text and clean
    pred_text = " ".join([t for t in data['text'] if t.strip() != ""]).strip()
    pred_text = clean_text(pred_text)

    # Filter by room names if provided
    if room_list:
        pred_text = filter_rooms(pred_text, room_list)

    # Confidence distribution
    confs = [int(c) for c in data['conf'] if isinstance(c, (int, float)) and c >= 0]
    mu_conf, std_conf = (np.mean(confs), np.std(confs)) if confs else (0, 0)

    results = {
        "image": os.path.basename(image_path),
        "pred_text": pred_text,
        "gt_text": gt_text,
        "char_acc": char_accuracy(gt_text, pred_text) if gt_text else None,
        "word_acc": word_accuracy(gt_text, pred_text) if gt_text else None,
        "cer": cer(gt_text, pred_text) if gt_text else None,
        "mu_conf": mu_conf,
        "std_conf": std_conf
    }
    return results

# --- Main ---
if __name__ == "__main__":
    test_image = r"D:\Company Projects\estimatepro-ai\datasets\testing\images\1.jpg"
    
    # Ground truth text
    gt_text = "Master Bedroom Kitchen 2 Car Garage"
    
    # Predefined room names
    room_names = ["Master Bedroom", "Kitchen", "Bathroom", "Garage", "Living Room", "Dining Room", "2 Car Garage"]

    result = evaluate_ocr(test_image, gt_text, room_list=room_names)

    print("\n--- OCR Evaluation ---")
    print(f"Image: {result['image']}")
    if result['char_acc'] is not None:
        print(f"Character Accuracy: {result['char_acc']*100:.2f}%")
        print(f"CER: {result['cer']:.3f}")
        print(f"Word Accuracy: {result['word_acc']*100:.2f}%")
    else:
        print("⚠️ No ground truth provided, only confidence scores available.")
    print(f"Confidence mean: {result['mu_conf']:.2f}, std: {result['std_conf']:.2f}")
    print("\nPredicted OCR Text:")
    print(result['pred_text'])
