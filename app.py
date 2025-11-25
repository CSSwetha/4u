import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests
from bs4 import BeautifulSoup
import json
import os
import re
import time
from datetime import datetime
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from difflib import SequenceMatcher
from urllib.parse import quote_plus

# ------------------------------------------------------------
#  STREAMLIT CLOUD FIX: Auto-detect Tesseract path
# ------------------------------------------------------------

# Remove the old Windows-only path like:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# New universal solution:
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")


# ------------------------------------------------------------
#  INGREDIENT DATABASE
# ------------------------------------------------------------

COMPREHENSIVE_INGREDIENTS = {
    'harmful': {
        'formaldehyde', 'hydroquinone', 'methylparaben', 'ethylparaben', 'propylparaben',
        'butylparaben', 'paraben', 'phthalate', 'triclosan', 'triclocarban',
        'mercury', 'lead', 'oxybenzone', 'sodium lauryl sulfate', 'sodium laureth sulfate',
        'benzene', 'toluene', 'coal tar', 'petrolatum', 'bha', 'bht'
    },

    'allergen': {
        'fragrance', 'parfum', 'methylisothiazolinone', 'methylchloroisothiazolinone',
        'lanolin', 'cinnamal', 'eugenol', 'limonene', 'linalool',
        'citral', 'benzyl alcohol', 'geraniol', 'citronellol'
    },

    'safe': {
        'water', 'aqua', 'glycerin', 'hyaluronic acid', 'niacinamide', 'panthenol',
        'zinc oxide', 'titanium dioxide', 'ceramide', 'peptide',
        'vitamin c', 'ascorbic acid', 'vitamin e', 'tocopherol',
        'retinol', 'squalane', 'aloe vera', 'shea butter', 'jojoba oil'
    }
}

ALL_KNOWN = set().union(
    COMPREHENSIVE_INGREDIENTS['harmful'],
    COMPREHENSIVE_INGREDIENTS['allergen'],
    COMPREHENSIVE_INGREDIENTS['safe']
)


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def fix_ocr_errors(text):
    fixes = {
        r'sod\s*ium': 'sodium',
        r'pot\s*assium': 'potassium',
        r'glycer\s*yl': 'glyceryl',
        r'cetear\s*yl': 'cetearyl',
        r'p\s*e\s*g': 'peg',
        r'e\s*d\s*t\s*a': 'edta',
    }
    for pattern, repl in fixes.items():
        text = re.sub(pattern, repl, text, flags=re.I)
    return text.strip()


def fix_hyphens(text):
    return re.sub(r'-\s*\n', '', text)


def normalize_ingredient(s):
    if not s:
        return ""
    s = fix_ocr_errors(s.lower())
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'[^a-z0-9\s\-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ------------------------------------------------------------
# OCR ENGINE (Streamlit Cloud compatible)
# ------------------------------------------------------------

def ocr_extract(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    methods = [
        gray,
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 15, 9),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 15, 9)
    ]

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 4",
        "--oem 3 --psm 11",
        "--oem 3 --psm 3"
    ]

    outputs = []
    for m in methods:
        for c in configs:
            try:
                text = pytesseract.image_to_string(m, config=c)
                if text.strip():
                    outputs.append(text)
            except:
                    pass

    if not outputs:
        return ""

    # pick the OCR output with most ingredient-like keywords
    key_words = ["water", "glycer", "acid", "oil", "extract", "ingredients", "sodium"]
    best = max(outputs, key=lambda t: sum(k in t.lower() for k in key_words))

    return fix_hyphens(best)


# ------------------------------------------------------------
# PARSING INGREDIENT LIST
# ------------------------------------------------------------

def parse_ingredients(text):
    if not text.strip():
        return []

    text = text.replace("\n", " ")

    # try to extract the section after "Ingredients:"
    m = re.search(r'(ingredients?|contains?)\s*[:\-]\s*(.+)', text, flags=re.I)
    ing_section = m.group(2) if m else text

    # split on comma, semicolon, or “ and ”
    parts = re.split(r'[,\;]| and ', ing_section)

    cleaned = []
    for p in parts:
        p = normalize_ingredient(p)
        if len(p) > 2:
            cleaned.append(p)

    # dedupe
    final = []
    seen = set()
    for c in cleaned:
        if c not in seen:
            seen.add(c)
            final.append(c)

    return final


# ------------------------------------------------------------
# CLASSIFICATION LOGIC
# ------------------------------------------------------------

class IngredientDB:
    def __init__(self):
        self.harmful = COMPREHENSIVE_INGREDIENTS['harmful']
        self.allergen = COMPREHENSIVE_INGREDIENTS['allergen']
        self.safe = COMPREHENSIVE_INGREDIENTS['safe']

    def classify(self, ing):
        n = normalize_ingredient(ing)

        if n in self.harmful:
            return "harmful", "Known harmful", 1.0

        if n in self.allergen:
            return "allergen", "Known allergen", 1.0

        if n in self.safe:
            return "safe", "Known safe", 1.0

        # fuzzy matching
        match, score = fuzzy_find(n, ALL_KNOWN)
        if match and score >= 0.82:
            if match in self.harmful:
                return "harmful", f"Matched harmful: {match}", score
            if match in self.allergen:
                return "allergen", f"Matched allergen: {match}", score
            if match in self.safe:
                return "safe", f"Matched safe: {match}", score

        # heuristic fallback
        if "paraben" in n or "sulfate" in n or "phthalate" in n:
            return "harmful", "Contains harmful keyword", 0.6

        if "fragrance" in n or "linalool" in n or "limonene" in n:
            return "allergen", "Contains allergen keyword", 0.6

        return "safe", "Assumed safe (low confidence)", 0.5


def fuzzy_find(word, known):
    best = None
    best_score = 0
    for k in known:
        score = similar(word, k)
        if score > best_score:
            best = k
            best_score = score
    if best_score >= 0.75:
        return best, best_score
    return None, 0


# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

def main():
    st.set_page_config(page_title="4u - Ingredient Analyzer Pro", page_icon="🧴", layout="wide")
    st.title("🧴 4u - Dermatological Ingredient Analyzer")

    st.write("Upload an ingredient panel image. The system extracts, parses, and classifies ingredients into **harmful**, **allergen**, or **safe**.")

    db = IngredientDB()

    uploaded = st.file_uploader("📷 Upload ingredient image", type=["png", "jpg", "jpeg"])

    if not uploaded:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("🔍 Extract & Analyze"):
        with st.spinner("Extracting text..."):
            text = ocr_extract(image)

        if not text.strip():
            st.error("❌ OCR could not detect text in the image. Try a clearer or higher-resolution photo.")
            return

        with st.expander("🔍 Raw OCR Text"):
            st.text_area("OCR Output", text, height=200)

        ingredients = parse_ingredients(text)

        if not ingredients:
            st.warning("⚠️ OCR succeeded, but no ingredients were detected. Edit the text above if needed.")
            return

        st.success(f"✅ Found {len(ingredients)} ingredients")

        # classify
        results = []
        prog = st.progress(0)

        for i, ing in enumerate(ingredients):
            cat, reason, conf = db.classify(ing)
            results.append({
                "ingredient": ing,
                "category": cat,
                "reason": reason,
                "confidence": f"{int(conf*100)}%"
            })
            prog.progress((i+1) / len(ingredients))

        df = pd.DataFrame(results)

        st.subheader("📋 Summary")
        c = df["category"].value_counts()
        st.metric("Harmful", c.get("harmful", 0))
        st.metric("Allergens", c.get("allergen", 0))
        st.metric("Safe", c.get("safe", 0))

        st.subheader("Detailed Report")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV Report", csv, "ingredient_report.csv", "text/csv")


if __name__ == "__main__":
    main()
