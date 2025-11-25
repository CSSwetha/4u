# FULL UPDATED APP.PY WITH HIGH-CONFIDENCE CLASSIFICATION + ALWAYS-ON WEB LOOKUP
# (All patches integrated cleanly)

import os
import re
import time
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Optional, Tuple, Dict

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("4u")

# ------------------------------------------------------------
# Tesseract auto-detection (Streamlit Cloud compatible)
# ------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")

# ------------------------------------------------------------
# Ingredient Database
# ------------------------------------------------------------
COMPREHENSIVE_INGREDIENTS = {
    'harmful': {
        'formaldehyde', 'hydroquinone', 'methylparaben', 'ethylparaben', 'propylparaben',
        'butylparaben', 'paraben', 'phthalate', 'triclosan', 'mercury', 'lead', 'oxybenzone',
        'sodium lauryl sulfate', 'sodium laureth sulfate', 'benzene', 'toluene', 'coal tar'
    },
    'allergen': {
        'fragrance', 'parfum', 'methylisothiazolinone', 'methylchloroisothiazolinone',
        'lanolin', 'cinnamal', 'eugenol', 'limonene', 'linalool', 'citral'
    },
    'safe': {
        'water', 'aqua', 'glycerin', 'hyaluronic acid', 'niacinamide', 'panthenol', 'zinc oxide',
        'titanium dioxide', 'ceramide', 'peptide', 'retinol', 'squalane', 'aloe vera',
        'shea butter', 'jojoba oil'
    }
}

ALL_KNOWN = set().union(*COMPREHENSIVE_INGREDIENTS.values())

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def fix_ocr_errors(text: str) -> str:
    if not text:
        return text
    fixes = {
        r'sod\s*ium': 'sodium',
        r'pot\s*assium': 'potassium',
        r'cetear\s*yl': 'cetearyl',
        r'p\s*e\s*g': 'peg',
        r'e\s*d\s*t\s*a': 'edta'
    }
    for pat, rep in fixes.items():
        text = re.sub(pat, rep, text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fix_hyphenated_words(text: str) -> str:
    return re.sub(r'-\s*\n\s*', '', text)

def normalize_ingredient(s: str) -> str:
    if not s:
        return ''
    s = s.lower().strip()
    s = fix_ocr_errors(s)
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'[^a-z0-9\-\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ------------------------------------------------------------
# OCR extraction
# ------------------------------------------------------------
def extract_text_ingredients_region(pil_image: Image.Image) -> str:
    try:
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except:
        return ""

    h, w = img.shape[:2]
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except:
        pass

    preproc = []
    preproc.append(gray)

    try:
        preproc.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 15, 9))
    except:
        pass

    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preproc.append(otsu)
    except:
        pass

    try:
        preproc.append(cv2.bitwise_not(gray))
    except:
        pass

    configs = ['--oem 3 --psm 6', '--oem 3 --psm 4', '--oem 3 --psm 11', '--oem 3 --psm 3']

    results = []
    for p in preproc:
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(p, config=cfg)
                txt = txt.strip()
                if txt:
                    results.append(txt)
            except:
                pass

    if not results:
        return ""

    keywords = ["water", "glycer", "acid", "oil", "extract", "ingredients", "sodium"]
    def score(t):
        t_low = t.lower()
        hits = sum(k in t_low for k in keywords)
        return hits * 2 + min(len(t_low.split()) / 20, 5)

    best = max(results, key=score)
    return fix_hyphenated_words(best)

# ------------------------------------------------------------
# Parsing ingredients
# ------------------------------------------------------------
def parse_ingredients_from_text(text: str) -> List[str]:
    if not text:
        return []

    t = fix_hyphenated_words(text)
    t = t.replace('\r', ' ').replace('\n', ' ')
    t = re.sub(r"\s+", " ", t)

    m = re.search(r'(ingredients?|contains?)\s*[:\-]\s*(.+?)(?:\.\s|$)', t, re.I | re.S)
    ing_section = m.group(2) if m else t

    ing_section = re.split(r'(directions?|warnings?|caution|for external use|store)', ing_section, flags=re.I)[0]

    parts = re.split(r'[;,•·]|\sand\s', ing_section)
    cleaned = []
    seen = set()

    for p in parts:
        p = re.sub(r'\([^)]*\)', '', p)
        p = re.sub(r'[^A-Za-z0-9\-\s]', ' ', p)
        p = p.strip()
        if not p:
            continue
        p_norm = normalize_ingredient(p)
        if len(p_norm) < 2:
            continue
        if p_norm not in seen:
            seen.add(p_norm)
            cleaned.append(p_norm)

    return cleaned

# ------------------------------------------------------------
# High-confidence classification (patched version)
# ------------------------------------------------------------
class EnhancedIngredientDatabase:
    def __init__(self, mapping=None):
        mapping = mapping or COMPREHENSIVE_INGREDIENTS
        self.harmful = set(mapping.get('harmful', []))
        self.allergen = set(mapping.get('allergen', []))
        self.safe = set(mapping.get('safe', []))
        self.all_known = self.harmful | self.allergen | self.safe

    def classify_ingredient(self, ingredient: str):
        ing = normalize_ingredient(ingredient)

        # Exact matches
        if ing in self.harmful:
            return ('harmful', 'Exact harmful match', 1.0)
        if ing in self.allergen:
            return ('allergen', 'Exact allergen match', 1.0)
        if ing in self.safe:
            return ('safe', 'Exact safe match', 1.0)

        # Fuzzy match
        best = None
        best_score = 0.0
        for k in self.all_known:
            score = similar(ing, normalize_ingredient(k))
            if score > best_score:
                best_score = score
                best = k

        if best_score >= 0.90:
            if best in self.harmful:
                return ('harmful', f'High-confidence fuzzy match: {best}', best_score)
            if best in self.allergen:
                return ('allergen', f'High-confidence fuzzy match: {best}', best_score)
            if best in self.safe:
                return ('safe', f'High-confidence fuzzy match: {best}', best_score)

        # Keyword-based match
        harmful_keys = ['paraben', 'phthalate', 'sulfate', 'benzene', 'toluene']
        allergen_keys = ['fragrance', 'parfum', 'limonene', 'linalool']

        if any(k
