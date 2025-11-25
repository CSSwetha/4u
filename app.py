# /mnt/data/app.py
# 4u - Ingredient Analyzer Pro (Fixed: no reason, no confidence; UI Option D; fast web lookup)
# - Classification returns ONLY category strings
# - Streamlit-safe caching (no unhashable params)
# - Fast DuckDuckGo-based web verification & product scraping
# - Tesseract auto-detection for Streamlit Cloud

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

# ----------------------------
# Config & Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("4u")

# Streamlit page config (safe to call here)
st.set_page_config(page_title="4u - Ingredient Analyzer Pro", page_icon="🧴", layout="wide")

# Tesseract auto-detect (use env var TESSERACT_CMD or 'tesseract' in PATH)
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")

# ----------------------------
# Core Ingredient DB
# ----------------------------
COMPREHENSIVE_INGREDIENTS = {
    'harmful': {
        'formaldehyde', 'hydroquinone', 'methylparaben', 'ethylparaben', 'propylparaben',
        'butylparaben', 'paraben', 'phthalate', 'triclosan', 'triclocarban',
        'mercury', 'lead', 'oxybenzone', 'sodium lauryl sulfate', 'sodium laureth sulfate',
        'benzene', 'toluene', 'coal tar', 'petrolatum', 'bha', 'bht', 'diethanolamine',
    },
    'allergen': {
        'fragrance', 'parfum', 'methylisothiazolinone', 'methylchloroisothiazolinone',
        'lanolin', 'cinnamal', 'eugenol', 'limonene', 'linalool', 'neomycin',
        'citral', 'geraniol', 'citronellol'
    },
    'safe': {
        'water', 'aqua', 'glycerin', 'glycerine', 'hyaluronic acid', 'sodium hyaluronate',
        'niacinamide', 'panthenol', 'zinc oxide', 'titanium dioxide', 'ceramide',
        'peptide', 'vitamin c', 'ascorbic acid', 'vitamin e', 'tocopherol',
        'retinol', 'squalane', 'aloe vera', 'shea butter', 'jojoba oil'
    }
}

ALL_KNOWN = set().union(*COMPREHENSIVE_INGREDIENTS.values())

# ----------------------------
# Utilities
# ----------------------------
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
        r'e\s*d\s*t\s*a': 'edta',
    }
    t = text
    for pat, rep in fixes.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def fix_hyphenated_words(text: str) -> str:
    if not text:
        return text
    return re.sub(r'-\s*\n\s*', '', text)

def normalize_ingredient(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = fix_ocr_errors(s)
    s = re.sub(r'\([^)]*\)', '', s)  # remove parenthesis
    s = re.sub(r'[^a-z0-9\-\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ----------------------------
# OCR functions
# ----------------------------
def extract_text_ingredients_region(pil_image: Image.Image) -> str:
    """Robust OCR pipeline using multiple preprocess flows and tesseract configs"""
    try:
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.debug("Image conversion error: %s", e)
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
    except Exception:
        pass

    preprocessed_images = []
    # adaptive threshold
    try:
        preprocessed_images.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, 15, 9))
    except Exception:
        pass
    # otsu
    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(otsu)
    except Exception:
        pass
    # inverted
    try:
        preprocessed_images.append(cv2.bitwise_not(gray))
    except Exception:
        pass
    # raw gray fallback
    preprocessed_images.append(gray)

    configs = [r'--oem 3 --psm 6', r'--oem 3 --psm 4', r'--oem 3 --psm 11', r'--oem 3 --psm 3']

    all_texts = []
    for img_proc in preprocessed_images:
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(img_proc, config=cfg)
                if txt and txt.strip():
                    all_texts.append(txt.strip())
            except Exception:
                continue

    if not all_texts:
        return ""

    # choose best candidate by ingredient keyword presence + length
    indicators = ['ingredients', 'water', 'glycerin', 'acid', 'oil', 'extract', 'sodium', 'paraben', 'sulfate', 'edta']
    def score_text(t):
        t_low = t.lower()
        hits = sum(1 for k in indicators if k in t_low)
        words = len(t_low.split())
        return hits * 2.0 + min(words / 20.0, 5.0)

    best_text = max(all_texts, key=score_text)
    best_text = fix_hyphenated_words(best_text)
    return best_text

# ----------------------------
# Parsing logic (robust)
# ----------------------------
def parse_ingredients_from_text(text: str) -> List[str]:
    if not text:
        return []
    t = fix_hyphenated_words(text)
    t = t.replace('\r', ' ').replace('\n', ' ')
    t = re.sub(r'\s+', ' ', t)

    # Try to find Ingredients: or Contains: blocks
    m = re.search(r'(ingredients?|contains?)\s*[:\-]\s*(.+?)(?:\.\s|$)', t, re.I | re.S)
    ing_section = m.group(2) if m else t

    # remove trailing sections (directions, warnings)
    ing_section = re.split(r'(directions?|warnings?|caution|for external use|store)', ing_section, flags=re.I)[0]

    parts = re.split(r'[,\;•\·]|\sand\s', ing_section)
    cleaned = []
    seen = set()
    skip_words = {'made', 'without', 'free', 'tested', 'ingredients', 'apply', 'use', 'directions', 'warning', 'caution'}

    for p in parts:
        p = re.sub(r'\([^)]*\)|\[[^]]*\]|\d+\.?\d*\s*%', '', p)
        p = re.sub(r'[^A-Za-z0-9\-\s]', ' ', p)
        p = ' '.join(p.split()).strip()
        if not p:
            continue
        if p.lower() in skip_words:
            continue
        p_norm = normalize_ingredient(p)
        if len(p_norm) < 2:
            continue
        if p_norm not in seen:
            seen.add(p_norm)
            cleaned.append(p_norm)

    return cleaned

# ----------------------------
# Fast web lookup helpers (safe for streamlit cache)
# ----------------------------
def requests_session_with_retries(total_retries: int = 3, backoff_factor: float = 0.3):
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor, status_forcelist=[429,500,502,503,504])
    s.mount('http://', HTTPAdapter(max_retries=retries))
    s.mount('https://', HTTPAdapter(max_retries=retries))
    s.headers.update({'User-Agent': '4u-Ingredient-Analyzer/1.0 (+contact@example.com)'})
    return s

@st.cache_data(ttl=3600)
def ddg_search_html(query: str, timeout: int = 8) -> List[Dict]:
    """Return top DuckDuckGo HTML search results (title, href, snippet) - safe cached (no session param)"""
    session = requests_session_with_retries()
    try:
        res = session.get("https://html.duckduckgo.com/html/", params={'q': query}, timeout=timeout)
        if res.status_code != 200:
            return []
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        for a in soup.select("a.result__a"):
            href = a.get('href')
            title = a.get_text(strip=True)
            snippet_node = a.find_parent()
            snippet = ""
            sn = snippet_node.select_one(".result__snippet") if snippet_node else None
            if sn:
                snippet = sn.get_text(strip=True)
            results.append({'title': title, 'href': href, 'snippet': snippet})
        if not results:
            for r in soup.select(".result"):
                a = r.select_one("a")
                if not a:
                    continue
                href = a.get('href')
                title = a.get_text(strip=True)
                snippet = ""
                sn = r.select_one(".result__snippet")
                if sn:
                    snippet = sn.get_text(strip=True)
                results.append({'title': title, 'href': href, 'snippet': snippet})
        return results
    except Exception as e:
        logger.debug("ddg_search_html error: %s", e)
        return []

@st.cache_data(ttl=3600)
def fetch_page_text(url: str, timeout: int = 8) -> str:
    """Fetch page text - session created inside so function remains hashable for cache"""
    session = requests_session_with_retries()
    try:
        res = session.get(url, timeout=timeout)
        if res.status_code != 200:
            return ""
        return res.text
    except Exception as e:
        logger.debug("fetch_page_text error: %s", e)
        return ""

@st.cache_data(ttl=3600)
def fetch_ingredient_summary_from_web(ingredient: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fast-scrape: search for "<ingredient> cosmetic" and return (summary_text, source)
    This function is cached and does not accept unhashable params.
    """
    if not ingredient:
        return None, None
    session = requests_session_with_retries()
    query = f"{ingredient} cosmetic ingredient safety"
    results = ddg_search_html(query)
    if not results:
        return None, None
    for r in results[:6]:
        snippet = r.get('snippet') or ""
        href = r.get('href') or ""
        if snippet and len(snippet) > 40:
            domain = href.split('/')[2] if href and href.startswi
