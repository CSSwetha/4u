# ------------------------------------------------------------
# 4u Ingredient Analyzer - Clean, No-Review Version
# - Classification rewritten (SAFE / HARMFUL / ALLERGEN)
# - No "reason"
# - No "confidence"
# - UI shows only three categories
# - Fast web lookup (DuckDuckGo HTML) - cached safely
# ------------------------------------------------------------

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

# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("4u")

# Use env var or default tesseract command
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")

# -----------------------------------------------------
# Ingredient DB (Simplified)
# -----------------------------------------------------
COMPREHENSIVE_INGREDIENTS = {
    'harmful': {
        'formaldehyde', 'hydroquinone', 'methylparaben', 'ethylparaben',
        'propylparaben', 'butylparaben', 'paraben', 'phthalate',
        'triclosan', 'mercury', 'lead', 'oxybenzone',
        'sodium lauryl sulfate', 'sodium laureth sulfate',
        'benzene', 'toluene', 'coal tar'
    },
    'allergen': {
        'fragrance', 'parfum', 'methylisothiazolinone',
        'methylchloroisothiazolinone', 'lanolin', 'cinnamal',
        'eugenol', 'limonene', 'linalool', 'citral'
    },
    'safe': {
        'water', 'aqua', 'glycerin', 'hyaluronic acid', 'niacinamide',
        'panthenol', 'zinc oxide', 'titanium dioxide', 'ceramide',
        'peptide', 'retinol', 'squalane', 'aloe vera', 'shea butter',
        'jojoba oil'
    }
}

ALL_KNOWN = set().union(*COMPREHENSIVE_INGREDIENTS.values())

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def fix_ocr_errors(text: str) -> str:
    if not text:
        return text
    patterns = {
        r'sod\s*ium': 'sodium',
        r'pot\s*assium': 'potassium',
        r'cetear\s*yl': 'cetearyl',
        r'p\s*e\s*g': 'peg',
        r'e\s*d\s*t\s*a': 'edta'
    }
    for pat, rep in patterns.items():
        text = re.sub(pat, rep, text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_ingredient(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = fix_ocr_errors(s)
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'[^a-z0-9\-\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# -----------------------------------------------------
# OCR Engine
# -----------------------------------------------------
def extract_text_ingredients_region(pil_image: Image.Image) -> str:
    try:
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return ""

    h, w = img.shape[:2]
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    preproc = []
    try:
        preproc.append(cv2.adaptiveThreshold(gray, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 15, 9))
    except Exception:
        pass
    try:
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preproc.append(otsu)
    except Exception:
        pass

    preproc.append(gray)
    try:
        preproc.append(cv2.bitwise_not(gray))
    except Exception:
        pass

    configs = ["--oem 3 --psm 6", "--oem 3 --psm 4", "--oem 3 --psm 11"]

    best_txt = ""
    best_len = 0

    for p in preproc:
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(p, config=cfg)
                txt = txt.strip()
                if len(txt) > best_len:
                    best_len = len(txt)
                    best_txt = txt
            except Exception:
                pass

    return best_txt if best_txt else ""

# -----------------------------------------------------
# Ingredient Parsing
# -----------------------------------------------------
def parse_ingredients_from_text(text: str) -> List[str]:
    if not text:
        return []

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text)

    # match "ingredients: ..."
    m = re.search(r'(ingredients?|contains?)\s*[:\-]\s*(.+)', text, re.I)
    if m:
        text = m.group(2)

    parts = re.split(r'[,\;•·]|\sand\s', text)
    cleaned = []

    for p in parts:
        p = p.strip()
        p = normalize_ingredient(p)
        if len(p) > 1:
            cleaned.append(p)

    # dedupe while preserving order
    return list(dict.fromkeys(cleaned))

# -----------------------------------------------------
# Classification — No-Review (only SAFE/HARMFUL/ALLERGEN)
# -----------------------------------------------------
class CleanClassifier:
    def __init__(self):
        self.harmful = COMPREHENSIVE_INGREDIENTS['harmful']
        self.allergen = COMPREHENSIVE_INGREDIENTS['allergen']
        self.safe = COMPREHENSIVE_INGREDIENTS['safe']
        self.all_known = ALL_KNOWN

    def classify(self, ing: str) -> str:
        ing_n = normalize_ingredient(ing)

        # exact match
        if ing_n in self.harmful:
            return "HARMFUL"
        if ing_n in self.allergen:
            return "ALLERGEN"
        if ing_n in self.safe:
            return "SAFE"

        # fuzzy match
        best = None
        best_score = 0.0
        for k in self.all_known:
            sc = similar(ing_n, normalize_ingredient(k))
            if sc > best_score:
                best = k
                best_score = sc

        if best and best_score >= 0.82:
            if best in self.harmful:
                return "HARMFUL"
            if best in self.allergen:
                return "ALLERGEN"
            if best in self.safe:
                return "SAFE"

        # keyword detection for strong hazard/allergen signals
        lower = ing_n.lower()
        harmful_keys = ['paraben', 'phthalate', 'sulfate', 'benzene', 'toluene', 'mercury', 'lead']
        allergen_keys = ['fragrance', 'parfum', 'limonene', 'linalool', 'benzyl', 'citral', 'geraniol']

        if any(k in lower for k in harmful_keys):
            return "HARMFUL"
        if any(k in lower for k in allergen_keys):
            return "ALLERGEN"

        # DEFAULT: classify unknown as SAFE (user requested no REVIEW)
        return "SAFE"

# -----------------------------------------------------
# Web Scraping Helpers (safe for Streamlit cache)
# -----------------------------------------------------
def requests_session_with_retries():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3,
                    status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "4u-Analyzer"})
    return s

@st.cache_data(ttl=3600)
def ddg_search_html(query: str) -> List[Dict]:
    url = "https://html.duckduckgo.com/html/"
    try:
        r = requests.get(url, params={"q": query}, timeout=8)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out = []
        for a in soup.select("a.result__a"):
            href = a.get("href")
            title = a.get_text(strip=True)
            snippet = ""
            sn = a.find_parent().select_one(".result__snippet")
            if sn:
                snippet = sn.get_text(strip=True)
            domain = href.split('/')[2] if (href and href.startswith("http")) else "DDG"
            out.append({"href": href, "title": title, "snippet": snippet, "domain": domain})
        # fallback
        if not out:
            for r in soup.select(".result"):
                a = r.select_one("a")
                if not a:
                    continue
                href = a.get("href")
                title = a.get_text(strip=True)
                snippet = ""
                sn = r.select_one(".result__snippet")
                if sn:
                    snippet = sn.get_text(strip=True)
                domain = href.split('/')[2] if (href and href.startswith("http")) else "DDG"
                out.append({"href": href, "title": title, "snippet": snippet, "domain": domain})
        return out
    except Exception:
        return []

@st.cache_data(ttl=3600)
def fetch_page_text(url: str) -> str:
    if not url or not url.startswith("http"):
        return ""
    try:
        r = requests.get(url, timeout=8)
        return r.text if r.status_code == 200 else ""
    except Exception:
        return ""

def extract_ingredient_section_from_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for tag in soup.find_all(text=re.compile(r'Ingredients?', re.I)):
        parent = tag.parent
        if parent:
            candidates.append(parent.get_text(separator=" ", strip=True))
            sib = parent.find_next_sibling()
            for _ in range(3):
                if not sib:
                    break
                candidates.append(sib.get_text(separator=" ", strip=True))
                sib = sib.find_next_sibling()
    if not candidates:
        for ul in soup.find_all(['ul','ol']):
            txt = ul.get_text(separator=" ", strip=True)
            if len(txt.split()) > 6:
                candidates.append(txt)
    best = ""
    for c in candidates:
        if 'ingredient' in c.lower() or ',' in c:
            if len(c) > len(best):
                best = c
    return best

@st.cache_data(ttl=3600)
def fetch_ingredient_summary_from_web(ingredient: str) -> Optional[str]:
    if not ingredient:
        return None
    query = f"{ingredient} cosmetic safety"
    results = ddg_search_html(query)
    if not results:
        return None
    for r in results[:6]:
        snippet = r.get("snippet") or ""
        href = r.get("href") or ""
        if snippet and len(snippet) > 40:
            return snippet[:400]
        if href and href.startswith("http"):
            html = fetch_page_text(href)
            if html:
                soup = BeautifulSoup(html, "html.parser")
                for p in soup.find_all("p"):
                    txt = p.get_text(" ", strip=True)
                    if len(txt) > 80 and ingredient.split()[0] in txt.lower():
                        return txt[:400]
                meta = soup.find('meta', attrs={'name':'description'}) or soup.find('meta', attrs={'property':'og:description'})
                if meta and meta.get('content'):
                    return meta.get('content')[:400]
    first = results[0]
    return (first.get('snippet')[:400] if first.get('snippet') else first.get('title'))

# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
def main():
    st.set_page_config(page_title="4u Ingredient Analyzer", layout="wide")
    st.title("🧴 4u — Ingredient Analyzer (No-Review)")
    st.caption("Classification: SAFE / HARMFUL / ALLERGEN (unknown defaults to SAFE)")

    classifier = CleanClassifier()

    col1, col2 = st.columns([1,2])

    with col1:
        uploaded = st.file_uploader("Upload ingredient image", ["jpg","jpeg","png"])
        manual = st.text_area("Or paste ingredients (comma-separated)")
        use_web = st.checkbox("Enable web lookup")

    with col2:
        if st.button("Analyze"):
            # OCR extraction
            ocr_text = ""
            if uploaded:
                try:
                    img = Image.open(uploaded).convert("RGB")
                except Exception as e:
                    st.error(f"Could not open image: {e}")
                    return
                with st.spinner("Extracting text..."):
                    ocr_text = extract_text_ingredients_region(img)

            source = manual.strip() if manual.strip() else ocr_text

            if not source:
                st.error("❌ No text found. Upload an image or paste ingredient text.")
                return

            ingredients = parse_ingredients_from_text(source)
            if not ingredients:
                st.warning("No ingredients detected.")
                return

            st.success(f"Detected {len(ingredients)} ingredients.")

            rows = []
            for ing in ingredients:
                cat = classifier.classify(ing)
                web_note = fetch_ingredient_summary_from_web(ing) if use_web else ""
                rows.append({"ingredient": ing, "category": cat, "web_note": web_note or ""})

            df = pd.DataFrame(rows)

            # Summary - only three categories
            st.subheader("Summary")
            counts = df["category"].value_counts().to_dict()
            c1, c2, c3 = st.columns(3)
            c1.metric("HARMFUL", counts.get("HARMFUL", 0))
            c2.metric("ALLERGEN", counts.get("ALLERGEN", 0))
            c3.metric("SAFE", counts.get("SAFE", 0))

            st.subheader("Detailed Results")
            st.dataframe(df[["ingredient","category"]], use_container_width=True)

            if use_web:
                with st.expander("Web Notes"):
                    for r in rows:
                        st.markdown(f"**{r['ingredient']}** — {r['category']}")
                        st.write(r["web_note"] or "_No web info available._")
                        st.markdown("---")

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "ingredient_report.csv", mime="text/csv")

if __name__ == "__main__":
    main()
