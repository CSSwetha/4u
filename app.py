# /mnt/data/app.py
# Optimized and corrected version for "4u - Ingredient Analyzer Pro"
#
# Notes:
# - Original uploaded path: /mnt/data/app.py
# - Make sure pytesseract is installed and available on the host.
#   You can override the local tesseract path by setting the environment variable TESSERACT_CMD.
#
# Tech stack: Streamlit, pytesseract, Pillow, OpenCV, requests, beautifulsoup4, numpy, pandas
# Purpose: Extract ingredient lists from images, parse clean ingredients, classify into harmful/allergen/safe,
# and optionally fetch short web-sourced notes per ingredient (with retry/backoff).

import os
import re
import time
import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import List, Tuple, Optional, Set, Dict

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pytesseract
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingredient_analyzer")

# ---------------------- Tesseract config ----------------------
# Allow override by environment variable for deployment (Streamlit Cloud)
TESS_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESS_CMD

# ---------------------- Ingredient Database ----------------------
# Use your existing comprehensive lists but simplified here for readability.
# In production you can load a JSON/YAML file from repo for easier maintenance.
COMPREHENSIVE_INGREDIENTS = {
    'harmful': {
        'formaldehyde', 'hydroquinone', 'methylparaben', 'ethylparaben', 'propylparaben',
        'butylparaben', 'paraben', 'phthalate', 'triclosan', 'mercury', 'lead',
        'oxybenzone', 'sodium lauryl sulfate', 'sodium laureth sulfate',
        'benzene', 'toluene', 'coal tar'
    },
    'allergen': {
        'fragrance', 'parfum', 'methylisothiazolinone', 'methylchloroisothiazolinone',
        'lanolin', 'cinnamal', 'eugenol', 'limonene', 'linalool', 'neomycin'
    },
    'safe': {
        'water', 'aqua', 'glycerin', 'glycerine', 'hyaluronic acid', 'niacinamide',
        'panthenol', 'zinc oxide', 'titanium dioxide', 'ceramide', 'peptide', 'retinol',
        'squalane', 'aloe vera', 'shea butter'
    }
}

# Combine for fuzzy matching
ALL_KNOWN = set().union(*COMPREHENSIVE_INGREDIENTS.values())

# Variations mapping to increase match robustness
VARIATIONS = {
    'aqua': ['water', 'eau'],
    'glycerin': ['glycerine', 'glycerol'],
    'tocopherol': ['vitamin e'],
    'ascorbic acid': ['vitamin c'],
    'parfum': ['fragrance', 'perfume'],
    'sodium chloride': ['salt', 'table salt']
}

# Apply variations to sets
for canonical, variants in VARIATIONS.items():
    for cat in COMPREHENSIVE_INGREDIENTS:
        if canonical in COMPREHENSIVE_INGREDIENTS[cat]:
            COMPREHENSIVE_INGREDIENTS[cat].update(variants)
            ALL_KNOWN.update(variants)

# ---------------------- Utilities ----------------------


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def fix_ocr_errors(text: str) -> str:
    """Fix common OCR artifacts (spacing, broken words)."""
    if not text:
        return text
    fixes = {
        r'sod\s*ium': 'sodium',
        r'pot\s*assium': 'potassium',
        r'calc\s*ium': 'calcium',
        r'magn\s*esium': 'magnesium',
        r'but\s*ylene': 'butylene',
        r'prop\s*ylene': 'propylene',
        r'eth\s*yl': 'ethyl',
        r'meth\s*yl': 'methyl',
        r'dime\s*thyl': 'dimethyl',
        r'poly\s*acryl': 'polyacryl',
        r'hydro\s*xy': 'hydroxy',
        r'iso\s*propyl': 'isopropyl',
        r'xan\s*than': 'xanthan',
        r'dis\s*odium': 'disodium',
        r'cetear\s*yl': 'cetearyl',
        r'glycer\s*yl': 'glyceryl',
        r'laur\s*ic': 'lauric',
        r'palm\s*itic': 'palmitic',
        r'stear\s*ic': 'stearic',
        r'p\s*e\s*g': 'peg',
        r'e\s*d\s*t\s*a': 'edta'
    }
    t = text
    for pat, rep in fixes.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def fix_hyphenated_words(text: str) -> str:
    # Remove hyphenation inserted by linebreaks (e.g., "glycer-\nin")
    if not text:
        return text
    return re.sub(r'-\s*\n\s*', '', text)


def normalize_ingredient(s: str) -> str:
    if not s:
        return ""
    s = fix_ocr_errors(s)
    s = s.lower().strip()
    s = re.sub(r'\([^)]*\)', '', s)  # remove parentheses
    s = re.sub(r'\[[^\]]*\]', '', s)
    s = re.sub(r'\d+\.?\d*\s*%?', '', s)  # remove percentages
    s = re.sub(r'[^a-z0-9\-\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def fuzzy_match_ingredient(ingredient: str, known_list: Set[str], threshold: float = 0.80) -> Tuple[Optional[str], float]:
    """Return best match and score (or (None, 0))."""
    ing_norm = normalize_ingredient(ingredient)
    if not ing_norm:
        return None, 0.0

    # Direct exact match
    if ing_norm in known_list:
        return ing_norm, 1.0

    # Containment checks (better for multi-word)
    for k in known_list:
        if normalize_ingredient(k) in ing_norm or ing_norm in normalize_ingredient(k):
            return k, 0.95

    # Fuzzy loop - stop early if high confidence
    best = None
    best_score = 0.0
    for k in known_list:
        score = similar(ing_norm, normalize_ingredient(k))
        if score > best_score:
            best_score = score
            best = k
            if best_score >= 0.98:  # near perfect
                break

    if best_score >= threshold:
        return best, best_score
    return None, best_score

# ---------------------- OCR & Preprocessing ----------------------


def extract_text_ingredients_region(pil_image: Image.Image, debug: bool = False) -> str:
    """
    Robust OCR: try multiple preprocessing flows, multiple psm modes,
    choose best candidate based on heuristics, and provide final fallback.
    """
    try:
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        return ""

    # Make sure image is not tiny; scale up a bit if small
    h, w = img.shape[:2]
    scale = 1.0
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply denoising and contrast
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    # CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    preprocessed_images = []

    # adaptive thresh
    try:
        preprocessed_images.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, 15, 9))
    except Exception:
        pass

    # Otsu threshold
    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(otsu)
    except Exception:
        pass

    # Slight blur then threshold
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    try:
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(thr)
    except Exception:
        pass

    # Inverted
    try:
        preprocessed_images.append(cv2.bitwise_not(preprocessed_images[0]) if preprocessed_images else cv2.bitwise_not(gray))
    except Exception:
        pass

    # Keep a fallback of the raw gray image
    preprocessed_images.append(gray)

    # Tesseract configs to try
    configs = [
        r'--oem 3 --psm 6',   # Assume a single uniform block of text
        r'--oem 3 --psm 4',   # Assume a single column of text of variable sizes
        r'--oem 3 --psm 11',  # Sparse text
        r'--oem 3 --psm 3',   # Fully automatic page segmentation
    ]

    all_texts = []
    for img_proc in preprocessed_images:
        for config in configs:
            try:
                # Convert to PIL image for pytesseract
                text = pytesseract.image_to_string(img_proc, config=config)
                if text and text.strip():
                    all_texts.append(f"CONFIG:{config}\n{text.strip()}")
            except Exception as e:
                # keep going on OCR exceptions
                logger.debug("Tesseract error: %s", e)
                continue

    # Heuristic: choose the OCR output that looks most like ingredient lists
    ingredient_indicators = [
        'ingredients', 'water', 'glycerin', 'acid', 'oil', 'extract',
        'paraben', 'sulfate', 'edta', 'alcohol', 'contains', 'may contain'
    ]

    def score_text(candidate: str) -> float:
        t = candidate.lower()
        # indicator hits
        hits = sum(1 for k in ingredient_indicators if k in t)
        # more words slightly increases score
        words = len(t.split())
        # penalize if too short
        if words < 5:
            return hits * 1.0 - 5.0
        return hits * 2.0 + min(words / 20.0, 5.0)

    best_text = ""
    best_score = -9999.0
    for t in all_texts:
        t_fixed = fix_hyphenated_words(t)
        sc = score_text(t_fixed)
        if sc > best_score:
            best_score = sc
            best_text = t_fixed

    # Final fallback: return first OCR output or empty string
    final = best_text or (all_texts[0] if all_texts else "")
    final = fix_ocr_errors(final)
    if debug:
        logger.info("OCR candidates: %d, selected score=%.2f", len(all_texts), best_score)
    return final


# ---------------------- Parsing ----------------------


def parse_ingredients_from_text(text: str) -> List[str]:
    """
    Stronger parsing:
    - Attempts to locate the ingredients section
    - Splits on commas, semicolons, bullets and ' and '
    - Keeps acronyms (PEG, EDTA, BHA)
    - Deduplicates while preserving order
    """
    if not text:
        return []

    t = fix_hyphenated_words(text)
    # Normalize newlines for better section matching
    t = t.replace('\r', '\n').replace('\n', ' ')
    t = re.sub(r'\s+', ' ', t)

    # Try to find 'Ingredients:' or 'Contains:'
    # Expand the capture to common trailing separators (dot, new section words)
    pattern = re.compile(r'(ingredients?|contains?)\s*[:\-]\s*(.+?)(?:\.\s|$)', re.I)
    m = pattern.search(t)
    if m:
        ing_section = m.group(2)
    else:
        # fallback: if the label isn't present, try to split the full text and guess
        # by looking for ingredient-like words (commas plus known words)
        ing_section = t

    # Remove common trailing sections (directions, warnings etc.)
    ing_section = re.split(r'(directions?|warnings?|cautions?|for external use|keep out of reach|storage|store)', ing_section, flags=re.I)[0]

    # Split more robustly: commas, semicolons, bullets, middle dots, ' and '
    parts = re.split(r'[,\;\·•\n]|(?:\s+\band\b\s+)', ing_section)

    cleaned = []
    seen = set()
    skip_words = {'made', 'without', 'free', 'contains', 'tested', 'ingredients', 'direction', 'warning', 'caution', 'apply', 'use'}

    for p in parts:
        if not p:
            continue
        # Remove percentages and parentheses
        p = re.sub(r'\([^)]*\)', '', p)
        p = re.sub(r'\d+\.?\d*\s*%?', '', p)
        # Remove stray punctuation
        p = re.sub(r'[^A-Za-z0-9\-\s\&\/]', ' ', p)
        p = p.strip()
        if not p:
            continue
        pn = normalize_ingredient(p)
        # Keep short but valid tokens like "PEG", "EDTA", "BHA", allow 2+ chars
        if len(pn) < 2:
            continue
        if pn in skip_words:
            continue
        if pn not in seen:
            seen.add(pn)
            cleaned.append(pn)

    # If nothing parsed, attempt a more aggressive split by capitals and slashes
    if not cleaned:
        alt = re.split(r'[\/\-\|\n,;]', ing_section)
        for p in alt:
            p = normalize_ingredient(p)
            if p and p not in seen:
                seen.add(p)
                cleaned.append(p)

    return cleaned


# ---------------------- Classification ----------------------


class EnhancedIngredientDatabase:
    def __init__(self, ingredients_map: Dict[str, Set[str]]):
        self.harmful = set(ingredients_map.get('harmful', set()))
        self.allergen = set(ingredients_map.get('allergen', set()))
        self.safe = set(ingredients_map.get('safe', set()))
        self.all_known = self.harmful | self.allergen | self.safe

    def classify_ingredient(self, raw_ing: str) -> Tuple[str, str, float]:
        """
        Return (category, reason, confidence)
        category in ['harmful','allergen','safe']
        """
        ing_norm = normalize_ingredient(raw_ing)

        # Exact membership
        if ing_norm in self.harmful:
            return 'harmful', 'Known harmful ingredient', 1.0
        if ing_norm in self.allergen:
            return 'allergen', 'Known allergen', 1.0
        if ing_norm in self.safe:
            return 'safe', 'Known safe ingredient', 1.0

        # Try fuzzy match to categories
        for cat_name, cat_set, thresh in [
            ('harmful', self.harmful, 0.82),
            ('allergen', self.allergen, 0.82),
            ('safe', self.safe, 0.78)
        ]:
            match, score = fuzzy_match_ingredient(ing_norm, cat_set, threshold=thresh)
            if match and score >= thresh:
                return cat_name, f'Matched to {match}', float(score)

        # Keyword heuristics
        lower = ing_norm.lower()
        harmful_keywords = ['paraben', 'phthalate', 'sulfate', 'benzene', 'toluene', 'mercury', 'lead', 'formaldehyde']
        allergen_keywords = ['fragrance', 'parfum', 'limonene', 'linalool', 'cinnamal', 'eugenol']
        if any(k in lower for k in harmful_keywords):
            return 'harmful', 'Contains harmful keyword', 0.65
        if any(k in lower for k in allergen_keywords):
            return 'allergen', 'Contains allergen keyword', 0.65

        # Safe guess based on common suffixes
        safe_suffixes = ['-ate', 'acid', 'oil', 'extract', 'ol', 'ene']
        if any(lower.endswith(suf) or suf in lower for suf in safe_suffixes):
            return 'safe', 'Inferred safe based on pattern', 0.55

        # Unknown default to safe with low confidence (but flagged in UI)
        return 'safe', 'Assumed safe (low confidence)', 0.45


# ---------------------- Web lookup (optional) ----------------------


def requests_session_with_retries(total_retries: int = 3, backoff_factor: float = 0.3, status_forcelist: Tuple[int] = (429, 500, 502, 503, 504)) -> requests.Session:
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor, status_forcelist=list(status_forcelist))
    adapter = HTTPAdapter(max_retries=retries)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    s.headers.update({'User-Agent': 'IngredientAnalyzer/1.0 (+contact@example.com)'})
    return s


def fetch_ingredient_summary(ingredient: str, session: Optional[requests.Session] = None, timeout: int = 6) -> Optional[str]:
    """
    Tries to fetch a short summary for an ingredient using a simple web search approach.
    Note: External network calls depend on deployment (Streamlit Cloud allows outbound HTTP).
    This function is best-effort and fails silently returning None on errors.
    """
    if not ingredient:
        return None

    try:
        s = session or requests_session_with_retries()
        # Use DuckDuckGo's "html" search page as a lightweight search (no API key). Keep it minimal.
        # This is a simple heuristic: search for "ingredient + cosmetic" and parse first snippet.
        query = f"{ingredient} cosmetic ingredient safety"
        params = {"q": query}
        res = s.get("https://html.duckduckgo.com/html/", params=params, timeout=timeout)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        # DuckDuckGo's snippet is in <a class="result__a"> or <a class="result__snippet"> depending on page.
        snippet = soup.find("a", {"class": "result__a"})
        if not snippet:
            # try the first paragraph
            p = soup.find("p")
            if p and p.text:
                text = p.text.strip()
                return text[:350]
            return None
        # look for sibling snippet
        summary = snippet.get_text(separator=" ", strip=True)
        return summary[:350]
    except Exception as e:
        logger.debug("fetch_ingredient_summary failed for %s: %s", ingredient, str(e))
        return None


# ---------------------- Streamlit App ----------------------


def main():
    st.set_page_config(page_title="4u - Ingredient Analyzer Pro", page_icon="🧴", layout="wide")
    st.title("🧴 4u - Skin Product Analyzer ")
    st.markdown("**Designed for dermatological insights — extracts and classifies product ingredients.**")

    # Sidebar
    with st.sidebar:
        st.header("📊 Database Stats")
        db_preview = EnhancedIngredientDatabase(COMPREHENSIVE_INGREDIENTS)
        st.metric("🔴 Harmful", len(db_preview.harmful))
        st.metric("🟡 Allergens", len(db_preview.allergen))
        st.metric("🟢 Safe", len(db_preview.safe))
        st.markdown("---")
        st.header("Usage Tips")
        st.markdown("""
        - Upload a clear photo of the ingredient panel (good lighting, legible text).
        - If extraction fails, use the raw OCR text to edit/clean before re-parsing.
        - You can fetch web-sourced brief notes per ingredient (best-effort).
        """)
        st.markdown("---")
        st.markdown("**Tech stack**: Streamlit • pytesseract • OpenCV • BeautifulSoup • Requests")

    # Initialize DB in session (persist across runs)
    if 'db' not in st.session_state:
        st.session_state.db = EnhancedIngredientDatabase(COMPREHENSIVE_INGREDIENTS)

    db = st.session_state.db

    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded = st.file_uploader("📷 Upload product ingredient image", type=['png', 'jpg', 'jpeg'])
        # Option to override Tesseract path if needed
        tess_override = st.text_input("Tesseract path override (optional)", value=os.getenv("TESSERACT_CMD", ""))
        if tess_override:
            try:
                pytesseract.pytesseract.tesseract_cmd = tess_override
            except Exception:
                pass

        fetch_web = st.checkbox("🔎 Fetch short web notes per ingredient (best-effort)", value=False)

    with col2:
        st.markdown("### How it works")
        st.markdown("""
        1. OCR extracts text from the uploaded image using multiple preprocessing strategies.  
        2. The app parses the ingredient section, splits into ingredients and normalizes them.  
        3. Each ingredient is classified into **harmful**, **allergen**, or **safe** using fuzzy-matching + heuristics.  
        4. Optionally brief web summaries are fetched for each ingredient.
        """)

    if not uploaded:
        st.info("Upload an image of the ingredient list panel to start.")
        st.stop()

    # Show image preview
    try:
        image = Image.open(uploaded).convert('RGB')
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)
    except Exception as e:
        st.error(f"Could not open uploaded image: {e}")
        st.stop()

    # Start extraction and analysis
    if st.button("🔍 Extract & Analyze", type="primary"):
        with st.spinner("Running OCR and parsing..."):
            ocr_text = extract_text_ingredients_region(image, debug=True)
            parsed = parse_ingredients_from_text(ocr_text)

        # If OCR produced nothing, show error
        if not ocr_text or not ocr_text.strip():
            st.error("❌ OCR could not detect text in the image. Try a clearer or higher-resolution photo (good lighting, flat angle).")
            return

        # If OCR has text but parser extracted nothing, show raw and allow edit
        if not parsed:
            st.warning("⚠️ OCR detected text but no ingredients were parsed automatically. Inspect and edit the raw OCR output below, then re-parse.")
            with st.expander("🔍 Raw OCR Output (editable)"):
                edited = st.text_area("Edit OCR text (then click 'Re-parse edited text')", value=ocr_text, height=300)
            if st.button("↻ Re-parse edited text"):
                parsed = parse_ingredients_from_text(edited)
                ocr_text = edited  # use edited version downstream

            # show raw OCR anyway
            st.markdown("**If parsing still fails:** try to copy/paste a clean list of ingredients into the 'Manual ingredients' box below.")
            with st.expander("📝 Manual ingredients (paste comma separated)"):
                manual = st.text_area("Paste ingredients (comma-separated)", "")
                if st.button("✅ Use manual ingredients"):
                    manual_list = [normalize_ingredient(x) for x in re.split(r'[,\n;]+', manual) if x.strip()]
                    parsed = manual_list

        if not parsed:
            # Still nothing after fallback
            st.error("❌ Could not extract any ingredients. Please try another image or paste the ingredient list manually.")
            with st.expander("🔍 Raw OCR Output"):
                st.text_area("OCR Output", ocr_text, height=250)
            return

        # Show extracted item list
        st.success(f"✅ Found {len(parsed)} ingredient(s)")
        with st.expander("🔍 Extracted Ingredients (normalized)"):
            st.write(", ".join(parsed))

        # Analyze each ingredient
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        session = requests_session_with_retries() if fetch_web else None

        for i, ing in enumerate(parsed, start=1):
            status.text(f"Analyzing: {ing}")
            category, reason, confidence = db.classify_ingredient(ing)
            summary = None
            if fetch_web:
                # best-effort web fetch
                summary = fetch_ingredient_summary(ing, session=session)
            results.append({
                'ingredient': ing,
                'category': category,
                'reason': reason,
                'confidence': f"{int(confidence * 100)}%",
                'web_note': summary or ""
            })
            progress_bar.progress(int(i / len(parsed) * 100))
            # small pause for UI fluidity
            time.sleep(0.03)

        progress_bar.empty()
        status.empty()

        df = pd.DataFrame(results)

        # Summary
        st.markdown('---')
        st.subheader('📋 Analysis Summary')
        counts = df['category'].value_counts().to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('🔴 Harmful', counts.get('harmful', 0))
        c2.metric('🟡 Allergens', counts.get('allergen', 0))
        c3.metric('🟢 Safe', counts.get('safe', 0))
        c4.metric('ℹ️ Total Extracted', len(parsed))

        # Detailed table
        st.markdown('---')
        st.subheader('🔬 Detailed Report')
        # Show key columns and allow user to view web notes in expander
        df_display = df[['ingredient', 'category', 'reason', 'confidence']]
        st.dataframe(df_display, use_container_width=True)

        # Show web notes per ingredient in an expander
        if fetch_web:
            with st.expander("🔎 Web-sourced Notes (first ~350 chars)"):
                for r in results:
                    st.markdown(f"**{r['ingredient']}** — *{r['category']}* — {r['confidence']}")
                    if r['web_note']:
                        st.write(r['web_note'])
                    else:
                        st.write("_No web note found (or failed to fetch)._")
                    st.markdown("---")

        # Tabs for categories
        tabs = st.tabs(['🔴 Harmful', '🟡 Allergens', '🟢 Safe'])
        with tabs[0]:
            harms = df[df['category'] == 'harmful']
            if harms.empty:
                st.success('✅ No harmful ingredients detected')
            else:
                st.warning(f'⚠️ Found {len(harms)} harmful ingredient(s)')
                st.table(harms[['ingredient', 'reason', 'confidence']])

        with tabs[1]:
            alls = df[df['category'] == 'allergen']
            if alls.empty:
                st.success('✅ No allergens detected')
            else:
                st.warning(f'⚠️ Found {len(alls)} potential allergen(s)')
                st.table(alls[['ingredient', 'reason', 'confidence']])

        with tabs[2]:
            safes = df[df['category'] == 'safe']
            st.success(f'✅ Found {len(safes)} safe ingredient(s)')
            st.table(safes[['ingredient', 'reason', 'confidence']])

        # Download CSV
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='📥 Download Full Report (CSV)',
            data=csv_bytes,
            file_name=f'ingredient_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            use_container_width=True
        )

        # Confidence metrics
        def pct(x): return int(x.strip('%'))
        high_conf = len([r for r in results if pct(r['confidence']) >= 85])
        med_conf = len([r for r in results if 65 <= pct(r['confidence']) < 85])
        low_conf = len([r for r in results if pct(r['confidence']) < 65])

        st.markdown('---')
        st.subheader('ℹ️ Classification Confidence')
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("🎯 High (≥85%)", high_conf)
        cc2.metric("📊 Medium (65-84%)", med_conf)
        cc3.metric("⚡ Low (<65%)", low_conf)

        st.markdown("""
        **Confidence Levels:**
        - **High (≥85%)**: Direct or strong fuzzy match with the database.
        - **Medium (65-84%)**: Keyword-pattern based classification or weaker fuzzy match.
        - **Low (<65%)**: Heuristic inference; review manually.
        """)
        st.caption("⚠️ This tool is informational. For medical advice or product safety concerns, consult a dermatologist.")

        st.markdown("---")
        st.caption("💡 Tip: For best results, photograph the whole ingredients panel flat, in good light, avoiding glare and angle distortion.")

    # End of main action


if __name__ == "__main__":
    main()
