# app.py
"""
4u - Ingredient Analyzer Pro (Extended)
Features:
 - OCR extraction from uploaded images (pytesseract + OpenCV preprocessing)
 - Ingredient parsing & classification (harmful / allergen / safe)
 - OPTIONAL: Live web-sourced short summaries per ingredient (best-effort)
 - OPTIONAL: Product search & ingredient scraping (search by product name or provide URL)
 - Streamlit Cloud friendly (auto Tesseract detection via env var or 'tesseract' in PATH)
"""

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

# -------------------------
# Config & Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("4u")

# Auto-detect Tesseract binary (Streamlit Cloud friendly)
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")

# -------------------------
# Base Ingredient Database
# (For maintainability, consider loading from JSON/YAML in repo)
# -------------------------
COMPREHENSIVE_INGREDIENTS = {
    'harmful': {
        'formaldehyde', 'hydroquinone', 'methylparaben', 'ethylparaben', 'propylparaben',
        'butylparaben', 'paraben', 'phthalate', 'triclosan', 'mercu', 'mercury',
        'lead', 'oxybenzone', 'sodium lauryl sulfate', 'sodium laureth sulfate',
        'benzene', 'toluene', 'coal tar'
    },
    'allergen': {
        'fragrance', 'parfum', 'methylisothiazolinone', 'methylchloroisothiazolinone',
        'lanolin', 'cinnamal', 'eugenol', 'limonene', 'linalool', 'citral'
    },
    'safe': {
        'water', 'aqua', 'glycerin', 'hyaluronic acid', 'niacinamide', 'panthenol',
        'zinc oxide', 'titanium dioxide', 'ceramide', 'peptide', 'retinol',
        'squalane', 'aloe vera', 'shea butter', 'jojoba oil'
    }
}
ALL_KNOWN = set().union(*COMPREHENSIVE_INGREDIENTS.values())

# -------------------------
# Utilities
# -------------------------
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
    t = text
    for pat, rep in patterns.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

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

# -------------------------
# OCR & Preprocessing
# -------------------------
def extract_text_ingredients_region(pil_image: Image.Image) -> str:
    """
    Robust OCR pipeline. Tries several preprocessing steps and multiple Tesseract configs,
    then picks the best candidate by simple heuristics.
    """
    try:
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error("Image conversion failed: %s", e)
        return ""

    h, w = img.shape[:2]
    # scale up small images for better OCR
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)

    # CLAHE contrast improvement
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    preproc = []
    # adaptive thresh
    try:
        preproc.append(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 15, 9))
    except Exception:
        pass
    # Otsu
    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preproc.append(otsu)
    except Exception:
        pass
    # inverted
    try:
        preproc.append(cv2.bitwise_not(gray))
    except Exception:
        pass
    # raw gray as fallback
    preproc.append(gray)

    configs = ['--oem 3 --psm 6', '--oem 3 --psm 4', '--oem 3 --psm 11', '--oem 3 --psm 3']

    candidates = []
    for img_proc in preproc:
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(img_proc, config=cfg)
                if txt and txt.strip():
                    candidates.append((cfg, txt.strip()))
            except Exception as e:
                logger.debug("Tesseract failed for config %s: %s", cfg, e)
                continue

    if not candidates:
        return ""

    # Heuristic score: presence of ingredient-like keywords + length
    keywords = ['ingredients', 'water', 'glycerin', 'acid', 'oil', 'extract', 'sodium', 'paraben', 'sulfate']
    def score_text(t: str) -> float:
        t_low = t.lower()
        hits = sum(1 for k in keywords if k in t_low)
        words = len(t_low.split())
        return hits * 2.0 + min(words / 20.0, 5.0)

    best = max(candidates, key=lambda c: score_text(c[1]))
    best_text = fix_hyphenated_words(best[1])
    return best_text

# -------------------------
# Parsing ingredients
# -------------------------
def parse_ingredients_from_text(text: str) -> List[str]:
    if not text:
        return []

    t = fix_hyphenated_words(text)
    t = t.replace('\r', ' ').replace('\n', ' ')
    t = re.sub(r'\s+', ' ', t)

    # Try to find ingredients section
    m = re.search(r'(ingredients?|contains?)\s*[:\-]\s*(.+?)(?:\.\s|$)', t, re.I | re.S)
    ing_section = m.group(2) if m else t

    # Remove warnings/directions
    ing_section = re.split(r'(directions?|warnings?|caution|for external use|store)', ing_section, flags=re.I)[0]

    parts = re.split(r'[,\;•\·]|\sand\s', ing_section)
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

# -------------------------
# Classification DB
# -------------------------
class EnhancedIngredientDatabase:
    def __init__(self, mapping=None):
        mapping = mapping or COMPREHENSIVE_INGREDIENTS
        self.harmful = set(mapping.get('harmful', []))
        self.allergen = set(mapping.get('allergen', []))
        self.safe = set(mapping.get('safe', []))
        self.all_known = self.harmful | self.allergen | self.safe

    def classify_ingredient(self, ingredient: str) -> Tuple[str, str, float]:
        ing = normalize_ingredient(ingredient)
        if ing in self.harmful:
            return 'harmful', 'Known harmful', 1.0
        if ing in self.allergen:
            return 'allergen', 'Known allergen', 1.0
        if ing in self.safe:
            return 'safe', 'Known safe', 1.0

        # fuzzy match
        best = None
        best_score = 0.0
        for k in self.all_known:
            score = similar(ing, normalize_ingredient(k))
            if score > best_score:
                best_score = score
                best = k
        if best_score >= 0.82:
            if best in self.harmful:
                return 'harmful', f'Matched to {best}', best_score
            if best in self.allergen:
                return 'allergen', f'Matched to {best}', best_score
            if best in self.safe:
                return 'safe', f'Matched to {best}', best_score

        # keywords
        if any(k in ing for k in ['paraben', 'phthalate', 'sulfate', 'benzene', 'toluene']):
            return 'harmful', 'Contains harmful keyword', 0.65
        if any(k in ing for k in ['fragrance', 'parfum', 'limonene', 'linalool']):
            return 'allergen', 'Contains allergen keyword', 0.65

        # default
        return 'safe', 'Safe', 0.45

# -------------------------
# Web helpers: session + search + scrape
# -------------------------
def requests_session_with_retries(total_retries: int = 3, backoff_factor: float = 0.3):
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor,
                    status_forcelist=[429, 500, 502, 503, 504])
    s.mount('http://', HTTPAdapter(max_retries=retries))
    s.mount('https://', HTTPAdapter(max_retries=retries))
    s.headers.update({'User-Agent': '4u-Ingredient-Analyzer/1.0 (+contact@example.com)'})
    return s

@st.cache_data(ttl=3600)
def ddg_search_html(query: str, session: Optional[requests.Session] = None, timeout: int = 8) -> List[Dict]:
    """
    Use DuckDuckGo HTML interface to search and return list of results (title, href, snippet).
    Note: this is a lightweight approach for demonstration — not a guaranteed API.
    """
    session = session or requests_session_with_retries()
    url = "https://html.duckduckgo.com/html/"
    try:
        res = session.get(url, params={'q': query}, timeout=timeout)
        if res.status_code != 200:
            return []
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        for a in soup.select("a.result__a"):
            href = a.get('href')
            title = a.get_text(strip=True)
            snippet_node = a.find_parent().select_one(".result__snippet")
            snippet = snippet_node.get_text(strip=True) if snippet_node else ""
            if href:
                results.append({'title': title, 'href': href, 'snippet': snippet})
        # fallback: look for .result (old structure)
        if not results:
            for r in soup.select(".result"):
                tag = r.select_one("a")
                if not tag: continue
                href = tag.get('href')
                title = tag.get_text(strip=True)
                snippet_node = r.select_one(".result__snippet")
                snippet = snippet_node.get_text(strip=True) if snippet_node else ""
                results.append({'title': title, 'href': href, 'snippet': snippet})
        return results
    except Exception as e:
        logger.debug("ddg_search_html error: %s", e)
        return []

@st.cache_data(ttl=3600)
def fetch_page_text(url: str, session: Optional[requests.Session] = None, timeout: int = 8) -> str:
    session = session or requests_session_with_retries()
    try:
        res = session.get(url, timeout=timeout)
        if res.status_code != 200:
            return ""
        return res.text
    except Exception as e:
        logger.debug("fetch_page_text error: %s", e)
        return ""

def extract_ingredient_section_from_html(html: str) -> str:
    """
    Heuristic: find blocks that look like 'Ingredients:' or lists of INCI names.
    Returns the raw extracted text (may contain markup).
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    # Try to find nodes containing the word 'Ingredients' (common)
    candidates = []
    for tag in soup.find_all(text=re.compile(r'Ingredients?', re.I)):
        parent = tag.parent
        # Check parent or next sibling text
        texts = []
        # parent text
        texts.append(parent.get_text(separator=" ", strip=True))
        # next siblings
        sib = parent.find_next_sibling()
        for _ in range(3):
            if sib is None:
                break
            texts.append(sib.get_text(separator=" ", strip=True))
            sib = sib.find_next_sibling()
        candidates.extend(texts)

    # Fallback: look for meta tags or big lists
    if not candidates:
        for ul in soup.find_all(['ul', 'ol']):
            txt = ul.get_text(separator=" ", strip=True)
            if len(txt.split()) > 5:
                candidates.append(txt)

    # Pick the longest candidate containing "ingredient" or comma separated tokens
    best = ""
    for c in candidates:
        if 'ingredient' in c.lower() or ',' in c:
            if len(c) > len(best):
                best = c
    # final cleanup
    return best

@st.cache_data(ttl=3600)
def fetch_ingredient_summary_from_web(ingredient: str, session: Optional[requests.Session] = None) -> Optional[str]:
    """
    Best-effort short summary for a single ingredient. Uses DuckDuckGo search and scrapes
    first relevant snippet or paragraph from the top result.
    """
    if not ingredient:
        return None
    session = session or requests_session_with_retries()
    query = f"{ingredient} cosmetic ingredient safety"
    results = ddg_search_html(query, session=session)
    if not results:
        return None
    # Try first few results to get a snippet
    for r in results[:5]:
        href = r.get('href')
        snippet = r.get('snippet') or ""
        if snippet and len(snippet) > 40:
            return snippet[:350]
        # fetch page and try to extract meaningful paragraph
        html = fetch_page_text(href, session=session)
        if html:
            # simple heuristic: first paragraph with ingredient word
            soup = BeautifulSoup(html, "html.parser")
            for p in soup.find_all('p'):
                txt = p.get_text(" ", strip=True)
                if len(txt) > 80 and ingredient.lower().split()[0] in txt.lower():
                    return txt[:350]
            # fallback: meta description
            meta = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
            if meta and meta.get('content'):
                return meta.get('content')[:350]
    return None

# -------------------------
# Product search & scrape
# -------------------------
def search_product_and_extract_ingredients(product_query: str, session: Optional[requests.Session] = None) -> List[Dict]:
    """
    Best-effort: search product by name, visit top results and attempt to extract ingredient sections.
    Returns list of dicts: [{'title':..., 'url':..., 'ingredients_text':...}, ...]
    """
    session = session or requests_session_with_retries()
    results = ddg_search_html(product_query + " ingredients", session=session)
    outputs = []
    for res in results[:6]:  # try top 6 results
        url = res.get('href')
        title = res.get('title')
        if not url:
            continue
        html = fetch_page_text(url, session=session)
        if not html:
            continue
        ing_text = extract_ingredient_section_from_html(html)
        outputs.append({'title': title, 'url': url, 'ingredients_text': ing_text})
    return outputs

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="4u - Ingredient Analyzer Pro (Web Lookup)", page_icon="🧴", layout="wide")
    st.title("🧴 4u - Ingredient Analyzer Pro")
    st.markdown("Extract ingredients from images, classify them, and optionally fetch web-sourced notes or scrape product pages for ingredient lists.")

    # DB init
    if 'db' not in st.session_state:
        st.session_state.db = EnhancedIngredientDatabase(COMPREHENSIVE_INGREDIENTS)
    db = st.session_state.db

    # Sidebar options
    with st.sidebar:
        st.header("Options")
        use_web = st.checkbox("Enable live web lookup (ingredient summaries)", value=False)
        allow_product_search = st.checkbox("Enable product search & scrape", value=False)
        st.markdown("---")
        st.caption("Network calls are best-effort. Some sites may block scraping; results vary.")

    # Left: image uploader and OCR
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded = st.file_uploader("Upload product ingredient image", type=['png', 'jpg', 'jpeg'])
        manual_ing = st.text_area("Manual ingredient list (optional, comma-separated)", height=80)
        product_search_query = st.text_input("Product search (name) or product page URL", "")
        if allow_product_search and st.button("🔎 Search product & scrape ingredients"):
            if not product_search_query.strip():
                st.warning("Enter a product name or a product URL to search.")
            else:
                with st.spinner("Searching for product pages and trying to extract ingredient sections..."):
                    session = requests_session_with_retries()
                    # If the user pasted a URL, try it directly first
                    outputs = []
                    if re.match(r'^https?://', product_search_query.strip()):
                        html = fetch_page_text(product_search_query.strip(), session=session)
                        ing_text = extract_ingredient_section_from_html(html)
                        outputs.append({'title': product_search_query.strip(), 'url': product_search_query.strip(), 'ingredients_text': ing_text})
                    # also run a search
                    outputs += search_product_and_extract_ingredients(product_search_query.strip(), session=session)
                    if not outputs:
                        st.warning("No candidate pages found or scraping failed.")
                    else:
                        for o in outputs:
                            st.markdown(f"**{o['title']}** — {o['url']}")
                            if o['ingredients_text']:
                                with st.expander("Extracted ingredients (raw)"):
                                    st.text_area("Ingredients text", o['ingredients_text'], height=160)
                                # Parse and show classification quick view
                                parsed = parse_ingredients_from_text(o['ingredients_text'])
                                if parsed:
                                    st.write(f"Detected {len(parsed)} ingredient(s).")
                                    # Show a small table
                                    rows = []
                                    for ing in parsed[:40]:
                                        cat, reason, conf = db.classify_ingredient(ing)
                                        rows.append({'ingredient': ing, 'category': cat})
                                    st.table(pd.DataFrame(rows))
                                else:
                                    st.info("Could not parse individual ingredients from the extracted section. Try the raw text above.")
                            else:
                                st.info("No ingredient-like block found on this page. View the page manually or provide a product URL.")
        st.markdown("---")

    with col2:
        st.markdown("### OCR → Parse → Classify")
        if not uploaded and not manual_ing:
            st.info("Upload an image or paste a manual ingredient list (comma-separated) to analyze.")
        if uploaded:
            try:
                img = Image.open(uploaded).convert('RGB')
                st.image(img, caption="Uploaded image", use_column_width=True)
            except Exception as e:
                st.error(f"Could not open uploaded image: {e}")
                return

        # Action buttons
        if st.button("🔍 Extract & Analyze (Image)"):
            ocr_text = ""
            if uploaded:
                with st.spinner("Running OCR..."):
                    ocr_text = extract_text_ingredients_region(img)
            else:
                st.warning("No image uploaded; using manual ingredients if provided.")

            if not ocr_text and not manual_ing:
                st.error("❌ OCR could not detect text in the image. Try a clearer image or paste the ingredients manually.")
            else:
                # choose source text (manual overrides OCR when provided)
                source_text = manual_ing.strip() if manual_ing.strip() else ocr_text
                if ocr_text and not manual_ing:
                    with st.expander("🔍 Raw OCR Output (editable)"):
                        edited = st.text_area("Edit OCR text and re-parse if needed", value=ocr_text, height=220)
                        # Use edited text if user changed it
                        source_text = edited if edited.strip() else ocr_text

                parsed = parse_ingredients_from_text(source_text)
                if not parsed:
                    st.warning("⚠️ No ingredients parsed — you can edit the OCR output above or paste a manual comma-separated list.")
                else:
                    st.success(f"✅ Found {len(parsed)} ingredient(s).")
                    # Build results with optional web lookup
                    session = requests_session_with_retries() if use_web else None
                    rows = []
                    prog = st.progress(0)
                    status = st.empty()
                    for i, ing in enumerate(parsed, start=1):
                        status.text(f"Analyzing: {ing}")
                        cat, reason, conf = db.classify_ingredient(ing)
                        web_note = None
                        if use_web and session:
                            try:
                                web_note = fetch_ingredient_summary_from_web(ing, session=session)
                            except Exception as e:
                                logger.debug("Web lookup failed for %s: %s", ing, e)
                        rows.append({'ingredient': ing, 'category': cat, 'reason': reason, 'confidence': f"{int(conf*100)}%", 'web_note': web_note or ""})
                        prog.progress(int(i/len(parsed)*100))
                        time.sleep(0.02)
                    prog.empty()
                    status.empty()

                    df = pd.DataFrame(rows)
                    st.markdown("#### Analysis Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    counts = df['category'].value_counts().to_dict()
                    c1.metric("🔴 Harmful", counts.get('harmful', 0))
                    c2.metric("🟡 Allergens", counts.get('allergen', 0))
                    c3.metric("🟢 Safe", counts.get('safe', 0))
                    c4.metric("ℹ️ Total", len(parsed))

                    st.markdown("---")
                    st.subheader("Detailed Report")
                    # show table without web_note by default
                    st.dataframe(df[['ingredient', 'category', 'reason', 'confidence']], use_container_width=True)

                    if use_web:
                        with st.expander("🔎 Web notes per ingredient (best-effort)"):
                            for r in rows:
                                st.markdown(f"**{r['ingredient']}** — *{r['category']}* — {r['confidence']}")
                                if r['web_note']:
                                    st.write(r['web_note'])
                                else:
                                    st.write("_No web note found or fetch failed._")
                                st.markdown("---")

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV Report", data=csv, file_name=f"ingredient_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime='text/csv')

    st.markdown("---")
    st.caption("⚠️ This tool is informational. For medical advice, product safety questions, or allergy concerns, consult a licensed dermatologist.")

if __name__ == "__main__":
    main()


