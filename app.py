# /mnt/data/app.py
# 4u - Ingredient Analyzer Pro (Fast Scrape version)
# - Fast web lookup mode (DuckDuckGo HTML search + lightweight page scraping)
# - High-confidence classification only (no assumed safe)
# - Unknown ingredients -> 'review'
# - Always-on web verification per ingredient
# - Tesseract auto-detection for Streamlit Cloud
#
# Original uploaded path reference: /mnt/data/app.py

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

# Streamlit page config
st.set_page_config(page_title="4u - Ingredient Analyzer Pro", page_icon="🧴", layout="wide")

# Tesseract auto-detect (use env var TESSERACT_CMD or 'tesseract' in PATH)
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")

# ----------------------------
# Core Ingredient DB (original + some safe additions)
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
# Fast web lookup helpers (DuckDuckGo HTML + lightweight page parsing)
# ----------------------------
def requests_session_with_retries(total_retries: int = 3, backoff_factor: float = 0.3):
    s = requests.Session()
    retries = Retry(total=total_retries, backoff_factor=backoff_factor, status_forcelist=[429,500,502,503,504])
    s.mount('http://', HTTPAdapter(max_retries=retries))
    s.mount('https://', HTTPAdapter(max_retries=retries))
    s.headers.update({'User-Agent': '4u-Ingredient-Analyzer/1.0 (+contact@example.com)'})
    return s

@st.cache_data(ttl=3600)
def ddg_search_html(query: str, session: Optional[requests.Session] = None, timeout: int = 8):
    """Return top DuckDuckGo HTML search results (title, href, snippet) - Light and robust"""
    session = session or requests_session_with_retries()
    try:
        res = session.get("https://html.duckduckgo.com/html/", params={'q': query}, timeout=timeout)
        if res.status_code != 200:
            return []
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        # Newer HTML uses 'result__a' links
        for a in soup.select("a.result__a"):
            href = a.get('href')
            title = a.get_text(strip=True)
            snippet_node = a.find_parent()
            snippet = ""
            # attempt to locate snippet
            sn = snippet_node.select_one(".result__snippet") if snippet_node else None
            if sn:
                snippet = sn.get_text(strip=True)
            results.append({'title': title, 'href': href, 'snippet': snippet})
        # fallback
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
def fetch_page_text(url: str, session: Optional[requests.Session] = None, timeout: int = 8):
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
    """Heuristic extraction of an ingredient block from a product/brand page"""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    # Find text nodes that contain "Ingredients"
    for tag in soup.find_all(text=re.compile(r'Ingredients?', re.I)):
        parent = tag.parent
        if parent:
            text = parent.get_text(separator=" ", strip=True)
            candidates.append(text)
            # also next siblings
            sib = parent.find_next_sibling()
            for _ in range(3):
                if not sib:
                    break
                candidates.append(sib.get_text(separator=" ", strip=True))
                sib = sib.find_next_sibling()

    # fallback: look for long <ul>/<ol> lists
    if not candidates:
        for ul in soup.find_all(['ul', 'ol']):
            txt = ul.get_text(separator=" ", strip=True)
            if len(txt.split()) > 6:
                candidates.append(txt)

    # pick the best candidate containing commas or 'Ingredients'
    best = ""
    for c in candidates:
        if 'ingredient' in c.lower() or ',' in c:
            if len(c) > len(best):
                best = c
    return best

@st.cache_data(ttl=3600)
def fetch_ingredient_summary_from_web(ingredient: str, session: Optional[requests.Session] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Fast-scrape: search for "<ingredient> cosmetic" and return (summary_text, source)
    Source is first result domain or 'DDG snippet'
    """
    if not ingredient:
        return None, None
    session = session or requests_session_with_retries()
    query = f"{ingredient} cosmetic ingredient safety"
    results = ddg_search_html(query, session=session)
    if not results:
        return None, None
    # prefer results with snippet
    for r in results[:6]:
        snippet = r.get('snippet') or ""
        href = r.get('href') or ""
        title = r.get('title') or ""
        if snippet and len(snippet) > 40:
            domain = href.split('/')[2] if href.startswith('http') else 'DDG'
            return snippet[:400], domain
        # fetch page text fallback
        html = fetch_page_text(href, session=session)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            # choose first paragraph containing the ingredient or a safety keyword
            for p in soup.find_all('p'):
                txt = p.get_text(" ", strip=True)
                if len(txt) > 80 and ingredient.split()[0] in txt.lower():
                    domain = href.split('/')[2] if href.startswith('http') else 'DDG'
                    return txt[:400], domain
            # meta description fallback
            meta = soup.find('meta', attrs={'name':'description'}) or soup.find('meta', attrs={'property':'og:description'})
            if meta and meta.get('content'):
                domain = href.split('/')[2] if href.startswith('http') else 'DDG'
                return meta.get('content')[:400], domain
    # last resort: return first result title
    first = results[0]
    return first.get('snippet')[:400] if first.get('snippet') else first.get('title'), first.get('href').split('/')[2] if first.get('href') else 'DDG'

# ----------------------------
# Classification with strict confidence rules
# ----------------------------
class EnhancedIngredientDatabaseFast:
    def __init__(self, mapping: Dict[str, set]):
        self.harmful = set(mapping.get('harmful', []))
        self.allergen = set(mapping.get('allergen', []))
        self.safe = set(mapping.get('safe', []))
        self.all_known = self.harmful | self.allergen | self.safe

    def classify_ingredient(self, ingredient: str, session: Optional[requests.Session] = None) -> Tuple[str, str, float, str]:
        """
        Returns (category, reason, score, source)
        category in ['harmful','allergen','safe','review']
        source: 'Local DB', 'Fuzzy', 'Keyword', 'Web'
        """
        ing_norm = normalize_ingredient(ingredient)
        session = session or requests_session_with_retries()

        # 1) Exact local DB match
        if ing_norm in self.harmful:
            return 'harmful', 'Exact local harmful match', 1.0, 'Local DB'
        if ing_norm in self.allergen:
            return 'allergen', 'Exact local allergen match', 1.0, 'Local DB'
        if ing_norm in self.safe:
            # Even safe exact match is considered high-confidence
            return 'safe', 'Exact local safe match', 1.0, 'Local DB'

        # 2) Strong fuzzy match against known DB (>= 0.90)
        best = None
        best_score = 0.0
        for k in self.all_known:
            score = similar(ing_norm, normalize_ingredient(k))
            if score > best_score:
                best_score = score
                best = k
        if best_score >= 0.90:
            if best in self.harmful:
                return 'harmful', f'High-confidence fuzzy match to {best}', best_score, 'Fuzzy'
            if best in self.allergen:
                return 'allergen', f'High-confidence fuzzy match to {best}', best_score, 'Fuzzy'
            if best in self.safe:
                return 'safe', f'High-confidence fuzzy match to {best}', best_score, 'Fuzzy'

        # 3) Keyword-based strong indicators (harmful/allergen)
        harmful_keys = ['paraben', 'phthalate', 'sulfate', 'benzene', 'toluene', 'mercury', 'lead']
        allergen_keys = ['fragrance', 'parfum', 'limonene', 'linalool', 'benzyl', 'citral', 'geraniol']
        lower = ing_norm.lower()
        if any(k in lower for k in harmful_keys):
            return 'harmful', 'Keyword indicator (harmful)', 0.85, 'Keyword'
        if any(k in lower for k in allergen_keys):
            return 'allergen', 'Keyword indicator (allergen)', 0.85, 'Keyword'

        # 4) FAST web verification (DuckDuckGo snippet/page parsing)
        try:
            web_summary, web_source = fetch_ingredient_summary_from_web(ingredient, session=session)
            if web_summary:
                web_lower = web_summary.lower()
                # If web content mentions hazard keywords, mark harmful
                if any(k in web_lower for k in harmful_keys):
                    return 'harmful', f'Web-verified hazardous keywords ({web_source})', 0.80, f'Web:{web_source}'
                # If web content has allergen keywords
                if any(k in web_lower for k in allergen_keys):
                    return 'allergen', f'Web-verified allergen keywords ({web_source})', 0.78, f'Web:{web_source}'
                # If web content strongly matches ingredient name, accept as safe-ish with medium-high confidence
                # (Only accept as safe if web explicitly references safety/usage in neutral/positive terms)
                if len(web_summary) > 80 and ing_norm.split()[0] in web_lower:
                    return 'safe', f'Web-verified mention ({web_source})', 0.75, f'Web:{web_source}'
        except Exception as e:
            logger.debug("Web verification failed for %s: %s", ingredient, e)

        # 5) No high-confidence result -> mark as review
        return 'review', 'Insufficient data — requires manual review', 0.0, 'Review'

# ----------------------------
# Product search & scraping (fast)
# ----------------------------
def search_product_and_extract_ingredients(product_query: str, session: Optional[requests.Session] = None) -> List[Dict]:
    session = session or requests_session_with_retries()
    results = ddg_search_html(product_query + " ingredients", session=session)
    outputs = []
    for r in results[:6]:
        href = r.get('href') or ""
        title = r.get('title') or ""
        html = fetch_page_text(href, session=session)
        ing_text = extract_ingredient_section_from_html(html) if html else ""
        outputs.append({'title': title, 'url': href, 'ingredients_text': ing_text})
    return outputs

# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.title("🧴 4u - Skin Product Analyzer (Fast Web Lookup)")
    st.markdown("Extract ingredient lists from images, classify using a high-confidence policy, and verify with fast web lookups.")

    # Initialize DB
    if 'db_fast' not in st.session_state:
        st.session_state.db_fast = EnhancedIngredientDatabaseFast(COMPREHENSIVE_INGREDIENTS)

    db = st.session_state.db_fast

    # Sidebar
    with st.sidebar:
        st.header("Options & Info")
        st.markdown("- Web lookup is **always on** (Fast mode).")
        st.markdown("- Unknown ingredients are marked `review` and require manual verification.")
        st.markdown("- For best OCR: photograph the ingredients panel flat and well-lit.")
        st.markdown("---")
        st.caption("Fast web lookup uses DuckDuckGo HTML search + lightweight page parsing. Results are best-effort and cached for 1 hour.")

    st.markdown("---")

    # Upload / manual input / product search
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded = st.file_uploader("📷 Upload product ingredient image", type=['png','jpg','jpeg'])
        manual_ingredients_text = st.text_area("Or paste manual ingredient text (comma separated)", height=120)
        product_search = st.text_input("Product search (name) or product page URL", "")

        if st.button("🔎 Search Product & Try Extracting Ingredients"):
            if not product_search.strip():
                st.warning("Enter a product name or product URL to search.")
            else:
                with st.spinner("Searching product pages (fast)..."):
                    session = requests_session_with_retries()
                    outputs = []
                    if re.match(r'^https?://', product_search.strip()):
                        html = fetch_page_text(product_search.strip(), session=session)
                        ing_text = extract_ingredient_section_from_html(html)
                        outputs.append({'title': product_search.strip(), 'url': product_search.strip(), 'ingredients_text': ing_text})
                    outputs += search_product_and_extract_ingredients(product_search.strip(), session=session)
                    if not outputs:
                        st.warning("No candidate pages found or scraping failed.")
                    else:
                        for o in outputs:
                            st.markdown(f"**{o['title']}** — {o['url']}")
                            if o['ingredients_text']:
                                with st.expander("Extracted ingredients (raw)"):
                                    st.text_area("Ingredients text", o['ingredients_text'], height=160)
                                # quick parse
                                parsed = parse_ingredients_from_text(o['ingredients_text'])
                                if parsed:
                                    st.write(f"Detected {len(parsed)} ingredient(s).")
                                    rows = []
                                    for ing in parsed[:40]:
                                        cat, reason, score, src = db.classify_ingredient(ing)
                                        rows.append({'ingredient': ing, 'category': cat, 'reason': reason, 'score': f"{round(score*100,2)}%", 'source': src})
                                    st.table(pd.DataFrame(rows))
                                else:
                                    st.info("Could not parse individual ingredients from the extracted section.")
                            else:
                                st.info("No ingredient-like block found on this page. Provide a product URL or try a different search.")

    with col2:
        st.markdown("### OCR → Parse → High-Confidence Classify (with web verification)")
        if uploaded:
            try:
                img = Image.open(uploaded).convert('RGB')
                st.image(img, use_column_width=True)
            except Exception as e:
                st.error(f"Could not open image: {e}")
                return

        if st.button("🔍 Extract & Analyze"):
            session = requests_session_with_retries()
            ocr_text = ""
            if uploaded:
                with st.spinner("Running OCR..."):
                    ocr_text = extract_text_ingredients_region(img)

            # If user provided manual text, prefer that (allow editing)
            source_text = manual_ingredients_text.strip() if manual_ingredients_text.strip() else ocr_text
            if not source_text or not source_text.strip():
                st.error("❌ No OCR text or manual ingredients provided. Try a clearer image or paste the ingredient list.")
                return

            with st.expander("🔍 Raw OCR / Source Text (editable)"):
                edited = st.text_area("Edit OCR text (then re-run Extract & Analyze)", value=source_text, height=220)
                if edited.strip():
                    source_text = edited.strip()

            parsed = parse_ingredients_from_text(source_text)
            if not parsed:
                st.warning("⚠️ No ingredients parsed. Edit the text above or paste a comma-separated list.")
                return

            st.success(f"✅ Found {len(parsed)} ingredient(s). Verifying each with fast web lookup...")

            results = []
            prog = st.progress(0)
            status = st.empty()
            for i, ing in enumerate(parsed, start=1):
                status.text(f"Analyzing: {ing}")
                cat, reason, score, src = db.classify_ingredient(ing, session=session)
                # Always fetch web summary to show to user (fast)
                web_summary, web_source = fetch_ingredient_summary_from_web(ing, session=session)
                results.append({
                    'ingredient': ing,
                    'category': cat,
                    'reason': reason,
                    'accuracy_score': f"{round(score*100,2)}%",
                    'source': src,
                    'web_summary': web_summary or "",
                    'web_domain': web_source or ""
                })
                prog.progress(int(i / len(parsed) * 100))
                time.sleep(0.02)

            prog.empty()
            status.empty()

            df = pd.DataFrame(results)

            # Summary counts (include 'review')
            counts = df['category'].value_counts().to_dict()
            st.markdown('---')
            st.subheader('📋 Analysis Summary')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('🔴 Harmful', counts.get('harmful', 0))
            c2.metric('🟡 Allergens', counts.get('allergen', 0))
            c3.metric('🟢 Safe', counts.get('safe', 0))
            c4.metric('⚠️ Needs Review', counts.get('review', 0))

            st.markdown('---')
            st.subheader('🔬 Detailed Report')
            st.dataframe(df[['ingredient', 'category', 'reason', 'accuracy_score', 'source', 'web_domain']], use_container_width=True)

            # Show web notes per ingredient
            with st.expander("🔎 Web-sourced notes (fast)"):
                for r in results:
                    st.markdown(f"**{r['ingredient']}** — *{r['category']}* — {r['accuracy_score']} — Source: {r['source']}")
                    if r['web_summary']:
                        st.write(r['web_summary'])
                        st.caption(f"Source domain: {r['web_domain']}")
                    else:
                        st.write("_No quick web note found._")
                    st.markdown("---")

            # Tabs for categories
            tabs = st.tabs(['🔴 Harmful','🟡 Allergens','🟢 Safe','⚠️ Review'])
            with tabs[0]:
                harms = df[df['category']=='harmful']
                if harms.empty:
                    st.success('✅ No harmful ingredients detected')
                else:
                    st.warning(f'⚠️ Found {len(harms)} harmful ingredient(s)')
                    st.table(harms[['ingredient','reason','accuracy_score','source','web_domain']])

            with tabs[1]:
                alls = df[df['category']=='allergen']
                if alls.empty:
                    st.success('✅ No allergens detected')
                else:
                    st.warning(f'⚠️ Found {len(alls)} potential allergen(s)')
                    st.table(alls[['ingredient','reason','accuracy_score','source','web_domain']])

            with tabs[2]:
                safes = df[df['category']=='safe']
                st.success(f'✅ Found {len(safes)} safe ingredient(s)')
                st.table(safes[['ingredient','reason','accuracy_score','source','web_domain']])

            with tabs[3]:
                reviews = df[df['category']=='review']
                if reviews.empty:
                    st.success('✅ No items need review')
                else:
                    st.info(f'{len(reviews)} ingredient(s) need manual review')
                    st.table(reviews[['ingredient','reason','accuracy_score','source','web_domain']])

            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                '📥 Download Full Report (CSV)',
                data=csv,
                file_name=f'ingredient_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )

            st.markdown('---')
            st.caption("⚠️ This tool is informational. For medical advice, product safety, or allergies, consult a licensed dermatologist.")

    st.markdown("---")
    st.caption("💡 Tip: For best OCR results, photograph the full ingredients panel flat, in good light, and avoid reflections or angles.")

if __name__ == '__main__':
    main()
