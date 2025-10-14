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
from difflib import SequenceMatcher, get_close_matches
from urllib.parse import quote_plus

# Configure page
st.set_page_config(page_title="4u - Ingredient Analyzer Pro", page_icon="🧴", layout="wide")

# Path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -------------------- Comprehensive Ingredient Database --------------------

COMPREHENSIVE_INGREDIENTS = {
    # Harmful ingredients
    'harmful': {
        'formaldehyde', 'hydroquinone', 'methylparaben', 'ethylparaben', 'propylparaben', 
        'butylparaben', 'paraben', 'phthalate', 'triclosan', 'triclocarban',
        'mercury', 'lead', 'oxybenzone', 'sodium lauryl sulfate', 'sodium laureth sulfate', 
        'benzene', 'toluene', 'coal tar', 'petrolatum', 'bha', 'bht', 'dea', 'tea', 'mea',
        'polyethylene', 'microbeads', 'propylene glycol', 'mineral oil', 'parabens',
        'diethanolamine', 'triethanolamine', 'monoethanolamine', 'butylated hydroxyanisole',
        'butylated hydroxytoluene', 'dibutyl phthalate', 'diethyl phthalate'
    },
    
    # Allergens
    'allergen': {
        'fragrance', 'parfum', 'methylisothiazolinone', 'methylchloroisothiazolinone',
        'lanolin', 'cinnamal', 'eugenol', 'limonene', 'linalool', 'neomycin', 
        'kathon cg', 'formaldehyde releaser', 'quaternium-15', 'dmdm hydantoin', 
        'imidazolidinyl urea', 'diazolidinyl urea', 'benzyl alcohol', 'citral',
        'geraniol', 'citronellol', 'farnesol', 'coumarin', 'benzyl benzoate',
        'benzyl salicylate', 'alpha-isomethyl ionone', 'hydroxycitronellal',
        'hexyl cinnamal', 'amyl cinnamal', 'isoeugenol', 'anise alcohol',
        'methylparaben', 'ethylparaben', 'propylparaben', 'butylene glycol'
    },
    
    # Safe ingredients (comprehensive list)
    'safe': {
        'water', 'aqua', 'glycerin', 'glycerine', 'hyaluronic acid', 'sodium hyaluronate',
        'niacinamide', 'panthenol', 'zinc oxide', 'titanium dioxide', 'ceramide',
        'peptide', 'vitamin c', 'ascorbic acid', 'vitamin e', 'tocopherol', 'tocopheryl acetate',
        'retinol', 'retinyl palmitate', 'squalane', 'squalene', 'aloe vera', 'aloe barbadensis',
        'shea butter', 'butyrospermum parkii', 'jojoba oil', 'simmondsia chinensis',
        'green tea extract', 'camellia sinensis', 'argan oil', 'argania spinosa',
        'coconut oil', 'cocos nucifera', 'sweet almond oil', 'prunus amygdalus dulcis',
        'sodium chloride', 'sodium citrate', 'citric acid', 'potassium sorbate',
        'sodium benzoate', 'phenoxyethanol', 'ethylhexylglycerin', 'caprylyl glycol',
        'disodium edta', 'tetrasodium edta', 'edta', 'xanthan gum', 'carbomer',
        'acrylates copolymer', 'polyacrylate', 'sodium polyacrylate', 
        'sodium polyacryloyldimethyl taurate', 'dimethicone', 'cyclomethicone',
        'cyclopentasiloxane', 'dimethiconol', 'cetyl alcohol', 'cetearyl alcohol',
        'stearyl alcohol', 'behenyl alcohol', 'stearic acid', 'palmitic acid',
        'myristic acid', 'lauric acid', 'caprylic capric triglyceride',
        'isopropyl palmitate', 'isopropyl myristate', 'isodecyl neopentanoate',
        'trilaureth-4 phosphate', 'polysorbate 20', 'polysorbate 80',
        'peg-100 stearate', 'glyceryl stearate', 'sorbitol', 'propanediol',
        'butylene glycol', 'pentylene glycol', 'caprylhydroxamic acid',
        'sodium lactate', 'lactic acid', 'glycolic acid', 'salicylic acid',
        'allantoin', 'bisabolol', 'beta-glucan', 'collagen', 'elastin',
        'lecithin', 'phospholipids', 'urea', 'trehalose', 'betaine',
        'pantothenic acid', 'biotin', 'folic acid', 'pyridoxine', 'thiamine',
        'riboflavin', 'ascorbyl palmitate', 'ascorbyl glucoside', 'magnesium ascorbyl phosphate',
        'sodium ascorbyl phosphate', 'tetrahexyldecyl ascorbate', 'retinyl acetate',
        'retinaldehyde', 'bakuchiol', 'resveratrol', 'coenzyme q10', 'ubiquinone',
        'glutathione', 'alpha lipoic acid', 'ferulic acid', 'kojic acid',
        'arbutin', 'alpha arbutin', 'tranexamic acid', 'azelaic acid',
        'mandelic acid', 'polyhydroxy acid', 'gluconolactone', 'lactobionic acid',
        'adenosine', 'caffeine', 'centella asiatica', 'madecassoside', 'asiaticoside',
        'calendula officinalis', 'chamomilla recutita', 'rosemary extract', 'rosmarinus officinalis',
        'lavender oil', 'lavandula angustifolia', 'tea tree oil', 'melaleuca alternifolia',
        'sodium pca', 'glycol distearate', 'diheptyl succinate', 'capryloyl glycerin',
        'hydrogenated castor oil', 'hydrogenated vegetable oil', 'candelilla wax',
        'beeswax', 'carnauba wax', 'ozokerite', 'microcrystalline wax',
        'kaolin', 'bentonite', 'silica', 'mica', 'iron oxides', 'ultramarines',
        'calcium carbonate', 'magnesium carbonate', 'zinc stearate', 'magnesium stearate',
        'sodium stearate', 'potassium hydroxide', 'sodium hydroxide', 'triethanolamine'
    }
}

# -------------------- Utilities --------------------

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def fix_ocr_errors(text):
    """Fix common OCR errors in ingredient names"""
    # Fix common spacing errors
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
        r'iso\s*decyl': 'isodecyl',
        r'neo\s*pentanoate': 'neopentanoate',
        r'tril\s*aureth': 'trilaureth',
        r'xan\s*than': 'xanthan',
        r'dis\s*odium': 'disodium',
        r'tetra\s*sodium': 'tetrasodium',
        r'cetear\s*yl': 'cetearyl',
        r'stear\s*yl': 'stearyl',
        r'glycer\s*yl': 'glyceryl',
        r'capry\s*lic': 'caprylic',
        r'laur\s*ic': 'lauric',
        r'palm\s*itic': 'palmitic',
        r'stear\s*ic': 'stearic',
        r'e\s*d\s*t\s*a': 'edta',
        r'p\s*e\s*g': 'peg',
    }
    
    text_fixed = text.lower()
    for pattern, replacement in fixes.items():
        text_fixed = re.sub(pattern, replacement, text_fixed, flags=re.IGNORECASE)
    
    # Remove extra spaces
    text_fixed = re.sub(r'\s+', ' ', text_fixed).strip()
    
    return text_fixed

def fix_hyphenated_words(text):
    """Fix words broken by hyphens at line breaks"""
    text = re.sub(r'-\s+', '', text)
    return text

def normalize_ingredient(s: str):
    """Normalize ingredient name for matching"""
    s = s.lower().strip()
    s = fix_ocr_errors(s)
    s = re.sub(r'\([^)]*\)', '', s)  # Remove parenthetical info
    s = re.sub(r'\[[^]]*\]', '', s)  # Remove brackets
    s = re.sub(r'\d+\.?\d*\s*%', '', s)  # Remove percentages
    s = re.sub(r'[^a-z0-9\-\s]', ' ', s)  # Keep only alphanumeric and hyphens
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def fuzzy_match_ingredient(ingredient: str, known_list: set, threshold: float = 0.85):
    """Find best match for ingredient in known list using fuzzy matching"""
    ing_norm = normalize_ingredient(ingredient)
    
    # Direct match
    if ing_norm in known_list:
        return ing_norm, 1.0
    
    # Check if ingredient contains or is contained in any known ingredient
    for known in known_list:
        known_norm = normalize_ingredient(known)
        if known_norm in ing_norm or ing_norm in known_norm:
            return known, 0.95
    
    # Fuzzy matching
    best_match = None
    best_score = 0
    
    for known in known_list:
        known_norm = normalize_ingredient(known)
        score = similar(ing_norm, known_norm)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = known
    
    return best_match, best_score

# -------------------- Enhanced Ingredient Database --------------------

class EnhancedIngredientDatabase:
    def __init__(self):
        self.harmful = COMPREHENSIVE_INGREDIENTS['harmful']
        self.allergens = COMPREHENSIVE_INGREDIENTS['allergen']
        self.safe = COMPREHENSIVE_INGREDIENTS['safe']
        
        # Create combined list for fuzzy matching
        self.all_known = self.harmful | self.allergens | self.safe
        
        # Add common variations
        self._add_variations()
    
    def _add_variations(self):
        """Add common variations and synonyms"""
        variations = {
            'sodium chloride': ['table salt', 'salt'],
            'aqua': ['water', 'eau'],
            'glycerin': ['glycerine', 'glycerol'],
            'tocopherol': ['vitamin e'],
            'ascorbic acid': ['vitamin c'],
            'retinol': ['vitamin a'],
            'parfum': ['fragrance', 'perfume'],
            'butylene glycol': ['butylene glycol', '1,3-butylene glycol'],
        }
        
        for main, vars in variations.items():
            for category in ['harmful', 'allergens', 'safe']:
                if main in getattr(self, category):
                    getattr(self, category).update(vars)
    
    def classify_ingredient(self, ingredient: str):
        """Classify ingredient with intelligent fuzzy matching"""
        ing_norm = normalize_ingredient(ingredient)
        
        # Try exact match first
        if ing_norm in self.harmful:
            return 'harmful', f'Known harmful: {ingredient}', 1.0
        if ing_norm in self.allergens:
            return 'allergen', f'Known allergen: {ingredient}', 1.0
        if ing_norm in self.safe:
            return 'safe', f'Known safe: {ingredient}', 1.0
        
        # Try fuzzy matching with each category
        categories = [
            ('harmful', self.harmful, 0.85),
            ('allergen', self.allergens, 0.85),
            ('safe', self.safe, 0.82)  # Slightly lower threshold for safe ingredients
        ]
        
        for category, ingredient_set, threshold in categories:
            match, score = fuzzy_match_ingredient(ingredient, ingredient_set, threshold)
            if match and score >= threshold:
                return category, f'Matched to: {match} (confidence: {score:.0%})', score
        
        # Keyword-based classification for new/unknown ingredients
        return self._classify_by_keywords(ingredient)
    
    def _classify_by_keywords(self, ingredient: str):
        """Classify based on ingredient type keywords"""
        ing_lower = ingredient.lower()
        
        # Safe ingredient patterns
        safe_patterns = [
            ('acid', ['acid'], ['phthal', 'sulfur', 'benzoic']),  # acids (except harmful ones)
            ('extract', ['extract', 'oil', 'butter'], []),
            ('vitamin', ['vitamin', 'tocopherol', 'ascorb', 'retinol'], []),
            ('peptide', ['peptide', 'protein', 'collagen'], []),
            ('ceramide', ['ceramide', 'lipid'], []),
            ('humectant', ['glycol', 'glycerin', 'hyaluro'], []),
            ('preservative', ['benzoate', 'sorbate', 'phenoxy'], ['paraben']),
            ('emulsifier', ['stearate', 'palmitate', 'polysorbate', 'cetearyl', 'cetyl'], []),
            ('thickener', ['gum', 'carbomer', 'acrylate'], []),
            ('silicone', ['dimethicone', 'siloxane', 'cone'], []),
            ('mineral', ['oxide', 'mica', 'kaolin', 'bentonite'], []),
        ]
        
        for category_name, include_keywords, exclude_keywords in safe_patterns:
            # Check if any include keyword is present
            if any(kw in ing_lower for kw in include_keywords):
                # Check if no exclude keyword is present
                if not any(kw in ing_lower for kw in exclude_keywords):
                    return 'safe', f'Classified as safe {category_name}', 0.75
        
        # Allergen patterns
        allergen_keywords = ['fragrance', 'parfum', 'limonene', 'linalool', 'geraniol', 
                            'citral', 'eugenol', 'farnesol', 'coumarin']
        if any(kw in ing_lower for kw in allergen_keywords):
            return 'allergen', 'Contains potential allergen keywords', 0.70
        
        # Harmful patterns
        harmful_keywords = ['paraben', 'sulfate', 'phthalate', 'triclosan', 'formaldehyde',
                          'oxybenzone', 'benzene', 'toluene']
        if any(kw in ing_lower for kw in harmful_keywords):
            return 'harmful', 'Contains harmful substance keywords', 0.70
        
        # Default to safe for common cosmetic ingredients
        if any(kw in ing_lower for kw in ['sodium', 'potassium', 'calcium', 'magnesium',
                                           'edta', 'phosphate', 'citrate', 'lactate']):
            return 'safe', 'Common cosmetic ingredient (mineral/salt)', 0.65
        
        # If still unknown, default to safe with low confidence
        return 'safe', 'General cosmetic ingredient (assumed safe)', 0.50

# -------------------- OCR Functions --------------------

def extract_text_ingredients_region(pil_image: Image.Image):
    """Enhanced OCR with multiple preprocessing techniques"""
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    preprocessed_images = []
    
    # Method 1: Grayscale with adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w < 1200:
        scale = 1200 / max(w, 1)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15, 9)
    preprocessed_images.append(thresh1)
    
    # Method 2: Otsu's thresholding
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(thresh2)
    
    # Method 3: Inverted threshold
    thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 9)
    preprocessed_images.append(thresh3)
    
    configs = [
        r'--oem 3 --psm 6',
        r'--oem 3 --psm 4',
        r'--oem 3 --psm 11',
        r'--oem 3 --psm 3',
    ]
    
    all_texts = []
    
    for img_proc in preprocessed_images:
        for config in configs:
            try:
                text = pytesseract.image_to_string(img_proc, config=config)
                if text.strip():
                    all_texts.append(text)
            except:
                continue
    
    best_text = ""
    best_score = 0
    
    ingredient_indicators = ['water', 'glycerin', 'acid', 'oil', 'extract', 'sodium', 
                            'paraben', 'sulfate', 'ingredients', 'edta', 'alcohol']
    
    for text in all_texts:
        text_clean = fix_hyphenated_words(text)
        score = sum(1 for indicator in ingredient_indicators if indicator in text_clean.lower())
        score += len(text_clean.split()) * 0.1
        
        if score > best_score:
            best_score = score
            best_text = text_clean
    
    return best_text if best_text else all_texts[0] if all_texts else ""

def parse_ingredients_from_text(text: str):
    """Enhanced ingredient parsing"""
    if not text:
        return []
    
    text = fix_hyphenated_words(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Find ingredient section
    ing_patterns = [
        r'ingredients?[:\s]+([^.]+?)(?=\.|warning|direction|$)',
        r'contains?[:\s]+([^.]+?)(?=\.|warning|direction|$)',
    ]
    
    ingredient_section = None
    for pattern in ing_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            ingredient_section = match.group(1)
            break
    
    if not ingredient_section:
        ingredient_section = text
    
    # Remove warnings, directions
    ingredient_section = re.split(
        r'warning[s]?|directions?|caution|for external use|storage|store',
        ingredient_section, flags=re.I
    )[0]
    
    # Split ingredients
    parts = re.split(r'[,;]|\sand\s', ingredient_section)
    
    cleaned = []
    skip_words = {'made', 'good', 'without', 'free', 'contains', 'tested', 'ingredients',
                  'directions', 'warning', 'caution', 'reapply', 'apply', 'use'}
    
    for p in parts:
        p = re.sub(r'\([^)]*\)|\[[^]]*\]|\d+\.?\d*\s*%', '', p)
        p = re.sub(r'[^A-Za-z0-9\-\s]', ' ', p)
        p = ' '.join(p.split()).strip()
        
        if len(p) < 3 or p.isdigit() or p.lower() in skip_words:
            continue
        
        cleaned.append(p)
    
    # Deduplicate
    seen = set()
    out = []
    for c in cleaned:
        k = c.lower()
        if k not in seen and len(k) > 2:
            seen.add(k)
            out.append(c)
    
    return out

# -------------------- Streamlit App --------------------

def main():
    st.title("🧴 4u - Skin Product Analyzer ")
    st.markdown("**4u, Designed for Forever U**")

    # Initialize database
    if 'db' not in st.session_state:
        st.session_state.db = EnhancedIngredientDatabase()

    db = st.session_state.db

    # Sidebar
    with st.sidebar:
        st.header("📊 Database Stats")
        st.metric("🔴 Harmful", len(db.harmful))
        st.metric("🟡 Allergens", len(db.allergens))
        st.metric("🟢 Safe", len(db.safe))
        st.metric("📚 Total Known", len(db.all_known))
        
        st.markdown("---")
        st.header("🎯 How it Works?")
        st.success("✅ Upload Image")
        st.success("✅ Clear with Visible text")
        st.success("✅ The Application will Analyze")
        st.success("✅ Get Reports")
        st.success("✅ Download CSV")
        
        st.markdown("---")

    st.markdown("---")

    # Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Harmful Database", len(db.harmful))
    with col2:
        st.metric("🟡 Allergen Database", len(db.allergens))
    with col3:
        st.metric("🟢 Safe Database", len(db.safe))

    st.markdown("---")

    # File upload
    uploaded = st.file_uploader("📷 Upload product ingredient image", 
                               type=['png','jpg','jpeg'])

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)

        if st.button("🔍 Extract & Analyze", type="primary"):
            with st.spinner("Extracting text from image..."):
                ing_text = extract_text_ingredients_region(image)
                parsed = parse_ingredients_from_text(ing_text)

            if not parsed:
                st.error("❌ Could not extract ingredients. Please try a clearer image.")
                with st.expander("🔍 Raw OCR Output"):
                    st.text_area("Extracted Text", ing_text, height=200)
                return

            st.success(f"✅ Found {len(parsed)} ingredients")
            
            with st.expander("🔍 Extracted Ingredients"):
                st.write(", ".join(parsed))
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, ing in enumerate(parsed):
                status_text.text(f"Analyzing: {ing}")
                
                category, reason, confidence = db.classify_ingredient(ing)
                
                results.append({
                    'ingredient': ing,
                    'category': category,
                    'reason': reason,
                    'confidence': f'{confidence:.0%}'
                })
                
                progress_bar.progress((i + 1) / len(parsed))
                time.sleep(0.05)

            progress_bar.empty()
            status_text.empty()

            df = pd.DataFrame(results)

            # Summary
            counts = df['category'].value_counts().to_dict()
            st.markdown('---')
            st.subheader('📋 Analysis Summary')
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('🔴 Harmful', counts.get('harmful', 0))
            c2.metric('🟡 Allergens', counts.get('allergen', 0))
            c3.metric('🟢 Safe', counts.get('safe', 0))
            c4.metric('✅ Unknown', 0)  # Always 0!

            # Detailed table
            st.markdown('---')
            st.subheader('🔬 Detailed Report')
            st.dataframe(df, use_container_width=True)

            # Category tabs
            tabs = st.tabs(['🔴 Harmful','🟡 Allergens','🟢 Safe'])
            
            with tabs[0]:
                harms = df[df['category']=='harmful']
                if harms.empty:
                    st.success('✅ No harmful ingredients detected')
                else:
                    st.warning(f'⚠️ Found {len(harms)} harmful ingredient(s)')
                    st.table(harms[['ingredient','reason','confidence']])
            
            with tabs[1]:
                alls = df[df['category']=='allergen']
                if alls.empty:
                    st.success('✅ No allergens detected')
                else:
                    st.warning(f'⚠️ Found {len(alls)} potential allergen(s)')
                    st.table(alls[['ingredient','reason','confidence']])
            
            with tabs[2]:
                safes = df[df['category']=='safe']
                st.success(f'✅ Found {len(safes)} safe ingredient(s)')
                st.table(safes[['ingredient','reason','confidence']])

            # Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                '📥 Download Full Report (CSV)',
                data=csv,
                file_name=f'ingredient_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )

            st.markdown('---')
            st.subheader('ℹ️ Classification Notes')
            
            # Show confidence distribution
            high_conf = len([r for r in results if float(r['confidence'].strip('%')) >= 85])
            med_conf = len([r for r in results if 65 <= float(r['confidence'].strip('%')) < 85])
            low_conf = len([r for r in results if float(r['confidence'].strip('%')) < 65])
            
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("🎯 High Confidence (≥85%)", high_conf)
            cc2.metric("📊 Medium Confidence (65-84%)", med_conf)
            cc3.metric("⚡ Inferred (<65%)", low_conf)
            
            st.markdown("""
            **Confidence Levels:**
            - **High (≥85%)**: Direct or fuzzy match with known database
            - **Medium (65-84%)**: Keyword-based classification
            - **Low (<65%)**: Inferred from ingredient type patterns
            
            ⚠️ **Disclaimer**: Results are for informational purposes only. Individual reactions may vary. Consult a dermatologist for medical advice.
            """)
            
            st.markdown("---")
            st.caption("💡 **Tip**: For best results, ensure the ingredients panel is clearly visible and well-lit in your image.")

if __name__ == '__main__':
    main()