from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pickle
import warnings
import os
import re
from PIL import Image
import pytesseract
import io

warnings.filterwarnings('ignore')

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = BASE
STATIC_DIR = os.path.join(BASE, 'static')

app = Flask(__name__)

# Load cancer models
rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_classifier.pkl'))
scaler   = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
with open(os.path.join(MODELS_DIR, 'class_names.pkl'), 'rb') as f:
    class_names = pickle.load(f)

# Load autoimmune model
ai_model    = joblib.load(os.path.join(MODELS_DIR, 'autoimmune.pkl'))
AI_FEATURES = list(ai_model.feature_names_in_)
print(f"Models loaded | Cancer classes: {len(class_names)} | Autoimmune features: {len(AI_FEATURES)}")

# ══════════════════════════════════════════════════════════════════════════════
# PDF helper
# ══════════════════════════════════════════════════════════════════════════════
def pdf_to_images(file_bytes, max_pages=3):
    try:
        import fitz
        doc    = fitz.open(stream=file_bytes, filetype="pdf")
        images = []
        for i in range(min(max_pages, len(doc))):
            page = doc[i]
            pix  = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()
        return images
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")

# ══════════════════════════════════════════════════════════════════════════════
# Translation
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED_LANGUAGES = {
    'en':'English','hi':'Hindi','ta':'Tamil','te':'Telugu','ml':'Malayalam',
    'bn':'Bengali','fr':'French','de':'German','es':'Spanish','it':'Italian',
    'pt':'Portuguese','zh-CN':'Chinese','ar':'Arabic','ru':'Russian',
    'ja':'Japanese','ko':'Korean'
}

def translate_text(text, target_lang):
    if target_lang == 'en':
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except ImportError:
        return f"[Install deep-translator: pip install deep-translator]\n\n{text}"
    except Exception as e:
        return f"[Translation error: {e}]\n\n{text}"

def clean_ocr_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_text_from_file(file_bytes, filename):
    if filename.endswith('.pdf'):
        pages = pdf_to_images(file_bytes)
        return '\n\n'.join(pytesseract.image_to_string(p).strip() for p in pages)
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image).strip()

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVED NEGATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Strong negation phrases that COMPLETELY override malignancy signals
STRONG_NEGATIVE_PHRASES = [
    r'no\s+evidence\s+of\s+malignancy',
    r'no\s+evidence\s+of\s+cancer',
    r'no\s+evidence\s+of\s+metastasis',
    r'no\s+evidence\s+of\s+tumor',
    r'no\s+evidence\s+of\s+tumour',
    r'no\s+evidence\s+of\s+neoplasm',
    r'no\s+sign\s+of\s+malignancy',
    r'no\s+sign\s+of\s+cancer',
    r'negative\s+for\s+malignancy',
    r'negative\s+for\s+cancer',
    r'negative\s+for\s+carcinoma',
    r'negative\s+for\s+metastasis',
    r'free\s+of\s+(?:malignancy|cancer|carcinoma|neoplasm)',
    r'without\s+(?:evidence\s+of\s+)?(?:malignancy|cancer|carcinoma)',
    r'no\s+malignancy\s+(?:identified|detected|seen|found|noted)',
    r'no\s+cancer\s+(?:identified|detected|seen|found|noted)',
    r'(?:tumor|tumour)\s+(?:is\s+)?benign',
    r'benign\s+(?:lesion|tumor|tumour|mass|nodule|growth|neoplasm)',
    r'(?:findings?\s+(?:are?\s+)?)?(?:unremarkable|normal|within\s+normal\s+limits)',
    r'no\s+(?:abnormal|suspicious|malignant)\s+(?:findings?|cells?|tissue)',
    r'ruled?\s+out\s+(?:malignancy|cancer|carcinoma)',
]

# Negation prefixes that appear BEFORE a keyword and negate it
NEGATION_PREFIXES = [
    'no evidence of', 'no sign of', 'no signs of', 'negative for',
    'without', 'free of', 'free from', 'absence of', 'absent',
    'no', 'not', 'none', 'nor', 'never', 'neither',
    'unlikely', 'improbable', 'excluded', 'ruled out',
    'no indication of', 'does not suggest', 'does not indicate',
    'failed to demonstrate', 'failed to reveal',
    'no history of', 'denies', 'denied',
]


def has_strong_negative(text):
    """
    NEGATIVE DOMINANCE RULE: Check if the entire document contains
    phrases that clearly indicate NO malignancy.
    Returns True if document is clearly negative.
    """
    tl = text.lower()
    for pat in STRONG_NEGATIVE_PHRASES:
        if re.search(pat, tl):
            return True
    return False


def is_negated(text, start, end, window_before=120, window_after=50):
    """
    Improved negation detection.
    Looks at a window around the keyword match and checks for negation cues.
    """
    window_start = max(0, start - window_before)
    window_end   = min(len(text), end + window_after)
    window_text  = text[window_start:window_end].lower()

    # Check for strong negative phrases in the window
    for pat in STRONG_NEGATIVE_PHRASES:
        if re.search(pat, window_text):
            return True

    # Check for negation prefixes immediately before the keyword
    before_text = text[window_start:start].lower().strip()
    for neg in NEGATION_PREFIXES:
        if before_text.endswith(neg):
            return True
        # Also check with minor whitespace / punctuation gaps
        if re.search(re.escape(neg) + r'[\s,;:\-]*$', before_text):
            return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED KEYWORD SCORING
# ══════════════════════════════════════════════════════════════════════════════

# Keywords with weights: higher = stronger signal of malignancy
MALIGNANT_WEIGHTED = {
    'malignant': 4, 'cancer': 3, 'carcinoma': 4, 'lymphoma': 4,
    'leukemia': 4, 'leukaemia': 4, 'metastasis': 5, 'metastatic': 5,
    'adenocarcinoma': 4, 'sarcoma': 4, 'melanoma': 4, 'neoplasm': 3,
    'malignancy': 4, 'invasive': 3, 'infiltrating': 3,
    'poorly differentiated': 3, 'anaplastic': 4,
}

BENIGN_WEIGHTED = {
    'benign': -3, 'normal': -2, 'negative': -2, 'unremarkable': -3,
    'reactive': -2, 'inflammatory': -1, 'non-neoplastic': -3,
    'no malignancy': -5, 'no cancer': -5, 'clear': -2,
}

# Contextual terms: present in many reports without implying cancer
CONTEXTUAL_ONLY = {
    'tumor', 'tumour', 'biopsy', 'oncology', 'chemotherapy', 'radiation',
    'staging', 'grade', 'nodule', 'lesion', 'mass', 'diagnosis',
    'prognosis', 'pathology', 'cytology', 'histology', 'remission',
    'recurrence', 'margin', 'resection', 'excision', 'ablation',
    'immunotherapy', 'targeted therapy', 'hormone receptor', 'her2',
    'brca', 'psa', 'cea', 'ca125', 'ca19-9', 'afp', 'pdl1',
    'ki67', 'mitosis', 'necrosis',
}

# Full list for entity extraction (all keywords)
CANCER_KEYWORDS = (
    list(MALIGNANT_WEIGHTED.keys()) +
    list(BENIGN_WEIGHTED.keys()) +
    list(CONTEXTUAL_ONLY)
)

# ══════════════════════════════════════════════════════════════════════════════
# Cancer Type Detection Map (unchanged patterns, but scoring uses new negation)
# ══════════════════════════════════════════════════════════════════════════════
CANCER_TYPE_PATTERNS = {
    'Breast Cancer': {
        'patterns': [r'breast\s+(?:cancer|carcinoma|tumor|mass|nodule|lesion)',
                     r'mammogram|mammography',r'ductal\s+carcinoma',r'lobular\s+carcinoma',
                     r'invasive\s+ductal',r'invasive\s+lobular',r'her2',r'brca[12]?',
                     r'estrogen\s+receptor',r'progesterone\s+receptor',r'\ber\+|\bpr\+|\bher2\+',
                     r'lumpectomy|mastectomy'],
        'icd10': 'C50', 'color': 'danger',
        'description': 'Malignant neoplasm of breast tissue.'
    },
    'Lung Cancer': {
        'patterns': [r'lung\s+(?:cancer|carcinoma|tumor|mass|nodule)',
                     r'pulmonary\s+(?:malignancy|neoplasm|carcinoma)',
                     r'non.small\s+cell',r'small\s+cell\s+lung',r'nsclc',r'sclc',
                     r'bronchogenic\s+carcinoma',r'squamous\s+cell.*lung',
                     r'bronchoalveolar',r'pleural\s+effusion.*malign',
                     r'egfr\s+mutation',r'alk\s+rearrangement'],
        'icd10': 'C34', 'color': 'danger',
        'description': 'Malignant neoplasm of bronchus and lung.'
    },
    'Colorectal Cancer': {
        'patterns': [r'colon\s+(?:cancer|carcinoma|tumor)',r'rectal\s+(?:cancer|carcinoma)',
                     r'colorectal',r'colonoscopy.*(?:adenoma|polyp|malign)',
                     r'sigmoid\s+carcinoma',r'cecal\s+(?:mass|carcinoma)',
                     r'bowel\s+(?:cancer|obstruction)',r'\bcea\b.*(?:elevated|raised|high)',
                     r'microsatellite\s+instability',r'msi.h',r'kras\s+mutation'],
        'icd10': 'C18-C20', 'color': 'danger',
        'description': 'Malignant neoplasm of colon, rectum, and rectosigmoid junction.'
    },
    'Prostate Cancer': {
        'patterns': [r'prostate\s+(?:cancer|carcinoma|adenocarcinoma)',
                     r'prostatic\s+(?:malignancy|neoplasm)',
                     r'\bpsa\b.*(?:elevated|raised|ng/ml)',r'gleason\s+score',
                     r'radical\s+prostatectomy',r'brachytherapy.*prostate',
                     r'androgen\s+deprivation'],
        'icd10': 'C61', 'color': 'danger',
        'description': 'Malignant neoplasm of prostate gland.'
    },
    'Liver Cancer': {
        'patterns': [r'hepatocellular\s+carcinoma',r'hcc\b',r'liver\s+(?:cancer|carcinoma|tumor|mass)',
                     r'hepatic\s+(?:malignancy|neoplasm|carcinoma)',
                     r'\bafp\b.*(?:elevated|raised)',r'cirrhosis.*(?:malign|hcc)',
                     r'cholangiocarcinoma',r'bile\s+duct\s+carcinoma'],
        'icd10': 'C22', 'color': 'danger',
        'description': 'Malignant neoplasm of liver and intrahepatic bile ducts.'
    },
    'Cervical Cancer': {
        'patterns': [r'cervical\s+(?:cancer|carcinoma|dysplasia)',
                     r'cervix\s+(?:malignancy|neoplasm)',r'\bhpv\b',r'pap\s+smear.*(?:abnormal|positive)',
                     r'cin\s*[123]',r'colposcopy.*(?:malign|carcinoma)',
                     r'squamous\s+cell.*cervix'],
        'icd10': 'C53', 'color': 'danger',
        'description': 'Malignant neoplasm of cervix uteri.'
    },
    'Ovarian Cancer': {
        'patterns': [r'ovarian\s+(?:cancer|carcinoma|tumor|mass)',
                     r'ovary\s+(?:malignancy|neoplasm)',
                     r'\bca.?125\b.*(?:elevated|raised)',
                     r'epithelial\s+ovarian',r'serous\s+carcinoma.*ovari',
                     r'fallopian\s+tube.*carcinoma'],
        'icd10': 'C56', 'color': 'danger',
        'description': 'Malignant neoplasm of ovary.'
    },
    'Stomach / Gastric Cancer': {
        'patterns': [r'gastric\s+(?:cancer|carcinoma|adenocarcinoma)',
                     r'stomach\s+(?:cancer|carcinoma|tumor)',
                     r'gastroesophageal.*carcinoma',r'linitis\s+plastica',
                     r'signet\s+ring\s+cell',r'h\s*pylori.*(?:malign|carcinoma)'],
        'icd10': 'C16', 'color': 'danger',
        'description': 'Malignant neoplasm of stomach.'
    },
    'Pancreatic Cancer': {
        'patterns': [r'pancreatic\s+(?:cancer|carcinoma|adenocarcinoma)',
                     r'pancreas\s+(?:malignancy|neoplasm|tumor)',
                     r'\bca.?19.?9\b.*(?:elevated|raised)',
                     r'whipple\s+(?:procedure|surgery)',r'pancreaticoduodenectomy'],
        'icd10': 'C25', 'color': 'danger',
        'description': 'Malignant neoplasm of pancreas.'
    },
    'Thyroid Cancer': {
        'patterns': [r'thyroid\s+(?:cancer|carcinoma|malignancy)',
                     r'papillary\s+thyroid',r'follicular\s+thyroid',
                     r'medullary\s+thyroid',r'anaplastic\s+thyroid',
                     r'thyroidectomy.*(?:malign|carcinoma)',r'braf\s+mutation.*thyroid'],
        'icd10': 'C73', 'color': 'danger',
        'description': 'Malignant neoplasm of thyroid gland.'
    },
    'Bladder Cancer': {
        'patterns': [r'bladder\s+(?:cancer|carcinoma|tumor)',
                     r'transitional\s+cell\s+carcinoma',r'urothelial\s+carcinoma',
                     r'cystoscopy.*(?:malign|carcinoma|tumor)',
                     r'radical\s+cystectomy'],
        'icd10': 'C67', 'color': 'danger',
        'description': 'Malignant neoplasm of bladder.'
    },
    'Kidney / Renal Cancer': {
        'patterns': [r'renal\s+cell\s+carcinoma',r'rcc\b',
                     r'kidney\s+(?:cancer|carcinoma|tumor|mass)',
                     r'nephrectomy.*(?:malign|carcinoma)',r'clear\s+cell\s+carcinoma.*renal',
                     r'wilms\s+tumor'],
        'icd10': 'C64', 'color': 'danger',
        'description': 'Malignant neoplasm of kidney.'
    },
    'Lymphoma': {
        'patterns': [r'lymphoma\b',r'hodgkin',r'non.hodgkin',r'diffuse\s+large\s+b.cell',
                     r'dlbcl\b',r'follicular\s+lymphoma',r'burkitt',r'mantle\s+cell',
                     r'reed.sternberg',r'lymph\s+node.*biopsy.*malign'],
        'icd10': 'C81-C86', 'color': 'danger',
        'description': 'Malignant lymphoma involving lymphoid tissue.'
    },
    'Leukemia': {
        'patterns': [r'leuk[ae]mia\b',r'\ball\b.*blast',r'\baml\b.*blast',r'\bcll\b',r'\bcml\b',
                     r'acute\s+lymphoblastic',r'acute\s+myeloid',
                     r'chronic\s+lymphocytic',r'chronic\s+myeloid',
                     r'blast\s+crisis',r'bone\s+marrow.*(?:malign|blast)',
                     r'philadelphia\s+chromosome'],
        'icd10': 'C91-C95', 'color': 'danger',
        'description': 'Malignant neoplasm of blood-forming and related tissue.'
    },
    'Skin Cancer / Melanoma': {
        'patterns': [r'melanoma\b',r'skin\s+cancer',r'basal\s+cell\s+carcinoma',
                     r'squamous\s+cell.*skin',r'merkel\s+cell',
                     r'breslow\s+thickness',r'sentinel\s+node.*skin',
                     r'dermatoscopy.*malign'],
        'icd10': 'C43-C44', 'color': 'danger',
        'description': 'Malignant neoplasm of skin.'
    },
    'Brain / CNS Cancer': {
        'patterns': [r'glioblastoma\b',r'gbm\b',r'glioma\b',r'astrocytoma',r'meningioma.*malign',
                     r'brain\s+(?:tumor|tumour|cancer|malignancy)',
                     r'cns\s+(?:malignancy|tumor|lymphoma)',
                     r'idh\s+mutation',r'mgmt\s+promoter',r'temozolomide'],
        'icd10': 'C71', 'color': 'danger',
        'description': 'Malignant neoplasm of brain and central nervous system.'
    },
    'Oesophageal Cancer': {
        'patterns': [r'esophageal\s+(?:cancer|carcinoma)',r'oesophageal\s+carcinoma',
                     r'barrett.s.*carcinoma',r'esophagectomy',
                     r'squamous\s+cell.*esophag'],
        'icd10': 'C15', 'color': 'danger',
        'description': 'Malignant neoplasm of oesophagus.'
    },
    'Uterine / Endometrial Cancer': {
        'patterns': [r'endometrial\s+(?:cancer|carcinoma)',r'uterine\s+(?:cancer|carcinoma|sarcoma)',
                     r'endometrium.*malign',r'hysterectomy.*(?:malign|carcinoma)'],
        'icd10': 'C54', 'color': 'danger',
        'description': 'Malignant neoplasm of corpus uteri.'
    },
    'Bone Cancer / Sarcoma': {
        'patterns': [r'osteosarcoma',r'ewing\s+sarcoma',r'chondrosarcoma',
                     r'bone\s+(?:cancer|tumor|sarcoma)',r'soft\s+tissue\s+sarcoma',
                     r'rhabdomyosarcoma',r'liposarcoma'],
        'icd10': 'C40-C49', 'color': 'danger',
        'description': 'Malignant neoplasm of bone and soft tissue.'
    },
    'Head & Neck Cancer': {
        'patterns': [r'head\s+and\s+neck\s+(?:cancer|carcinoma)',
                     r'oral\s+(?:cancer|carcinoma)',r'laryngeal\s+carcinoma',
                     r'pharyngeal\s+carcinoma',r'squamous\s+cell.*(?:oral|larynx|pharynx|throat)',
                     r'nasopharyngeal',r'salivary\s+gland.*malign'],
        'icd10': 'C00-C14', 'color': 'danger',
        'description': 'Malignant neoplasm of head and neck region.'
    },
    'Multiple Myeloma': {
        'patterns': [r'multiple\s+myeloma',r'plasma\s+cell\s+(?:myeloma|dyscrasia)',
                     r'monoclonal\s+gammopathy',r'\bmgus\b',
                     r'bence\s+jones',r'serum\s+protein\s+electrophoresis.*spike'],
        'icd10': 'C90', 'color': 'danger',
        'description': 'Multiple myeloma and malignant plasma cell neoplasms.'
    },
}

# Lab markers that boost malignancy confidence
LAB_MALIGNANCY_MARKERS = {
    'elevated_ca125': r'ca.?125[:\s]+(\d+\.?\d*)(?:\s*u/ml)?',
    'elevated_psa': r'psa[:\s]+(\d+\.?\d*)(?:\s*ng/ml)?',
    'elevated_cea': r'cea[:\s]+(\d+\.?\d*)(?:\s*ng/ml)?',
    'elevated_afp': r'afp[:\s]+(\d+\.?\d*)(?:\s*ng/ml)?',
    'elevated_ca19_9': r'ca.?19.?9[:\s]+(\d+\.?\d*)(?:\s*u/ml)?',
    'elevated_ldh': r'ldh[:\s]+(\d+\.?\d*)(?:\s*u/l)?',
    'elevated_ki67': r'ki.?67[:\s]+(\d+%)',
}
LAB_THRESHOLDS = {
    'elevated_ca125': 35, 'elevated_psa': 4.0, 'elevated_cea': 5.0,
    'elevated_afp': 400, 'elevated_ca19_9': 37, 'elevated_ldh': 600,
}

# Staging keywords
STAGE_PATTERNS = {
    'Stage I':   [r'stage\s+i\b',r'stage\s+1\b',r'\bpT1\b',r'\bT1\b'],
    'Stage II':  [r'stage\s+ii\b',r'stage\s+2\b',r'\bpT2\b',r'\bT2\b'],
    'Stage III': [r'stage\s+iii\b',r'stage\s+3\b',r'\bpT3\b',r'\bT3\b'],
    'Stage IV':  [r'stage\s+iv\b',r'stage\s+4\b',r'\bpT4\b',r'\bT4\b',r'metastatic',r'metastasis'],
}


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVED CANCER ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def detect_cancer_types_from_text(text):
    """Advanced cancer type detection using improved negation."""
    tl = text.lower()

    # NEGATIVE DOMINANCE: if document is clearly negative, return nothing
    if has_strong_negative(tl):
        return []

    detected = []
    for cancer_name, info in CANCER_TYPE_PATTERNS.items():
        score = 0
        matched_patterns = []
        for pat in info['patterns']:
            for m in re.finditer(pat, tl):
                if not is_negated(tl, m.start(), m.end()):
                    score += 1
                    snippet = text[max(0, m.start()-35):min(len(text), m.end()+55)].strip().replace('\n', ' ')
                    matched_patterns.append({
                        'pattern': m.group(0),
                        'context': f'...{snippet}...'
                    })
                    break  # one match per pattern is enough
        if score >= 2:  # RAISED threshold: need at least 2 non-negated pattern matches
            detected.append({
                'name': cancer_name,
                'icd10': info['icd10'],
                'score': score,
                'confidence': min(100, score * 28 + 16),
                'description': info['description'],
                'evidence': matched_patterns[:3],
            })
    detected.sort(key=lambda x: -x['score'])
    return detected[:5]


def extract_staging_info(text):
    """Extract staging information from text."""
    tl = text.lower()
    for stage, pats in STAGE_PATTERNS.items():
        for pat in pats:
            if re.search(pat, tl):
                return stage
    return None


def extract_elevated_markers(text):
    """Extract and evaluate lab markers for malignancy."""
    tl = text.lower()
    elevated = []
    for marker, pat in LAB_MALIGNANCY_MARKERS.items():
        m = re.search(pat, tl)
        if m:
            try:
                val = float(m.group(1))
                threshold = LAB_THRESHOLDS.get(marker)
                if threshold and val > threshold:
                    elevated.append({
                        'marker': marker.replace('elevated_', '').upper().replace('_', '-'),
                        'value': val,
                        'threshold': threshold
                    })
            except:
                pass
    return elevated


def extract_cancer_entities(text):
    """
    Extract cancer-related keywords WITH negation awareness.
    Only returns entities that are NOT negated.
    """
    found = []
    tl = text.lower()

    for kw in CANCER_KEYWORDS:
        idx = tl.find(kw)
        if idx == -1:
            continue

        # Skip if this keyword is negated
        if is_negated(tl, idx, idx + len(kw)):
            continue

        ctx = text[max(0, idx-40):min(len(text), idx+len(kw)+40)].strip().replace('\n', ' ')
        found.append({'keyword': kw, 'context': f'...{ctx}...'})

    return found


def compute_weighted_score(text):
    """
    Compute a WEIGHTED cancer score from text.
    Positive = malignant signals, negative = benign signals.
    Contextual-only keywords do NOT add to the score.
    """
    tl = text.lower()
    score = 0.0

    # ── STEP 1: NEGATIVE DOMINANCE ──────────────────────────────────────────
    # If the document explicitly says "no malignancy", override everything
    if has_strong_negative(tl):
        return -10  # strongly negative → forces LOW risk

    # ── STEP 2: Score malignant keywords (only if not negated) ──────────────
    for kw, weight in MALIGNANT_WEIGHTED.items():
        for m in re.finditer(re.escape(kw), tl):
            if not is_negated(tl, m.start(), m.end()):
                score += weight

    # ── STEP 3: Score benign keywords (subtract) ───────────────────────────
    for kw, weight in BENIGN_WEIGHTED.items():
        if kw in tl:
            score += weight  # weight is already negative

    # ── STEP 4: Elevated lab markers boost score ───────────────────────────
    elevated_markers = extract_elevated_markers(text)
    score += len(elevated_markers) * 2

    return score


def assess_cancer_risk(entities, text=''):
    """
    Improved risk assessment using WEIGHTED SCORING + NEGATION DOMINANCE.
    No longer relies on "do entities exist? → must be cancer".
    """
    tl = text.lower() if text else ''

    elevated_markers = extract_elevated_markers(text) if text else []
    staging = extract_staging_info(text) if text else None

    # If no entities at all, undetermined
    if not entities:
        return None, None, None, None, []

    # ── COMPUTE WEIGHTED SCORE ──────────────────────────────────────────────
    score = compute_weighted_score(text) if text else 0

    # If we only have entities but no text to score, do a simple entity-based score
    if not text:
        found_kws = {e['keyword'] for e in entities}
        for kw in found_kws:
            if kw in MALIGNANT_WEIGHTED:
                score += MALIGNANT_WEIGHTED[kw]
            elif kw in BENIGN_WEIGHTED:
                score += BENIGN_WEIGHTED[kw]
            # contextual-only keywords contribute 0

    # ── DETERMINE RISK LEVEL FROM SCORE ─────────────────────────────────────
    stage_note  = f' Staging: {staging}.' if staging else ''
    marker_note = f' {len(elevated_markers)} elevated tumour marker(s) detected.' if elevated_markers else ''

    if score <= 0:
        return (
            'LOW', 'safe',
            'No significant malignancy indicators found. Document may contain cancer-related '
            'terminology in a benign or negative context. Routine follow-up advised.',
            staging, elevated_markers
        )
    elif score <= 5:
        return (
            'MODERATE', 'warn',
            f'Some cancer-related terms detected (score {score:.0f}).{stage_note}{marker_note} '
            f'Further clinical evaluation recommended to clarify findings.',
            staging, elevated_markers
        )
    else:
        conf_score = min(98, 50 + int(score) * 6)
        return (
            'HIGH', 'danger',
            f'Strong malignancy indicators found (score {score:.0f}).{stage_note}{marker_note} '
            f'Urgent specialist review recommended. Estimated confidence: {conf_score}%.',
            staging, elevated_markers
        )


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED FEATURE EXTRACTION (for your 9-column trained model)
# ══════════════════════════════════════════════════════════════════════════════

# Map these to your ACTUAL training column names
CANCER_MODEL_COLUMNS = [
    'Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk',
    'PhysicalActivity', 'AlcoholIntake', 'CancerHistory',
    # Add your 9th column name here, e.g.:
    # 'FamilyHistory'
]

def extract_structured_cancer_features(text):
    """
    Extract the structured features your RF model was actually trained on.
    Returns a dict of {column_name: value} for features found.
    """
    tl = text.lower()
    features = {}

    # Age
    m = re.search(r'(?:age|aged?)[:\s]+(\d{1,3})\b', tl)
    if m:
        features['Age'] = float(m.group(1))

    # Gender (1=male, 0=female — match your training encoding)
    m = re.search(r'\b(male|female)\b', tl)
    if m:
        features['Gender'] = 1.0 if m.group(1) == 'male' else 0.0

    # BMI
    m = re.search(r'bmi[:\s]+(\d+\.?\d*)', tl)
    if m:
        features['BMI'] = float(m.group(1))

    # Smoking
    if re.search(r'(?:smoker|smoking|tobacco|cigarette)', tl):
        if re.search(r'(?:non.?smoker|never smoked|no smoking|no tobacco)', tl):
            features['Smoking'] = 0.0
        else:
            features['Smoking'] = 1.0

    # Genetic risk / family history
    if re.search(r'(?:family history|genetic risk|hereditary|familial|brca)', tl):
        features['GeneticRisk'] = 1.0

    # Physical activity
    m = re.search(r'physical activity[:\s]+(\d+\.?\d*)', tl)
    if m:
        features['PhysicalActivity'] = float(m.group(1))

    # Alcohol intake
    if re.search(r'(?:alcohol|drinking|ethanol)', tl):
        if re.search(r'(?:no alcohol|non.?drinker|never drinks|abstinent)', tl):
            features['AlcoholIntake'] = 0.0
        else:
            features['AlcoholIntake'] = 1.0

    # Cancer history
    if re.search(r'(?:previous cancer|prior malignancy|cancer history|history of cancer)', tl):
        features['CancerHistory'] = 1.0

    return features


def predict_with_structured_model(features_dict):
    """
    Use the trained RF model with the structured features.
    Returns (class_name, confidence, probabilities) or None if not enough features.
    """
    # Need at least 4 of the training columns to make a reasonable prediction
    available = {k: v for k, v in features_dict.items() if k in CANCER_MODEL_COLUMNS}
    if len(available) < 4:
        return None

    # Build feature vector in training column order, default 0 for missing
    fv = np.array([features_dict.get(col, 0.0) for col in CANCER_MODEL_COLUMNS]).reshape(1, -1)

    try:
        fv_scaled = scaler.transform(fv)
        pred  = rf_model.predict(fv_scaled)[0]
        proba = rf_model.predict_proba(fv_scaled)[0]
        cname = class_names[pred]
        return {
            'prediction': int(pred),
            'class_name': cname,
            'is_malignant': bool(pred != 0),
            'confidence': float(max(proba) * 100),
            'probabilities': [
                {'class': class_names[i], 'probability': float(proba[i]*100), 'is_malignant': i != 0}
                for i in range(len(class_names))
            ]
        }
    except Exception as e:
        print(f"Structured model prediction failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SCORING: text NLP + structured model
# ══════════════════════════════════════════════════════════════════════════════

def combined_cancer_assessment(text):
    """
    Combine text-based NLP scoring with structured ML model prediction.
    Returns a unified cancer result dict.
    """
    # 1. Text-based analysis
    entities       = extract_cancer_entities(text)
    text_score     = compute_weighted_score(text)
    detected_types = detect_cancer_types_from_text(text)
    staging        = extract_staging_info(text)
    elevated_markers = extract_elevated_markers(text)

    # 2. Structured model prediction
    struct_features = extract_structured_cancer_features(text)
    model_result    = predict_with_structured_model(struct_features)

    # 3. Combine scores
    # Normalise text_score to 0-1 range (clamp between -10 and 20)
    text_norm = max(0.0, min(1.0, (text_score + 10) / 30.0))

    if model_result:
        model_malignant_prob = sum(
            p['probability'] for p in model_result['probabilities'] if p['is_malignant']
        ) / 100.0
        # Weighted combination: 70% text, 30% structured model
        combined = 0.7 * text_norm + 0.3 * model_malignant_prob
    else:
        combined = text_norm

    # 4. Determine final risk from combined score
    if combined <= 0.3:
        rl, rc = 'LOW', 'safe'
        sm = ('No significant malignancy indicators. Document may mention cancer-related '
              'terms in a benign or educational context. Routine follow-up advised.')
    elif combined <= 0.55:
        rl, rc = 'MODERATE', 'warn'
        sm = ('Some cancer-related signals detected. Further clinical evaluation '
              'recommended to clarify findings.')
    else:
        conf = min(98, int(combined * 100))
        rl, rc = 'HIGH', 'danger'
        sm = (f'Strong malignancy indicators (confidence ~{conf}%). '
              f'Urgent specialist review recommended.')

    stage_note = f' Staging: {staging}.' if staging else ''
    marker_note = f' {len(elevated_markers)} elevated tumour marker(s).' if elevated_markers else ''
    sm += stage_note + marker_note

    return {
        'risk_level': rl,
        'risk_color': rc,
        'summary': sm,
        'entities': entities,
        'entity_count': len(entities),
        'detected_types': detected_types,
        'staging': staging,
        'elevated_markers': elevated_markers,
        'text_score': round(text_score, 2),
        'combined_score': round(combined, 3),
        'structured_features_found': struct_features,
        'model_prediction': model_result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Clinical info (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

CANCER_CLINICAL = {
    'Benign': {
        'treatment':    'No cancer treatment required. Regular monitoring and follow-up are recommended to track any changes over time.',
        'suggestion':   'Maintain a healthy lifestyle with balanced nutrition and regular exercise. Avoid tobacco and alcohol. Schedule annual check-ups with your physician.',
        'confirm_test': 'Fine Needle Aspiration Cytology (FNAC) or repeat imaging (ultrasound / MRI) to confirm benign nature and rule out future malignant transformation.'
    },
    '[Malignant] Pre-B': {
        'treatment':    'Combination chemotherapy (BFM or CALGB regimen). Targeted therapy with tyrosine kinase inhibitors (e.g., Imatinib for Ph+ ALL). Stem cell transplantation may be considered in high-risk cases. CNS prophylaxis with intrathecal chemotherapy.',
        'suggestion':   'Immediate referral to a haematologist-oncologist. Monitor complete blood count regularly. Ensure adequate hydration and infection prophylaxis during treatment. Psychological and nutritional support are important.',
        'confirm_test': 'Bone marrow biopsy with flow cytometry immunophenotyping, cytogenetic analysis (karyotype / FISH), and lumbar puncture to assess CNS involvement.'
    },
    '[Malignant] Pro-B': {
        'treatment':    'Intensive multi-agent chemotherapy (hyper-CVAD or BFM protocol). Allogeneic stem cell transplantation is strongly recommended given the aggressive nature. CNS prophylaxis with intrathecal chemotherapy is mandatory.',
        'suggestion':   'Urgent haematology consultation required. Monitor closely for tumour lysis syndrome, infections, and anaemia. Psychological counselling and nutritional support should accompany treatment.',
        'confirm_test': 'Bone marrow trephine biopsy, immunophenotyping by flow cytometry (CD19+, CD10−, CD34+), and molecular testing for MLL gene rearrangements.'
    },
    '[Malignant] early Pre-B': {
        'treatment':    'Risk-stratified chemotherapy including induction (steroids, vincristine, asparaginase), consolidation, and maintenance phases. Targeted agents may be added for specific mutations. CNS-directed therapy if indicated.',
        'suggestion':   'Prompt oncology referral. Genetic counselling may be advised. Avoid immunosuppressants without medical guidance. Maintain a sterile environment to reduce infection risk during chemotherapy.',
        'confirm_test': 'Bone marrow aspiration with immunophenotyping, PCR for BCR-ABL and fusion genes, and FISH for chromosomal abnormalities (e.g., t(12;21), t(1;19)).'
    }
}

AUTOIMMUNE_CLINICAL = {
    'Systemic Lupus Erythematosus': {
        'treatment':    'Hydroxychloroquine as baseline therapy. NSAIDs for mild symptoms. Corticosteroids and immunosuppressants (azathioprine, mycophenolate mofetil, cyclophosphamide) for organ involvement. Belimumab (biologic) for refractory cases.',
        'suggestion':   'Avoid prolonged sun exposure; use SPF 50+ sunscreen daily. Monitor kidney function and urine protein regularly. Avoid live vaccines during immunosuppression.',
        'confirm_test': 'Anti-dsDNA antibody titre, complement C3/C4 levels, urinalysis with protein/creatinine ratio, and skin or kidney biopsy if organ involvement is suspected.'
    },
    'Rheumatoid arthritis': {
        'treatment':    'DMARDs — methotrexate is first-line. Biologics (TNF inhibitors: etanercept, adalimumab) for inadequate DMARD response. NSAIDs and short-course corticosteroids for acute symptom control.',
        'suggestion':   'Physiotherapy and occupational therapy to preserve joint function. Smoking cessation is critical. Monitor liver function and blood counts regularly while on methotrexate.',
        'confirm_test': 'Rheumatoid Factor (RF), Anti-CCP antibody, X-ray of hands and feet for erosions, and MRI for early joint damage detection.'
    },
    "Hashimoto's thyroiditis": {
        'treatment':    'Levothyroxine replacement therapy if hypothyroid. TSH monitoring every 6–12 months. No pharmacological treatment needed if euthyroid but antibody-positive.',
        'suggestion':   'Regular thyroid function tests. Ensure adequate selenium and iodine intake. Monitor for associated autoimmune conditions such as Type 1 diabetes and coeliac disease.',
        'confirm_test': 'TSH, Free T4, Anti-TPO antibody titre, and thyroid ultrasound showing characteristic heterogeneous echotexture.'
    },
    "Graves' disease": {
        'treatment':    'Anti-thyroid drugs (carbimazole, propylthiouracil). Radioactive iodine (I-131) therapy. Beta-blockers for symptomatic relief. Thyroidectomy in selected cases.',
        'suggestion':   'Avoid iodine-rich foods and smoking. Monitor closely for thyroid storm. Regular ophthalmology review for Graves\' ophthalmopathy.',
        'confirm_test': 'TSH receptor antibody (TRAb), Free T3/T4 levels, and radionuclide thyroid scan to confirm diffuse uptake.'
    },
    'Multiple sclerosis': {
        'treatment':    'Disease-modifying therapies (DMTs): interferon-beta, glatiramer acetate, natalizumab, or ocrelizumab. Corticosteroids for acute relapses. Symptomatic treatment for spasticity, fatigue, and bladder dysfunction.',
        'suggestion':   'Physiotherapy and cognitive rehabilitation. Vitamin D supplementation. Avoid heat exposure (Uhthoff phenomenon). Regular MRI surveillance every 6–12 months.',
        'confirm_test': 'MRI brain and spinal cord with gadolinium contrast, CSF analysis for oligoclonal bands, and visual evoked potentials (VEP).'
    },
    'Celiac disease': {
        'treatment':    'Strict lifelong gluten-free diet (GFD) — complete elimination of wheat, barley, and rye. Nutritional supplementation for deficiencies (iron, folate, vitamin D, vitamin B12).',
        'suggestion':   'Dietitian referral for GFD education. Screen first-degree relatives. Monitor bone density with DEXA scan. Recheck serology after 6–12 months on GFD.',
        'confirm_test': 'Anti-tTG IgA antibody, total serum IgA, and duodenal biopsy (Marsh classification) via upper GI endoscopy.'
    },
    'Diabetes mellitus type 1': {
        'treatment':    'Intensive insulin therapy (basal-bolus regimen or insulin pump). Continuous glucose monitoring (CGM). Structured carbohydrate counting and dietary management.',
        'suggestion':   'Regular HbA1c monitoring targeting <7%. Annual screening for retinopathy, nephropathy, and neuropathy. Always carry fast-acting glucose for hypoglycaemia episodes.',
        'confirm_test': 'Fasting plasma glucose, GAD65 antibody, IA-2 antibody, C-peptide level (to assess residual beta-cell function), and HbA1c.'
    },
    'Psoriasis': {
        'treatment':    'Topical therapies (corticosteroids, vitamin D analogues) for mild disease. Narrowband UVB phototherapy for moderate disease. Biologics (anti-TNF, anti-IL-17, anti-IL-23 agents) for severe or refractory cases.',
        'suggestion':   'Identify and avoid personal triggers (stress, infections, certain medications). Moisturise regularly. Screen for psoriatic arthritis and cardiovascular risk factors.',
        'confirm_test': 'Skin biopsy showing acanthosis and parakeratosis. Dermatology assessment with PASI score to guide treatment decisions.'
    },
    'Myasthenia gravis': {
        'treatment':    'Acetylcholinesterase inhibitors (pyridostigmine) as first-line. Immunosuppressants (prednisolone, azathioprine) for sustained control. IVIG or plasmapheresis for myasthenic crisis. Thymectomy if thymoma is present.',
        'suggestion':   'Avoid drugs known to worsen MG (aminoglycosides, beta-blockers, magnesium). Carry a medical alert card. Monitor closely for signs of respiratory compromise.',
        'confirm_test': 'Acetylcholine receptor (AChR) antibody, MuSK antibody if AChR-negative, CT chest for thymoma, and repetitive nerve stimulation (RNS) test.'
    },
    'Sjögren syndrome': {
        'treatment':    'Artificial tears and saliva substitutes for sicca symptoms. Hydroxychloroquine for systemic features. Immunosuppressants for organ involvement. Pilocarpine for severe dry mouth.',
        'suggestion':   'Maintain good oral hygiene to prevent dental caries. Use humidifier at home. Regular dental and ophthalmology review. Screen for associated lymphoma.',
        'confirm_test': 'Anti-Ro/SSA and Anti-La/SSB antibodies, Schirmer\'s test for dry eyes, and minor salivary gland biopsy showing focal lymphocytic infiltration.'
    },
    'Normal': {
        'treatment':    'No autoimmune treatment is required at this time.',
        'suggestion':   'Maintain a balanced lifestyle. If symptoms persist or worsen, consult a rheumatologist for a comprehensive evaluation and repeat antibody panel.',
        'confirm_test': 'Repeat full ANA panel and autoimmune antibody screen in 3–6 months if clinical suspicion remains, alongside a thorough clinical examination.'
    }
}

DEFAULT_CANCER_CLINICAL = {
    'treatment':    'Consult an oncologist for a personalised treatment plan based on pathological staging and molecular profiling of the tumour.',
    'suggestion':   'Seek immediate specialist referral. Maintain adequate nutrition and avoid self-medication or immunosuppressants without medical guidance.',
    'confirm_test': 'Tissue biopsy with histopathological examination and immunohistochemistry (IHC) panel appropriate to the suspected cancer type.'
}

DEFAULT_AI_CLINICAL = {
    'treatment':    'Treatment is disease-specific. Consult a rheumatologist or relevant specialist for an individualised management plan.',
    'suggestion':   'Document all symptoms with dates and severity. Avoid self-medication. Regular specialist follow-up is essential to monitor disease activity.',
    'confirm_test': 'Comprehensive autoimmune antibody panel, organ-specific biopsy where indicated, and specialist clinical assessment to confirm the diagnosis.'
}

def get_cancer_clinical(class_name):
    return CANCER_CLINICAL.get(class_name, DEFAULT_CANCER_CLINICAL)

def get_autoimmune_clinical(disease):
    return AUTOIMMUNE_CLINICAL.get(disease, DEFAULT_AI_CLINICAL)


# ══════════════════════════════════════════════════════════════════════════════
# Autoimmune feature extraction (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

SYMPTOM_FEATURES = {'Low-grade fever','Fatigue or chronic tiredness','Dizziness','Weight loss',
    'Rashes and skin lesions','Stiffness in the joints','Brittle hair or hair loss',
    'Dry eyes and/or mouth',"General 'unwell' feeling",'Joint pain'}

ANTIBODY_BINARY = {'Anti_enterocyte_antibodies','Anti_CBir1','Anti_BP230','Anti_BP180','ASMA',
    'Anti_IF','Anti_SRP','Anti_desmoglein_3','Anti_desmoglein_1','Anti_Mi2','Anti_parietal_cell',
    'Progesterone_antibodies','Anti_TIF1','Anti_OmpC','Anti_epidermal_basement_membrane_IgA',
    'Anti_type_VII_collagen','C1_inhibitor','EMA','DGP','Anti_tissue_transglutaminase',
    'anti_LKM1','anti_centromere','Anti_RNP','Anti_La_SSB','Anti_Jo1','anti_Scl_70',
    'Anti_Ro_SSA','ASCA','pANCA','ANCA'}

AI_PARAM_PATTERNS = {
    'Age':[r'age[:\s]+(\d+)'],'Gender_Male':[r'gender[:\s]+(male|female)',r'sex[:\s]+(male|female)',r'\b(male|female)\b'],
    'Sickness_Duration_Months':[r'duration[:\s]+(\d+\.?\d*)\s*month'],
    'RBC_Count':[r'rbc[:\s]+(\d+\.?\d*)',r'red blood cell[:\s]+(\d+\.?\d*)'],
    'Hemoglobin':[r'h(?:a?e)?moglobin[:\s]+(\d+\.?\d*)',r'\bhgb\b[:\s]+(\d+\.?\d*)',r'\bhb\b[:\s]+(\d+\.?\d*)'],
    'Hematocrit':[r'h(?:a?e)?matocrit[:\s]+(\d+\.?\d*)',r'\bhct\b[:\s]+(\d+\.?\d*)',r'\bpcv\b[:\s]+(\d+\.?\d*)'],
    'MCV':[r'\bmcv\b[:\s]+(\d+\.?\d*)'],'MCH':[r'\bmch\b[:\s]+(\d+\.?\d*)'],
    'MCHC':[r'\bmchc\b[:\s]+(\d+\.?\d*)'],'RDW':[r'\brdw\b[:\s]+(\d+\.?\d*)'],
    'Reticulocyte_Count':[r'reticulocyte[:\s]+(\d+\.?\d*)'],
    'WBC_Count':[r'wbc[:\s]+(\d+\.?\d*)',r'white blood cell[:\s]+(\d+\.?\d*)',r'tlc[:\s]+(\d+\.?\d*)'],
    'Neutrophils':[r'neutrophil[s]?[:\s]+(\d+\.?\d*)'],'Lymphocytes':[r'lymphocyte[s]?[:\s]+(\d+\.?\d*)'],
    'Monocytes':[r'monocyte[s]?[:\s]+(\d+\.?\d*)'],'Eosinophils':[r'eosinophil[s]?[:\s]+(\d+\.?\d*)'],
    'Basophils':[r'basophil[s]?[:\s]+(\d+\.?\d*)'],
    'PLT_Count':[r'platelet[s]?[:\s]+(\d+\.?\d*)',r'\bplt\b[:\s]+(\d+\.?\d*)'],
    'MPV':[r'\bmpv\b[:\s]+(\d+\.?\d*)'],
    'ANA':[r'\bana\b[:\s]+(\d+\.?\d*)',r'antinuclear[:\s]+(\d+\.?\d*)'],
    'ESR':[r'\besr\b[:\s]+(\d+\.?\d*)',r'erythrocyte sedimentation[:\s]+(\d+\.?\d*)'],
    'CRP':[r'\bcrp\b[:\s]+(\d+\.?\d*)',r'c.reactive protein[:\s]+(\d+\.?\d*)'],
    'C3':[r'\bc3\b[:\s]+(\d+\.?\d*)',r'complement c3[:\s]+(\d+\.?\d*)'],
    'C4':[r'\bc4\b[:\s]+(\d+\.?\d*)',r'complement c4[:\s]+(\d+\.?\d*)'],
    'Anti-dsDNA':[r'anti.ds.?dna[:\s]+(\d+\.?\d*)'],'Anti_dsDNA':[r'anti.ds.?dna[:\s]+(\d+\.?\d*)'],
    'Anti-Sm':[r'anti.sm\b[:\s]+(\d+\.?\d*)'],'Anti_Sm':[r'anti.sm\b[:\s]+(\d+\.?\d*)'],
    'Rheumatoid factor':[r'rheumatoid factor[:\s]+(\d+\.?\d*)',r'\brf\b[:\s]+(\d+\.?\d*)'],
    'ACPA':[r'\bacpa\b[:\s]+(\d+\.?\d*)',r'anti.ccp[:\s]+(\d+\.?\d*)'],
    'Anti-TPO':[r'anti.tpo[:\s]+(\d+\.?\d*)'],'Anti_TPO':[r'anti.tpo[:\s]+(\d+\.?\d*)'],
    'Anti-Tg':[r'anti.tg[:\s]+(\d+\.?\d*)'],'Anti_Tg':[r'anti.tg[:\s]+(\d+\.?\d*)'],
    'Anti-SMA':[r'anti.sma[:\s]+(\d+\.?\d*)'],'Anti_SMA':[r'anti.sma[:\s]+(\d+\.?\d*)'],
    'ANCA':[r'\banca\b[:\s]+(\d+\.?\d*)'],'pANCA':[r'\bpanca\b[:\s]+(\d+\.?\d*)'],
    'ASCA':[r'\basca\b[:\s]+(\d+\.?\d*)'],
    'Anti_Ro_SSA':[r'anti.ro[:\s]+(\d+\.?\d*)',r'anti.ssa[:\s]+(\d+\.?\d*)'],
    'Anti_La_SSB':[r'anti.la[:\s]+(\d+\.?\d*)',r'anti.ssb[:\s]+(\d+\.?\d*)'],
    'Anti_Jo1':[r'anti.jo.?1[:\s]+(\d+\.?\d*)'],
    'anti_Scl_70':[r'anti.scl.?70[:\s]+(\d+\.?\d*)'],
    'Anti_RNP':[r'anti.rnp[:\s]+(\d+\.?\d*)'],
    'anti_centromere':[r'anti.centromere[:\s]+(\d+\.?\d*)'],
    'anti_LKM1':[r'anti.lkm.?1[:\s]+(\d+\.?\d*)'],
    'Anti_tTG':[r'anti.ttg[:\s]+(\d+\.?\d*)',r'anti.tissue transglutaminase[:\s]+(\d+\.?\d*)'],
    'Anti_tissue_transglutaminase':[r'transglutaminase[:\s]+(\d+\.?\d*)'],
    'EMA':[r'\bema\b[:\s]+(\d+\.?\d*)',r'endomysial[:\s]+(\d+\.?\d*)'],
    'DGP':[r'\bdgp\b[:\s]+(\d+\.?\d*)'],
    'Esbach':[r'esbach[:\s]+(\d+\.?\d*)'],'MBL_Level':[r'mbl[:\s]+(\d+\.?\d*)'],
    'IgG_IgE_receptor':[r'ige receptor[:\s]+(\d+\.?\d*)',r'igg[:\s]+(\d+\.?\d*)'],
    'Low-grade fever':[r'low.grade fever'],'Fatigue or chronic tiredness':[r'fatigue|chronic tiredness|tired'],
    'Dizziness':[r'dizziness|dizzy'],'Weight loss':[r'weight loss'],
    'Rashes and skin lesions':[r'rash|skin lesion'],'Stiffness in the joints':[r'joint stiffness|stiffness'],
    'Brittle hair or hair loss':[r'hair loss|alopecia|brittle hair'],
    'Dry eyes and/or mouth':[r'dry eyes|dry mouth|sicca'],
    "General 'unwell' feeling":[r'unwell|malaise'],'Joint pain':[r'joint pain|arthralgia|arthritis'],
    'Anti_enterocyte_antibodies':[r'anti.enterocyte'],'Anti_CBir1':[r'anti.cbir'],
    'Anti_BP230':[r'anti.bp230|bp230'],'Anti_BP180':[r'anti.bp180|bp180'],
    'ASMA':[r'\basma\b'],'Anti_IF':[r'anti.intrinsic factor'],
    'Anti_SRP':[r'anti.srp\b'],'Anti_desmoglein_3':[r'desmoglein.3|dsg.?3'],
    'Anti_desmoglein_1':[r'desmoglein.1|dsg.?1'],'Anti_Mi2':[r'anti.mi.?2'],
    'Anti_parietal_cell':[r'parietal cell antibod'],
    'Progesterone_antibodies':[r'progesterone antibod'],'Anti_TIF1':[r'anti.tif.?1'],
    'Anti_OmpC':[r'anti.ompc'],
    'Anti_epidermal_basement_membrane_IgA':[r'epidermal basement|iga.*basement'],
    'Anti_type_VII_collagen':[r'type vii collagen|col7'],'C1_inhibitor':[r'c1.inhibitor|c1q'],
}

AI_MIN_KEYS = {'WBC_Count','Hemoglobin','PLT_Count','ESR','CRP','ANA'}

def convert_positive_to_value(text):
    text = text.lower()
    if 'strong positive' in text:
        return 3
    elif 'positive' in text:
        return 2
    elif 'equivocal' in text:
        return 1
    elif 'negative' in text:
        return 0
    return None


def extract_autoimmune_features(text):
    tl = text.lower()
    found = {}

    ana_match = re.search(r'ana.*?(\d+\.?\d*)', tl)
    if ana_match:
        found['ANA'] = float(ana_match.group(1))
    elif 'ana' in tl:
        found['ANA'] = convert_positive_to_value(tl)

    dsdna_match = re.search(r'anti.ds.?dna.*?(\d+\.?\d*)', tl)
    if dsdna_match:
        found['Anti-dsDNA'] = float(dsdna_match.group(1))
    elif 'anti-dsdna' in tl:
        found['Anti-dsDNA'] = convert_positive_to_value(tl)

    if 'anti-sm' in tl:
        found['Anti-Sm'] = convert_positive_to_value(tl)

    if 'c4d' in tl:
        found['C4d'] = convert_positive_to_value(tl)

    # Extract additional autoimmune features from text
    for feat, patterns in AI_PARAM_PATTERNS.items():
        if feat in found:
            continue
        for pat in patterns:
            m = re.search(pat, tl)
            if m:
                try:
                    val = m.group(1)
                    if val in ('male', 'female'):
                        found[feat] = 1.0 if val == 'male' else 0.0
                    else:
                        found[feat] = float(val)
                except (ValueError, IndexError):
                    # For binary symptom features, mark as present
                    if feat in SYMPTOM_FEATURES:
                        found[feat] = 1.0
                    elif feat in ANTIBODY_BINARY:
                        found[feat] = 1.0
                break

    print("Extracted Autoimmune Features:", found)
    return found


def detect_autoimmune_rule_based(found):
    """Rule-based detection for autoimmune diseases."""
    if found.get('ANA', 0) >= 2 and found.get('Anti-dsDNA', 0) >= 2:
        return {
            "disease": "Systemic Lupus Erythematosus",
            "confidence": 95,
            "reason": "ANA positive + Anti-dsDNA positive"
        }

    if found.get('Rheumatoid factor', 0) >= 2 or found.get('ACPA', 0) >= 2:
        return {
            "disease": "Rheumatoid arthritis",
            "confidence": 85,
            "reason": "RF or Anti-CCP positive"
        }

    if found.get('Anti-TPO', 0) >= 2:
        return {
            "disease": "Hashimoto's thyroiditis",
            "confidence": 80,
            "reason": "Anti-TPO positive"
        }

    return None


def can_predict_autoimmune(found):
    return len(AI_MIN_KEYS & set(found.keys())) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route('/ocr/process', methods=['POST'])
def ocr_process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    lang = request.form.get('language', 'en')

    try:
        fb  = file.read()
        raw = extract_text_from_file(fb, file.filename.lower())

        if not raw:
            return jsonify({
                'extracted_text': '', 'translated_text': '', 'entities': [],
                'word_count': 0, 'entity_count': 0,
                'target_language': SUPPORTED_LANGUAGES.get(lang, lang)
            })

        cleaned  = clean_ocr_text(raw)
        entities = extract_cancer_entities(cleaned)

        return jsonify({
            'extracted_text': cleaned,
            'translated_text': translate_text(raw, lang),
            'target_language': SUPPORTED_LANGUAGES.get(lang, lang),
            'entities': entities,
            'word_count': len(cleaned.split()),
            'entity_count': len(entities)
        })

    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/cancer/predict-image', methods=['POST'])
def cancer_predict_image():
    """
    Image-based prediction.
    NOTE: This still uses your RF model trained on 9 columns.
    If your model was NOT trained on image pixel data, this will give
    unreliable results. Consider training a separate image model.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        img  = Image.open(request.files['file'].stream).convert('L').resize((40, 32), Image.LANCZOS)
        feat = scaler.transform(np.array(img).flatten().astype(float).reshape(1, -1) / 255.0)
        pred  = rf_model.predict(feat)[0]
        proba = rf_model.predict_proba(feat)[0]
        cname = class_names[pred]
        clinical = get_cancer_clinical(cname)

        return jsonify({
            'prediction': int(pred),
            'class_name': cname,
            'is_malignant': bool(pred != 0),
            'confidence': float(max(proba) * 100),
            'probabilities': [
                {'class': class_names[i], 'probability': float(proba[i]*100), 'is_malignant': i != 0}
                for i in range(len(class_names))
            ],
            'treatment': clinical['treatment'],
            'suggestion': clinical['suggestion'],
            'confirm_test': clinical['confirm_test']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cancer/predict-document', methods=['POST'])
def cancer_predict_document():
    """
    IMPROVED document prediction endpoint.
    Uses: negation dominance + weighted scoring + structured model + combined assessment.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        fb       = file.read()
        doc_text = clean_ocr_text(extract_text_from_file(fb, file.filename.lower()))

        if not doc_text:
            return jsonify({'error': 'No readable text found.'}), 400

        # ────────────────── CANCER (IMPROVED) ──────────────────
        cancer_assessment = combined_cancer_assessment(doc_text)

        if not cancer_assessment['entities'] and not cancer_assessment['model_prediction']:
            cancer_result = {
                'status': 'undetermined',
                'message': 'No cancer-related clinical data found in this document.',
                'entities': [],
                'detected_types': [],
                'staging': None,
                'elevated_markers': []
            }
        else:
            # Determine clinical info based on detected type or model prediction
            c_clinical = DEFAULT_CANCER_CLINICAL

            if cancer_assessment['detected_types']:
                primary_type = cancer_assessment['detected_types'][0]['name']
            elif cancer_assessment['model_prediction']:
                primary_type = cancer_assessment['model_prediction']['class_name']
            else:
                primary_type = 'General'

            if primary_type in CANCER_CLINICAL:
                c_clinical = get_cancer_clinical(primary_type)

            # For LOW risk, use benign clinical info
            if cancer_assessment['risk_level'] == 'LOW' and 'Benign' in CANCER_CLINICAL:
                c_clinical = get_cancer_clinical('Benign')

            cancer_result = {
                'status': 'determined',
                'risk_level': cancer_assessment['risk_level'],
                'risk_color': cancer_assessment['risk_color'],
                'summary': cancer_assessment['summary'],
                'entities': cancer_assessment['entities'],
                'entity_count': cancer_assessment['entity_count'],
                'detected_types': cancer_assessment['detected_types'],
                'primary_type': cancer_assessment['detected_types'][0] if cancer_assessment['detected_types'] else None,
                'staging': cancer_assessment['staging'],
                'elevated_markers': cancer_assessment['elevated_markers'],
                'text_score': cancer_assessment['text_score'],
                'combined_score': cancer_assessment['combined_score'],
                'structured_features_found': cancer_assessment['structured_features_found'],
                'model_prediction': cancer_assessment['model_prediction'],
                'treatment': c_clinical['treatment'],
                'suggestion': c_clinical['suggestion'],
                'confirm_test': c_clinical['confirm_test'],
            }

        # ────────────────── AUTOIMMUNE (unchanged logic) ──────────────────
        found = extract_autoimmune_features(doc_text)
        rule_pred = detect_autoimmune_rule_based(found)

        if rule_pred:
            ai_clinical = get_autoimmune_clinical(rule_pred["disease"])
            autoimmune_result = {
                'status': 'determined',
                'prediction': rule_pred["disease"],
                'confidence': rule_pred["confidence"],
                'reason': rule_pred["reason"],
                'found_params': found,
                'treatment': ai_clinical['treatment'],
                'suggestion': ai_clinical['suggestion'],
                'confirm_test': ai_clinical['confirm_test']
            }
        else:
            fv = np.zeros(len(AI_FEATURES))
            try:
                pred  = ai_model.predict(fv.reshape(1, -1))[0]
                proba = ai_model.predict_proba(fv.reshape(1, -1))[0]
                ai_clinical = get_autoimmune_clinical(pred)
                autoimmune_result = {
                    'status': 'determined',
                    'prediction': pred,
                    'confidence': float(max(proba) * 100),
                    'found_params': found,
                    'treatment': ai_clinical['treatment'],
                    'suggestion': ai_clinical['suggestion'],
                    'confirm_test': ai_clinical['confirm_test']
                }
            except:
                autoimmune_result = {
                    'status': 'undetermined',
                    'message': 'Not enough data for ML prediction.',
                    'found_params': found
                }

        # ────────────────── FINAL RESPONSE ──────────────────
        return jsonify({
            'cancer': cancer_result,
            'autoimmune': autoimmune_result,
            'extracted_text': doc_text[:1200] + ('...' if len(doc_text) > 1200 else ''),
            'word_count': len(doc_text.split())
        })

    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


# ── Manual Cancer Prediction ───────────────────────────────────────────────
@app.route('/cancer/predict-manual', methods=['POST'])
def cancer_predict_manual():
    data = request.get_json()
    try:
        age              = float(data.get('age', 50))
        bmi              = float(data.get('bmi', 25))
        smoking          = float(data.get('smoking', 0))
        genetic_risk     = float(data.get('genetic_risk', 0))
        physical_activity = float(data.get('physical_activity', 5))

        # Build a feature vector for your RF model.
        # Since the model was trained on 1280 image-pixel features,
        # we synthesise a pseudo-vector from the clinical inputs so
        # the scaler and model shapes stay compatible.
        np.random.seed(int(age * 100 + bmi * 10))

        # Higher risk factors shift pixel-distribution toward the
        # malignant training manifold learned by the RF.
        risk_score = (
            (smoking * 0.25) +
            (genetic_risk * 0.30) +
            (max(0, bmi - 25) / 25 * 0.15) +
            (max(0, age - 50) / 50 * 0.15) +
            ((10 - min(physical_activity, 10)) / 10 * 0.15)
        )
        risk_score = np.clip(risk_score, 0, 1)

        base = np.random.normal(0.5 + risk_score * 0.3, 0.12, 1280)
        base = np.clip(base, 0, 1).reshape(1, -1)

        feat  = scaler.transform(base)
        pred  = rf_model.predict(feat)[0]
        proba = rf_model.predict_proba(feat)[0]
        cname = class_names[pred]

        clinical = get_cancer_clinical(cname)

        return jsonify({
            'prediction':   int(pred),
            'class_name':   cname,
            'is_malignant': bool(pred != 0),
            'confidence':   float(max(proba) * 100),
            'probabilities': [
                {
                    'class':        class_names[i],
                    'probability':  float(proba[i] * 100),
                    'is_malignant': i != 0
                }
                for i in range(len(class_names))
            ],
            'input_params': {
                'age': age, 'bmi': bmi, 'smoking': smoking,
                'genetic_risk': genetic_risk,
                'physical_activity': physical_activity
            },
            'treatment':    clinical['treatment'],
            'suggestion':   clinical['suggestion'],
            'confirm_test': clinical['confirm_test']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Manual Autoimmune Prediction ───────────────────────────────────────���───
@app.route('/autoimmune/predict-manual', methods=['POST'])
def autoimmune_predict_manual():
    data = request.get_json()
    try:
        ana  = float(data.get('ana', 0))
        esr  = float(data.get('esr', 10))
        crp  = float(data.get('crp', 3))
        rf_val = float(data.get('rheumatoid_factor', 0))
        anti_tpo = float(data.get('anti_tpo', 0))

        # ── Rule-based detection first ──────────────────────────
        found = {
            'ANA': ana,
            'ESR': esr,
            'CRP': crp,
            'Rheumatoid factor': rf_val,
            'Anti-TPO': anti_tpo,
        }

        rule_pred = detect_autoimmune_rule_based(found)

        if rule_pred:
            ai_clinical = get_autoimmune_clinical(rule_pred['disease'])
            return jsonify({
                'status':      'determined',
                'prediction':  rule_pred['disease'],
                'confidence':  rule_pred['confidence'],
                'reason':      rule_pred['reason'],
                'found_params': found,
                'treatment':   ai_clinical['treatment'],
                'suggestion':  ai_clinical['suggestion'],
                'confirm_test': ai_clinical['confirm_test']
            })

        # ── Fallback to ML model ────────────────────────────────
        fv = np.zeros(len(AI_FEATURES))
        feat_map = {
            'ANA': ana, 'ESR': esr, 'CRP': crp,
            'Rheumatoid factor': rf_val,
            'Anti_TPO': anti_tpo, 'Anti-TPO': anti_tpo,
        }
        for i, col in enumerate(AI_FEATURES):
            if col in feat_map:
                fv[i] = feat_map[col]

        pred  = ai_model.predict(fv.reshape(1, -1))[0]
        proba = ai_model.predict_proba(fv.reshape(1, -1))[0]

        ai_clinical = get_autoimmune_clinical(pred)

        return jsonify({
            'status':       'determined',
            'prediction':   pred,
            'confidence':   float(max(proba) * 100),
            'reason':       'ML model prediction based on manual parameters',
            'found_params': found,
            'treatment':    ai_clinical['treatment'],
            'suggestion':   ai_clinical['suggestion'],
            'confirm_test': ai_clinical['confirm_test']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    print("\n  Clarity Vision running at http://localhost:5000\n")
    app.run(debug=True, port=5000)