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

# PDF helper
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

SUPPORTED_LANGUAGES = {
    'en':'English','hi':'Hindi','ta':'Tamil','te':'Telugu','ml':'Malayalam',
    'bn':'Bengali','fr':'French','de':'German','es':'Spanish','it':'Italian',
    'pt':'Portuguese','zh-CN':'Chinese','ar':'Arabic','ru':'Russian','ja':'Japanese','ko':'Korean'
}

def translate_text(text, target_lang):
    if target_lang == 'en': return text
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

# Cancer keywords
CANCER_KEYWORDS = ['cancer','malignant','benign','tumor','tumour','carcinoma','lymphoma',
    'leukemia','leukaemia','metastasis','metastatic','biopsy','oncology','chemotherapy',
    'radiation','staging','grade','nodule','lesion','mass','diagnosis','prognosis',
    'pathology','cytology','histology','adenocarcinoma','sarcoma','melanoma','neoplasm',
    'malignancy','remission','recurrence']
MALIGNANT_KWS = {'malignant','cancer','carcinoma','lymphoma','leukemia','leukaemia',
    'metastasis','metastatic','adenocarcinoma','sarcoma','melanoma','neoplasm','malignancy'}
BENIGN_KWS = {'benign','normal','negative','no malignancy','no cancer','clear','unremarkable'}

def extract_cancer_entities(text):
    found, tl = [], text.lower()
    for kw in CANCER_KEYWORDS:
        if kw in tl:
            idx = tl.find(kw)
            ctx = text[max(0,idx-40):min(len(text),idx+len(kw)+40)].strip().replace('\n',' ')
            found.append({'keyword':kw,'context':f'...{ctx}...'})
    return found

def assess_cancer_risk(entities):
    found = {e['keyword'] for e in entities}
    mal   = sum(1 for k in found if k in MALIGNANT_KWS)
    ben   = sum(1 for k in found if k in BENIGN_KWS)
    if not entities:     return None,None,None
    if mal > 0:          return 'HIGH','danger',f'{mal} malignancy indicator(s) found. Specialist review recommended.'
    if ben > 0:          return 'LOW','safe','Benign findings indicated. Routine follow-up advised.'
    return 'MODERATE','warn',f'{len(entities)} cancer-related term(s) present. Further evaluation recommended.'

# ── Clinical info: treatment, suggestion, confirm test ──────────────
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

# Autoimmune feature patterns
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

def extract_autoimmune_features(text):
    tl = text.lower()
    fv = np.zeros(77)
    found, missing = {}, []
    for i, feat in enumerate(AI_FEATURES):
        patterns = AI_PARAM_PATTERNS.get(feat, [])
        if feat == 'Gender_Male':
            for pat in patterns:
                m = re.search(pat, tl)
                if m:
                    vs = m.group(1) if m.lastindex else m.group(0)
                    val = 1.0 if 'male' in vs and 'female' not in vs else 0.0
                    fv[i] = val; found[feat] = 'Male' if val else 'Female'; break
        elif feat in SYMPTOM_FEATURES:
            for pat in patterns:
                if re.search(pat, tl):
                    fv[i] = 1.0; found[feat] = 'Present'; break
        elif feat in ANTIBODY_BINARY:
            for pat in patterns:
                m = re.search(pat, tl)
                if m:
                    snip = tl[m.start():min(len(tl), m.end()+30)]
                    pos  = bool(re.search(r'positive|detected|present|\d+', snip))
                    neg  = bool(re.search(r'negative|not detected|absent', snip))
                    if pos and not neg: fv[i] = 1.0; found[feat] = 'Positive'
                    break
        else:
            for pat in patterns:
                m = re.search(pat, tl)
                if m:
                    try: val = float(m.group(1)); fv[i] = val; found[feat] = val; break
                    except: pass
        if feat not in found: missing.append(feat)
    return fv, found, missing

def can_predict_autoimmune(found):
    return len(AI_MIN_KEYS & set(found.keys())) >= 2

# Routes
@app.route('/')
def index(): return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename): return send_from_directory(STATIC_DIR, filename)

@app.route('/ocr/process', methods=['POST'])
def ocr_process():
    if 'file' not in request.files: return jsonify({'error':'No file uploaded'}),400
    file = request.files['file']; lang = request.form.get('language','en')
    try:
        fb = file.read(); raw = extract_text_from_file(fb, file.filename.lower())
        if not raw: return jsonify({'extracted_text':'','translated_text':'','entities':[],'word_count':0,'entity_count':0,'target_language':SUPPORTED_LANGUAGES.get(lang,lang)})
        cleaned = clean_ocr_text(raw)
        return jsonify({'extracted_text':cleaned,'translated_text':translate_text(raw,lang),'target_language':SUPPORTED_LANGUAGES.get(lang,lang),'entities':extract_cancer_entities(cleaned),'word_count':len(cleaned.split()),'entity_count':len(extract_cancer_entities(cleaned))})
    except RuntimeError as e: return jsonify({'error':str(e)}),500
    except Exception as e: return jsonify({'error':f'Processing failed: {str(e)}'}),500

@app.route('/cancer/predict-image', methods=['POST'])
def cancer_predict_image():
    if 'file' not in request.files: return jsonify({'error':'No file uploaded'}),400
    try:
        img  = Image.open(request.files['file'].stream).convert('L').resize((40,32),Image.LANCZOS)
        feat = scaler.transform(np.array(img).flatten().astype(float).reshape(1,-1)/255.0)
        pred = rf_model.predict(feat)[0]; proba = rf_model.predict_proba(feat)[0]
        cname = class_names[pred]
        clinical = get_cancer_clinical(cname)
        return jsonify({'prediction':int(pred),'class_name':cname,'is_malignant':bool(pred!=0),'confidence':float(max(proba)*100),'probabilities':[{'class':class_names[i],'probability':float(proba[i]*100),'is_malignant':i!=0} for i in range(len(class_names))],'treatment':clinical['treatment'],'suggestion':clinical['suggestion'],'confirm_test':clinical['confirm_test']})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/cancer/predict-document', methods=['POST'])
def cancer_predict_document():
    if 'file' not in request.files: return jsonify({'error':'No file uploaded'}),400
    file = request.files['file']
    try:
        fb       = file.read()
        doc_text = clean_ocr_text(extract_text_from_file(fb, file.filename.lower()))
        if not doc_text: return jsonify({'error':'No readable text found.'}),400

        # Cancer
        cent = extract_cancer_entities(doc_text)
        rl, rc, sm = assess_cancer_risk(cent)
        if rl is None:
            cancer_result = {'status':'undetermined','message':'No cancer-related clinical data found in this document. Cancer prediction cannot be made.','entities':[]}
        else:
            mal_kw = [e['keyword'] for e in cent if e['keyword'] in MALIGNANT_KWS]
            doc_cname = mal_kw[0].capitalize() if mal_kw else 'General'
            c_clinical = get_cancer_clinical(doc_cname) if doc_cname in CANCER_CLINICAL else DEFAULT_CANCER_CLINICAL
            cancer_result = {'status':'determined','risk_level':rl,'risk_color':rc,'summary':sm,'entities':cent,'entity_count':len(cent),'treatment':c_clinical['treatment'],'suggestion':c_clinical['suggestion'],'confirm_test':c_clinical['confirm_test']}

        # Autoimmune
        fv, found, missing = extract_autoimmune_features(doc_text)
        if not can_predict_autoimmune(found):
            autoimmune_result = {'status':'undetermined','message':'Insufficient autoimmune markers found in this document. Autoimmune prediction cannot be made.','found_params':found,'found_count':len(found),'missing_count':len(missing)}
        else:
            pred  = ai_model.predict(fv.reshape(1,-1))[0]
            proba = ai_model.predict_proba(fv.reshape(1,-1))[0]
            top3  = sorted(zip(ai_model.classes_, proba.tolist()), key=lambda x:-x[1])[:3]
            ai_clinical = get_autoimmune_clinical(pred)
            autoimmune_result = {'status':'determined','prediction':pred,'is_normal':bool(pred=='Normal'),'confidence':float(max(proba)*100),'top3':[{'disease':d,'probability':round(p*100,2)} for d,p in top3],'found_params':found,'found_count':len(found),'missing_count':len(missing),'treatment':ai_clinical['treatment'],'suggestion':ai_clinical['suggestion'],'confirm_test':ai_clinical['confirm_test']}

        return jsonify({'cancer':cancer_result,'autoimmune':autoimmune_result,'extracted_text':doc_text[:1200]+('...' if len(doc_text)>1200 else ''),'word_count':len(doc_text.split())})
    except RuntimeError as e: return jsonify({'error':str(e)}),500
    except Exception as e: return jsonify({'error':f'Processing failed: {str(e)}'}),500

@app.route('/cancer/predict-manual', methods=['POST'])
def cancer_predict_manual():
    data = request.get_json()
    try:
        mi,si,tx,cs,ch,sy = float(data.get('mean_intensity',.5)),float(data.get('std_intensity',.1)),float(data.get('texture',.5)),float(data.get('cell_size',.5)),float(data.get('cell_shape',.5)),float(data.get('symmetry',.5))
        np.random.seed(42)
        f = np.clip((np.random.normal(mi,max(si,.001),1280)+np.sin(np.linspace(0,tx*10*np.pi,1280))*.1+np.random.normal(0,(1-ch)*.2,1280)+np.cos(np.linspace(0,(1-sy)*np.pi,1280))*.05)*cs,0,1).reshape(1,-1)
        feat = scaler.transform(f); pred = rf_model.predict(feat)[0]; proba = rf_model.predict_proba(feat)[0]
        cname = class_names[pred]
        clinical = get_cancer_clinical(cname)
        return jsonify({'prediction':int(pred),'class_name':cname,'is_malignant':bool(pred!=0),'confidence':float(max(proba)*100),'probabilities':[{'class':class_names[i],'probability':float(proba[i]*100),'is_malignant':i!=0} for i in range(len(class_names))],'treatment':clinical['treatment'],'suggestion':clinical['suggestion'],'confirm_test':clinical['confirm_test']})
    except Exception as e: return jsonify({'error':str(e)}),500

if __name__ == '__main__':
    print("\n  Clarity Vision running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
