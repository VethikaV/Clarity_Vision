from flask import Flask, request, jsonify, render_template
from cancer_predict import predict_cancer
from ner_model import extract_cancer_entities
from preprocess_ocr import clean_ocr_text
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
from deep_translator import GoogleTranslator

app = Flask(__name__)

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English', 'hi': 'Hindi', 'ta': 'Tamil', 'te': 'Telugu',
    'fr': 'French', 'de': 'German', 'es': 'Spanish', 'zh': 'Chinese',
    'ar': 'Arabic', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean'
}

# ✅ Home landing page
@app.route('/')
def home():
    return render_template("home.html")

# ✅ OCR & Translation Page
@app.route('/ocr')
def ocr_page():
    return render_template("ocr.html", languages=SUPPORTED_LANGUAGES)

# ✅ Cancer Prediction Page
@app.route('/predict')
def predict_page():
    return render_template("predict.html")

# ✅ OCR + Translation (PDF/Image)
@app.route('/ocr', methods=['POST'])
def ocr_and_translate():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    filename = file.filename.lower()
    target_lang = request.form.get("language", "en")

    try:
        # Handle PDF or image
        if filename.endswith('.pdf'):
            pages = convert_from_bytes(file.read(), first_page=1, last_page=1)
            image = pages[0]
        else:
            image = Image.open(file.stream)

        # OCR
        raw_text = pytesseract.image_to_string(image).strip()
        if not raw_text:
            return jsonify({
                "error": "No readable text detected.",
                "extracted_text": "",
                "translated_text": "",
                "target_language": target_lang,
                "entities": []
            }), 200

        cleaned_text = clean_ocr_text(raw_text)
        entities = extract_cancer_entities(cleaned_text)

        # Translation
        try:
            translated_text = GoogleTranslator(source='auto', target=target_lang).translate(raw_text)
        except Exception as e:
            translated_text = f"Translation failed: {str(e)}"

        return jsonify({
            "extracted_text": raw_text,
            "translated_text": translated_text,
            "target_language": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
            "entities": entities
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

# ✅ ML Cancer Prediction
@app.route('/predict-cancer', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get("features", [])
    if len(features) != 30:
        return jsonify({"error": "Exactly 30 features required"}), 400

    try:
        prediction = predict_cancer(features)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
