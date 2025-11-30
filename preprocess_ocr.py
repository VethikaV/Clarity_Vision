
import re

def clean_ocr_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-zA-Z0-9:.\- ]', '', text)
    return text.strip()
