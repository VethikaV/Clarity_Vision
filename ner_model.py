
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_cancer_entities(text):
    doc = nlp(text)
    result = []
    for ent in doc.ents:
        if ent.label_ in ["DISEASE", "ORG", "GPE", "PERSON"]:
            result.append({"text": ent.text, "label": ent.label_})
    return result
