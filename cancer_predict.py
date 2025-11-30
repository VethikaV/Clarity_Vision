import pickle
import numpy as np

with open("cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def predict_cancer(features_list):
    arr = np.array(features_list).reshape(1, -1)
    label = model.predict(arr)[0]
    return label_encoder.inverse_transform([label])[0]
