import pickle
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "Code"))
from Preprocessing import preprocess

sys.path.append(BASE_DIR)
from config import MODEL_PATH


# LOAD MODEL
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict_email(text):
    clean_text = preprocess(text)
    return model.predict([clean_text])[0]


# TEST
text=input("Enter an email message: ")
print(predict_email(text))