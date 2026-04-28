import pandas as pd
import pickle
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "Code"))
from Preprocessing import preprocess
sys.path.append(BASE_DIR)
from config import LABELLED_PATH, MODEL_PATH
# LOAD DATA
df = pd.read_csv(LABELLED_PATH)

# PREPROCESS TEXT
df["clean_text"] = df["Message"].apply(preprocess)

X = df["clean_text"]
y = df["label"]
# PIPELINE 
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", DecisionTreeClassifier(random_state=42))
])

# TRAIN
model.fit(X, y)

# SAVE MODEL

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")