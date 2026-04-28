import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from Preprocessing import preprocess
import os

# LOAD DATA
print("A")
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "Data", "labeled_emails.csv"))

# PREPROCESS TEXT
print("V")
df["clean_text"] = df["Message"].apply(preprocess)

print("K")
X = df["clean_text"]
y = df["label"]
print("J")
# PIPELINE 
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", DecisionTreeClassifier(random_state=42))
])

# TRAIN
model.fit(X, y)

# SAVE MODEL
file_path = os.path.join(
    os.path.dirname(__file__),
    "Data",
    "model.pkl"
)

with open(file_path, "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")