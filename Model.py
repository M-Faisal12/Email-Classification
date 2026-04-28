import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from Preprocessing import preprocess


# LOAD DATA
df = pd.read_csv("data/labeled_emails.csv")

# PREPROCESS TEXT
df["clean_text"] = df["email_body"].apply(preprocess)

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
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")