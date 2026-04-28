import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "Code"))
from Preprocessing import preprocess
sys.path.append(BASE_DIR)
from config import MODEL_PATH, Test_PATH
# -----------------------------
# LOAD MODEL
# -----------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# -----------------------------
# LOAD TEST DATA
# -----------------------------
df_test = pd.read_csv(Test_PATH)

# -----------------------------
# PREPROCESS TEXT
# -----------------------------
df_test["clean_text"] = df_test["Message"].apply(preprocess)

X_test = df_test["clean_text"]
y_true = df_test["label"]


# -----------------------------
# PREDICT
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# EVALUATION METRICS
# -----------------------------
print("\n=== Accuracy ===")
print(accuracy_score(y_true, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))