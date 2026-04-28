import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from Preprocessing import preprocess

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DATA_DIR = os.path.join(BASE_DIR, "Data")

# model_path = os.path.join(DATA_DIR, "model.pkl")
# test_data_path = os.path.join(DATA_DIR, "test_emails.csv")


# -----------------------------
# LOAD MODEL
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# -----------------------------
# LOAD TEST DATA
# -----------------------------
df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), "Data", "Test.csv"))

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