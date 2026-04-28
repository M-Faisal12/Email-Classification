import pandas as pd
import re
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from config import LABELLED_PATH, DATASET_PATH
# Load dataset
df = pd.read_csv(DATASET_PATH)

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text


# -----------------------------
# LABEL FUNCTION 
# -----------------------------
def tag_email(text):
    text = clean_text(text)

    # ---- SPAM KEYWORDS ----
    spam_keywords = [
    "win", "free", "lottery", "prize", "urgent",
    "cash", "click here", "limited offer", "money back",
    "congratulations", "selected", "claim now",
    "congrats", "offer expires", "act now",
    "exclusive deal", "guaranteed", "risk free",
    "earn money", "make money fast"
]

    # ---- BUSINESS KEYWORDS ----
    business_keywords = [
        "invoice", "payment", "meeting", "project",
        "proposal", "contract", "quotation", "client",
        "schedule", "deadline", "report", "budget",
        "purchase order", "quotation"
    ]
    
    # ---- Inquiry KEYWORDS ----
    inquiry_keywords = [
    "how", "what", "when", "where", "why",
    "can you", "could you", "help", "assist",
    "explain", "details", "clarify", "guide",
    "support", "issue", "problem", "trouble",
    "reset", "login", "access", "forgot",
    "question", "information", "inquiry"
]
    # ---- SPAM ----
    if any(word in text for word in spam_keywords):
        return "spam"

    # ----BUSINESS ----
    elif any(word in text for word in business_keywords):
        return "business"

    # ---- INQUIRY ----
    elif any(word in text for word in inquiry_keywords):
        return "inquiry"

    # ---- OTHERS ----
    else:
        return "others"


# -----------------------------
# APPLY LABELS
# -----------------------------
df["label"] = df["Message"].apply(tag_email)


# -----------------------------
# SAVE DATASET
# -----------------------------
df.to_csv(LABELLED_PATH, index=False)

print(df["label"].value_counts())
print("Saved as labeled_emails.csv")