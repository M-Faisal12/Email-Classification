import pandas as pd
import re

# Load dataset
df = pd.read_csv("d.csv")


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
        "congratulations", "selected", "claim now"
    ]

    # ---- BUSINESS KEYWORDS ----
    business_keywords = [
        "invoice", "payment", "meeting", "project",
        "proposal", "contract", "quotation", "client",
        "schedule", "deadline", "report", "budget",
        "purchase order", "quotation"
    ]

    # ---- SPAM ----
    if any(word in text for word in spam_keywords):
        return "spam"

    # ----BUSINESS ----
    elif any(word in text for word in business_keywords):
        return "business"

    # ---- INQUIRY ----
    elif any(word in text for word in [
        "how", "what", "when", "where", "can you", "help", "why"
    ]):
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
df.to_csv("labeled_emails_4class.csv", index=False)

print(df["label"].value_counts())
print("Saved as labeled_emails_4class.csv")