import pandas as pd
import re

# Load dataset
df = pd.read_csv("Dataset.csv")

# -----------------------------
# Clean text 
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    return text

# -----------------------------
# Tag label
# -----------------------------
def tag_email(text):
    text = clean_text(text)

    # ---- SPAM RULES ----
    spam_keywords = [
        "win", "free", "lottery", "prize", "urgent",
        "cash", "click here", "limited offer", "money back"
    ]

    # ---- BUSINESS RULES ----
    business_keywords = [
        "invoice", "payment", "meeting", "project",
        "proposal", "contract", "quotation", "client",
        "schedule", "deadline", "report"
    ]

    # Spam detection
    if any(word in text for word in spam_keywords):
        return "spam"

    # Business detection
    elif any(word in text for word in business_keywords):
        return "business"

    # Default category
    else:
        return "other"


def main():
     df["tag"] = df["Message"].apply(tag_email)
# df.to_csv("labeled_emails.csv", index=False)

# print("Tagging complete. Saved as labeled_emails.csv")
# print(df["tag"].value_counts())
main()
