import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "Data")

DATASET_PATH = os.path.join(DATA_DIR, "Dataset.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
LABELLED_PATH = os.path.join(DATA_DIR, "labeled_emails.csv")
Test_PATH = os.path.join(DATA_DIR, "Test.csv")