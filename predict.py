import pickle
from Preprocessing import preprocess
import os

# LOAD MODEL
# model_path = os.path.join(
#     os.path.dirname(__file__),
#     "Data",
#     "model.pkl"
# )

# with open(model_path, "rb") as f:
#     model = pickle.load(f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_email(text):
    clean_text = preprocess(text)
    return model.predict([clean_text])[0]


# TEST
print(predict_email("Congrats! you have won 100$"))