# 📧 Email Classification System using Decision Tree

## 🚀 Project Overview

This project is a machine learning-based **Email Classification System** that automatically categorizes emails into predefined classes using a **Decision Tree classifier**.

The system processes raw email content, applies NLP preprocessing, and predicts the category of each email.

---

## 🎯 Objective

To build an intelligent system that classifies emails into:

- 📩 Spam  
- 💼 Business  
- ❓ Inquiry  
- 📌 Others  

using a supervised machine learning approach.

---

## 🧠 Machine Learning Approach

- **Model:** Decision Tree Classifier  
- **Feature Extraction:** TF-IDF Vectorization  
- **Text Processing:** spaCy NLP preprocessing  
- **Labeling:** Rule-based dataset creation  

---

## 🏗️ Project Structure

```text
Email-Classification/
│
├── Code/
│   ├── Preprocessing.py
│   ├── Train/
│   │   ├── Label_Email.py
│   │   ├── Model.py
│   │
│   ├── Test/
│   │   └── Test_Model.py
│
├── Data/
│   ├── Dataset.csv
│   ├── labeled_emails.csv
│   ├── Test.csv
│   └── model.pkl
│
├── config.py
├── README.md
```

---

## ⚙️ Workflow

### 1. Data Collection
Raw email dataset stored in CSV format.

### 2. Labeling System
Emails are categorized into 4 classes:
- Spam (marketing/scam emails)
- Business (professional emails)
- Inquiry (questions/support requests)
- Others (general/uncategorized emails)

---

### 3. Text Preprocessing
Using spaCy NLP library:

- Tokenization  
- Lemmatization  
- Stopword removal  
- Text cleaning  

---

### 4. Feature Extraction
Using TF-IDF vectorization from scikit-learn:

- Converts text into numerical feature vectors  

---

### 5. Model Training
- Algorithm: Decision Tree Classifier  
- Trained on labeled dataset  
- Saved using pickle (`model.pkl`)  

---

### 6. Prediction
New emails are:
- Preprocessed  
- Vectorized  
- Classified into one of the 4 categories  

---

## 📊 Example Output

| Email Text | Predicted Label |
|------------|----------------|
| Win a free iPhone now | Spam |
| Please send invoice details | Business |
| How do I reset my password? | Inquiry |
| Hello, hope you're doing well | Others |

---

## 🧪 Model Evaluation

- Accuracy Score  
- Precision / Recall / F1-score  
- Confusion Matrix  

---

## 🛠️ Tech Stack

- Python 🐍  
- spaCy (NLP)  
- scikit-learn (Machine Learning)  
- pandas (Data handling)  
- pickle (Model saving)  

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/your-username/email-classification.git
cd email-classification

2. Install Dependencies
 Pandas,Scikit-learn ,spacy
3. Generate Labels
python Code/Train/Label_Email.py
4. Train Model
python Code/Train/Model.py
5. Test Model
python Code/Test/predict.py
💡 Key Features
Automated email labeling system
NLP-based preprocessing pipeline
Multi-class classification (4 categories)
Modular project structure
Easily extensible for deployment
🔥 Future Improvements
Replace Decision Tree with Random Forest / XGBoost
Gmail API integration for real-time email filtering
FastAPI/Flask deployment
Transformer-based embeddings (BERT)
👨‍💻 Author

Built as a Machine Learning + NLP project demonstrating:

Text classification
NLP preprocessing
End-to-end ML pipeline design
Modular software architecture
