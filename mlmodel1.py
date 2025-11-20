import pandas as pd
import numpy as np
import nltk
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
# Download necessary NLTK resources
# nltk.download('punkt')

# ---------------------------
# Load and Combine Datasets
# ---------------------------

def load_datasets(file_paths):
    combined_df = pd.DataFrame()
    for path in file_paths:
        try:
            df = pd.read_csv(path, encoding='latin-1')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # Try to find label and text columns
        if {'label', 'text'}.issubset(df.columns):
            df = df[['label', 'text']]
        elif df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['label', 'text']
        else:
            print(f"Skipping {path}: not enough columns")
            continue

        # Drop missing rows
        df.dropna(subset=['label', 'text'], inplace=True)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

# Add your dataset file paths here
dataset_paths = [
    "spam_dataset.csv",
    # "emails.csv",
    "email_spam.csv",
    "fake_mail.csv"
]

df = load_datasets(dataset_paths)

# ---------------------------
# Preprocessing
# ---------------------------

# Remove duplicates and empty messages
df.drop_duplicates(inplace=True)
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() > 0]

# Normalize labels
def normalize_label(label):
    label = str(label).strip().lower()
    return 1 if label in ['spam', '1', 'yes', 'junk'] else 0

df['label'] = df['label'].apply(normalize_label)

# ---------------------------
# Feature Extraction
# ---------------------------

X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Bag of Words
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# TF-IDF
tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_counts)
X_test_tfidf = tfidf.transform(X_test_counts)

# ---------------------------
# Model Training
# ---------------------------

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---------------------------
# Evaluation
# ---------------------------

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# Predict a New Message
# ---------------------------

def predict_message(message):
    message = str(message).strip()
    if not message:
        return "Invalid message"
    msg_vec = vectorizer.transform([message])
    msg_tfidf = tfidf.transform(msg_vec)
    prediction = model.predict(msg_tfidf)
    print(prediction[0])
    return "SPAM" if prediction[0] == 1 else "HAM"

# # Example
sample = "get gifts if you want to win more prizes enter the credit card details "
print("\nSample Prediction:", predict_message(sample))
