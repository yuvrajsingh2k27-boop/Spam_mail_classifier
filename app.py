import streamlit as st
import welcome
import login1
import base64
import pandas as pd
import numpy as np
import nltk
import os
import database
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
st.set_page_config(page_title="Spamsilly", layout="wide")

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Segoe UI', sans-serif;
        }}
        .block-container {{
            background-color: #A1AAB1;
            padding: 3rem 5rem;
            border-radius: 30px;
        }}
        .stTextInput > div > div > input,
        .stTextArea textarea {{
            background-color: #ffffff20;
            color: black;
            border: 1px solid #ccc;
            border-radius: 10px;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #f1f1f1 !important;
        }}
        .info-card {{
            background: rgba(255,255,255,0.05);
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #555;
            border-radius: 10px;
        }}
        </style>
    """, unsafe_allow_html=True)
set_background("Frame2.png")
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("📂 Dashboard Menu")
st.sidebar.markdown("👨‍💻 **Project by: Yuvraj Singh**")
sidebar_option = st.sidebar.radio("Navigation:", [
    "📬 Detect Message", "📸 Gallery", "📊 Model Info", "📚 About", "⚙️ Settings","Signup"])
if sidebar_option == "Signup":
    import signup
    signup.signup()
welcome.show_welcome()
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    login1.login()
    st.stop()

# ---------------------------
# Load and Combine Datasets
# ---------------------------

# Download necessary NLTK resources
# nltk.download('punkt')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
acc=accuracy_score(y_test, y_pred)

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
    return prediction[0]

#-------------------------------
# content of the sidebar buttons
#-------------------------------

st.title("📧 Spam Mail Detection System")

if sidebar_option == "📬 Detect Message":
    st.header("📥 Paste or Type a Message")
    st.markdown(f"<div class='info-card'><strong>🔍 Model Accuracy:</strong> {acc * 100:.2f}%</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-card'><strong>📚 Dataset:</strong> SMS Spam Collection Dataset from UCI / Kaggle (ham/spam labels)</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-card'><strong>🧠 ML Algorithm:</strong> Naive Bayes Classifier (Multinomial)</div>", unsafe_allow_html=True)

    with st.expander("✏️ Open Message Input"):
        user_input = st.text_area("Write or paste your message:")
        if user_input:
            if predict_message(user_input) == 1:
                st.error("🚫 This message is SPAM.")
            else:
                st.success("✅ This message is NOT spam.")

elif sidebar_option == "📸 Gallery":
    st.header("📷 Visual Gallery")
    col1, col2 = st.columns(2)
    with col1:
        st.image("spam_example_1.png", caption="Spam Filtering Visual", use_container_width=True)
    with col2:
        st.image("spam_example_2.png", caption="Model Workflow", use_container_width=True)

elif sidebar_option == "📊 Model Info":
    st.header("📊 ML & Dataset Details")
    st.markdown("""
    - **Model**: Multinomial Naive Bayes  
    - **Library**: Scikit-learn  
    - **Vectorizer**: CountVectorizer  
    - **Split Ratio**: 80% training / 20% testing  
    - **Accuracy**: {:.2f}%  
    """.format(acc * 100))

elif sidebar_option == "📚 About":
    st.header("📚 About This App")
    st.markdown("""
    This is a smart email classifier that uses machine learning to detect spam messages.  
    Developed by Yuvraj using Python, Streamlit, and scikit-learn, it includes:
    - Login system
    - Sidebar navigation
    - Real-time classification
    - Project images and overview
    """)

elif sidebar_option == "⚙️ Settings":
    st.header("⚙️ Settings")
    st.info("This panel is for preview only.")

st.markdown("---")
st.caption("© 2025 | Spamsilly (Spam Detector) | Developed by Yuvraj Singh | Streamlit App")