
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def load_datasets(file_paths):
    combined_df=pd.DataFrame()
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
        df=combined_df
    return df

def train_model(data):
    
    X = data['label']
    y = data['text']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    vect = CountVectorizer()
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec = vect.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_vec))
    return model, vect, acc

def duplicate(df):
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