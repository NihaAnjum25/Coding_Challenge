# Sentiment Analysis Project 

import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("IMDB Dataset.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# -------------------------------
# Step 2: Data Preprocessing
# -------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

print("Cleaning text data...")
df['cleaned_review'] = df['review'].apply(clean_text)

# Convert labels to binary (positive=1, negative=0)
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})

# -------------------------------
# Step 3: Feature Extraction
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 5: Train Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 6: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Step 7: Test Custom Input
# -------------------------------
def predict_sentiment(text):
    text = clean_text(text)
    text = vectorizer.transform([text])
    prediction = model.predict(text)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😠"

# Example Test
print("\nCustom Predictions:")
print(predict_sentiment("This movie was fantastic! I loved it."))
print(predict_sentiment("Worst movie ever. Waste of time."))