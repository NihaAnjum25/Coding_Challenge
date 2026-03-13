# NLP Feature Extraction: Bag-of-Words and TF-IDF
# ABTalksOnAI Global Coding Challenge - Day 10

# Import required libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# -------------------------------------------------
# Step 1: Sample Text Dataset
# -------------------------------------------------
documents = [
    "Natural language processing makes machines understand text",
    "Machine learning and NLP are important for AI",
    "Text data needs preprocessing before machine learning",
    "NLP helps computers understand human language"
]

# -------------------------------------------------
# Step 2: Text Preprocessing (Tokenization + Stopword Removal)
# -------------------------------------------------
stop_words = set(stopwords.words('english'))
processed_docs = []

for doc in documents:
    tokens = word_tokenize(doc.lower())  # Convert to lowercase and tokenize
    filtered_words = [word for word in tokens if word.isalpha() and word not in stop_words]
    processed_docs.append(" ".join(filtered_words))

# Display processed documents
print("\n==============================")
print("Processed Documents")
print("==============================")

for i, doc in enumerate(processed_docs, start=1):
    print(f"Doc {i}: {doc}")

# -------------------------------------------------
# Step 3: Bag-of-Words Representation
# -------------------------------------------------
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(processed_docs)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=bow_vectorizer.get_feature_names_out()
)

print("\n==============================")
print("Bag-of-Words Matrix")
print("==============================")
print(bow_df)

# -------------------------------------------------
# Step 4: TF-IDF Representation
# -------------------------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\n==============================")
print("TF-IDF Matrix")
print("==============================")
print(tfidf_df)