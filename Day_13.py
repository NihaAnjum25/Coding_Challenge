# Import required libraries
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# Step 1: Convert text to lowercase
def to_lowercase(text):
    return text.lower()


# Step 2: Remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


# Step 3: Remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)


# Step 4: Tokenize text (split into words)
def tokenize(text):
    return text.split()


# Step 5: Remove stopwords
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


# Step 6: Apply stemming
def apply_stemming(tokens):
    return [stemmer.stem(word) for word in tokens]


# Step 7: Join tokens back to text
def join_tokens(tokens):
    return " ".join(tokens)


# Main preprocessing pipeline
def preprocess_text(text):
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = apply_stemming(tokens)
    clean_text = join_tokens(tokens)
    
    return clean_text


# Example dataset
messages = [
    "Congratulations! You won a free lottery ticket. Call now!",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been selected for a $1000 prize.",
    "Don't forget to submit your assignment."
]


# Apply preprocessing
processed_messages = []

for msg in messages:
    clean_msg = preprocess_text(msg)
    processed_messages.append(clean_msg)


# Display results
for i in range(len(messages)):
    print("Original:", messages[i])
    print("Processed:", processed_messages[i])
    print("-" * 50)