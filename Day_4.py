import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def clean_text(text):
    # Normalize case
    text = text.lower()

    # Remove punctuation
    text = remove_punctuation(text)

    # Remove numbers
    text = remove_numbers(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = remove_stopwords(tokens)

    # Join cleaned tokens
    cleaned_text = " ".join(tokens)

    return cleaned_text


def process_dataset(text_list):
    cleaned_data = []
    for text in text_list:
        cleaned_data.append(clean_text(text))
    return cleaned_data


# Example dataset
dataset = [
    "Hello!!! This is Day-04 of ABTalksOnAI Challenge 2026.",
    "NLP preprocessing removes noise like numbers 123 and punctuation!!!",
    "Machine learning models work better with clean data."
]

cleaned_output = process_dataset(dataset)

for i, text in enumerate(cleaned_output):
    print(f"Cleaned Text {i+1}: {text}")