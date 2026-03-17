# Import required libraries
import re
import string
from collections import Counter

# Step 1: Convert text to lowercase
def to_lowercase(text):
    return text.lower()

# Step 2: Remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Step 3: Remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# Step 4: Tokenization (split text into words)
def tokenize(text):
    return text.split()

# Step 5: Remove stopwords
def remove_stopwords(tokens):
    stopwords = {
        "the", "is", "in", "and", "to", "of", "a", "for", "on", "with",
        "that", "this", "it", "as", "are", "was"
    }
    return [word for word in tokens if word not in stopwords]

# Step 6: Word frequency analysis
def word_frequency(tokens):
    return Counter(tokens)

# Main NLP Pipeline Function
def nlp_pipeline(text):
    """
    A reusable NLP pipeline that processes text step-by-step.
    """
    
    # Apply preprocessing steps
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    
    frequency = word_frequency(tokens)
    
    return {
        "clean_tokens": tokens,
        "word_frequency": frequency
    }

# Example usage
sample_text = "AI is transforming the world of technology! AI helps automate tasks and analyze data."

result = nlp_pipeline(sample_text)

print("Tokens after preprocessing:")
print(result["clean_tokens"])

print("\nWord Frequency:")
print(result["word_frequency"])