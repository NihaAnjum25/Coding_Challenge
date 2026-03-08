import nltk
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessingPipeline:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    # Convert text to lowercase
    def to_lowercase(self, text):
        return text.lower()

    # Remove punctuation
    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    def tokenize(self, text):
        return word_tokenize(text)

    # Remove stopwords
    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    # Lemmatization
    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    # Complete preprocessing pipeline
    def preprocess(self, text):
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)


def load_dataset(file_path, column_name):
    df = pd.read_csv(file_path)
    return df[column_name]


def process_dataset(text_series):
    pipeline = TextPreprocessingPipeline()
    processed_text = text_series.apply(pipeline.preprocess)
    return processed_text


if __name__ == "__main__":

    file_path = "C:\\Users\\USER\\OneDrive\\Desktop\\projects\\Coding_Challenge\\bbc_news.csv"
    column_name = "title"

    headlines = load_dataset(file_path, column_name)

    cleaned_headlines = process_dataset(headlines)

    print("\nProcessed Headlines:\n")
    print(cleaned_headlines.head())