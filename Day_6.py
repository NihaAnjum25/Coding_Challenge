import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords


class WordFrequencyAnalyzer:

    def __init__(self, file_path):
        self.file_path = file_path
        self.stop_words = set(stopwords.words('english'))

    # Load dataset
    def load_dataset(self):
        df = pd.read_csv(self.file_path)
        return df

    # Preprocess text
    def preprocess_text(self, text):
        text = text.lower()                           # convert to lowercase
        text = re.sub(r'[^a-z\s]', '', text)          # remove punctuation and numbers
        words = text.split()                          # tokenize
        words = [w for w in words if w not in self.stop_words]  # remove stopwords
        return words

    # Compute word frequencies
    def compute_frequencies(self, words):
        counter = Counter(words)

        word_array = np.array(list(counter.keys()))
        freq_array = np.array(list(counter.values()))

        return word_array, freq_array

    # Get top N words
    def get_top_words(self, word_array, freq_array, top_n=20):
        indices = np.argsort(freq_array)[::-1][:top_n]

        top_words = word_array[indices]
        top_freq = freq_array[indices]

        return top_words, top_freq

    # Plot results
    def plot_results(self, words, freq):
        plt.figure(figsize=(10,6))
        plt.bar(words, freq)
        plt.xticks(rotation=45)
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.title("Top 20 Most Frequent Words in BBC News Dataset")
        plt.tight_layout()
        plt.show()

    # Complete pipeline
    def run_pipeline(self):

        df = self.load_dataset()

        # Combine all news articles
        text_data = " ".join(df['description'].astype(str))

        words = self.preprocess_text(text_data)

        word_array, freq_array = self.compute_frequencies(words)

        top_words, top_freq = self.get_top_words(word_array, freq_array)

        print("\nTop 20 Most Frequent Words:\n")
        for word, freq in zip(top_words, top_freq):
            print(f"{word} : {freq}")

        self.plot_results(top_words, top_freq)


# MAIN PROGRAM
if __name__ == "__main__":

    file_path = "C:\\Users\\USER\\OneDrive\\Desktop\\projects\\Coding_Challenge\\bbc_news.csv"

    analyzer = WordFrequencyAnalyzer(file_path)
    analyzer.run_pipeline()