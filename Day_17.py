import re
from collections import Counter
import numpy as np
from scipy.stats import rv_discrete, entropy

def preprocess_text(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def calculate_word_probabilities(words):
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    probabilities = {word: count / total_words for word, count in word_counts.items()}
    return word_counts, probabilities, total_words

def create_distribution(probabilities):
    words = list(probabilities.keys())
    prob_values = list(probabilities.values())
    indices = np.arange(len(words))
    distribution = rv_discrete(name='word_distribution', values=(indices, prob_values))
    return words, prob_values, distribution

def main():
    text = """
    Natural Language Processing allows computers to understand human language.
    Language models analyze text and calculate the probability of word occurrences.
    Probability distributions are very useful in NLP and AI systems.
    """

    words = preprocess_text(text)
    word_counts, probabilities, total_words = calculate_word_probabilities(words)

    print("Total Words:", total_words)
    print("\nWord Frequency Distribution:")
    for word, count in sorted(word_counts.items()):
        print(f"{word:<15} Count = {count:<3} Probability = {probabilities[word]:.4f}")

    words_list, prob_values, distribution = create_distribution(probabilities)

    print("\nSciPy Distribution Analysis:")
    print(f"Mean     : {distribution.mean():.4f}")
    print(f"Variance : {distribution.var():.4f}")
    print(f"Entropy  : {entropy(prob_values, base=2):.4f} bits")

    target_word = "probability"
    if target_word in probabilities:
        print(f"\nProbability of '{target_word}': {probabilities[target_word]:.4f}")
    else:
        print(f"\n'{target_word}' not found in text.")

if __name__ == "__main__":
    main()