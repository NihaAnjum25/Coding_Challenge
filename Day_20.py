import numpy as np
from scipy.spatial.distance import euclidean, cosine

def cosine_similarity(vector_a, vector_b):
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return np.dot(vector_a, vector_b) / denominator

def calculate_similarity_metrics(vector_a, vector_b):
    return {
        "cosine_similarity": cosine_similarity(vector_a, vector_b),
        "euclidean_distance": euclidean(vector_a, vector_b),
        "cosine_distance": cosine(vector_a, vector_b)
    }

def compare_text_embeddings(text1, text2, embedding_dictionary):
    if text1 not in embedding_dictionary or text2 not in embedding_dictionary:
        return None

    vector_a = embedding_dictionary[text1]
    vector_b = embedding_dictionary[text2]

    metrics = calculate_similarity_metrics(vector_a, vector_b)

    return {
        "text_1": text1,
        "text_2": text2,
        "cosine_similarity": round(metrics["cosine_similarity"], 4),
        "euclidean_distance": round(metrics["euclidean_distance"], 4),
        "cosine_distance": round(metrics["cosine_distance"], 4)
    }

def main():
    embedding_dictionary = {
        "artificial intelligence": np.array([0.9, 0.8, 0.7, 0.6]),
        "machine learning": np.array([0.85, 0.75, 0.65, 0.55]),
        "deep learning": np.array([0.88, 0.78, 0.68, 0.58]),
        "cyber security": np.array([0.2, 0.3, 0.4, 0.5]),
        "data science": np.array([0.8, 0.7, 0.6, 0.5])
    }

    comparisons = [
        ("artificial intelligence", "machine learning"),
        ("artificial intelligence", "cyber security"),
        ("machine learning", "deep learning")
    ]

    for text1, text2 in comparisons:
        result = compare_text_embeddings(text1, text2, embedding_dictionary)
        print(result)

if __name__ == "__main__":
    main()