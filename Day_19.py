import numpy as np

def normalize_vector(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector
    return vector / magnitude

def cosine_similarity(vector_a, vector_b):
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return np.dot(vector_a, vector_b) / (magnitude_a * magnitude_b)

def main():
    embedding_a = np.array([3.0, 4.0, 1.0, 2.0])
    embedding_b = np.array([6.0, 8.0, 2.0, 4.0])
    embedding_c = np.array([1.0, 0.0, 2.0, 1.0])

    normalized_a = normalize_vector(embedding_a)
    normalized_b = normalize_vector(embedding_b)
    normalized_c = normalize_vector(embedding_c)

    print("Original Embeddings:")
    print("Embedding A:", embedding_a)
    print("Embedding B:", embedding_b)
    print("Embedding C:", embedding_c)

    print("\nNormalized Embeddings:")
    print("Normalized A:", np.round(normalized_a, 4))
    print("Normalized B:", np.round(normalized_b, 4))
    print("Normalized C:", np.round(normalized_c, 4))

    print("\nMagnitudes:")
    print("||A|| =", round(np.linalg.norm(embedding_a), 4))
    print("||B|| =", round(np.linalg.norm(embedding_b), 4))
    print("||C|| =", round(np.linalg.norm(embedding_c), 4))

    print("\nCosine Similarities:")
    print("A vs B (before normalization):", round(cosine_similarity(embedding_a, embedding_b), 4))
    print("A vs C (before normalization):", round(cosine_similarity(embedding_a, embedding_c), 4))
    print("A vs B (after normalization):", round(cosine_similarity(normalized_a, normalized_b), 4))
    print("A vs C (after normalization):", round(cosine_similarity(normalized_a, normalized_c), 4))

if __name__ == "__main__":
    main()