import numpy as np

# -----------------------------
# Step 1: Create Embeddings
# -----------------------------
# Example: 3 text embeddings (each of size 5)
embeddings = np.array([
    [0.2, 0.5, 0.1, 0.7, 0.9],
    [0.8, 0.1, 0.3, 0.4, 0.2],
    [0.6, 0.9, 0.5, 0.3, 0.8]
])

print("Original Embeddings:\n", embeddings)


# -----------------------------
# Step 2: Normalize Embeddings
# -----------------------------
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized_embeddings = embeddings / norms

print("\nNormalized Embeddings:\n", normalized_embeddings)


# -----------------------------
# Step 3: Cosine Similarity
# -----------------------------
def cosine_similarity(matrix1, matrix2):
    dot_product = np.dot(matrix1, matrix2.T)
    norm1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
    return dot_product / (norm1 * norm2.T)

similarity_matrix = cosine_similarity(embeddings, embeddings)

print("\nCosine Similarity Matrix:\n", similarity_matrix)


# -----------------------------
# Step 4: Find Most Similar Text
# -----------------------------
# Ignore self-similarity
np.fill_diagonal(similarity_matrix, -1)

most_similar = np.argmax(similarity_matrix, axis=1)

print("\nMost Similar Texts:")
for i, idx in enumerate(most_similar):
    print(f"Text {i} is most similar to Text {idx}")


# -----------------------------
# Step 5: Mean Embedding
# -----------------------------
mean_embedding = np.mean(embeddings, axis=0)

print("\nMean Embedding:\n", mean_embedding)