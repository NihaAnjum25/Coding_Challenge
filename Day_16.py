import numpy as np

# Sample embeddings (each row = a text vector)
embeddings = np.array([
    [0.2, 0.8, 0.5],
    [0.1, 0.9, 0.4],
    [0.7, 0.3, 0.6],
    [0.6, 0.2, 0.7]
])

# Query embedding
query = np.array([0.2, 0.85, 0.45])

# -------------------------------
# 1. Cosine Similarity (Vectorized)
# -------------------------------
def cosine_similarity(query, embeddings):
    dot_product = np.dot(embeddings, query)
    query_norm = np.linalg.norm(query)
    embeddings_norm = np.linalg.norm(embeddings, axis=1)
    
    similarity = dot_product / (embeddings_norm * query_norm)
    return similarity

# -------------------------------
# 2. Euclidean Distance (Vectorized)
# -------------------------------
def euclidean_distance(query, embeddings):
    return np.linalg.norm(embeddings - query, axis=1)

# -------------------------------
# 3. Find Most Similar Embedding
# -------------------------------
def find_most_similar(similarity_scores):
    return np.argmax(similarity_scores)

# -------------------------------
# Run All Operations
# -------------------------------
cos_sim = cosine_similarity(query, embeddings)
euc_dist = euclidean_distance(query, embeddings)

most_similar_index = find_most_similar(cos_sim)

# -------------------------------
# Output
# -------------------------------
print("Cosine Similarities:", cos_sim)
print("Euclidean Distances:", euc_dist)
print("Most Similar Embedding Index:", most_similar_index)