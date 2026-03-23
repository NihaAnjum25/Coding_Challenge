import numpy as np
from scipy.optimize import minimize

# Example text vectors (could represent embeddings or word-frequency features)
text_vector_1 = np.array([0.8, 0.6, 0.2, 0.9, 0.4])
text_vector_2 = np.array([0.7, 0.5, 0.3, 0.8, 0.6])

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

def cost_function(weights):
    # Apply weights to both vectors
    weighted_vector_1 = text_vector_1 * weights
    weighted_vector_2 = text_vector_2 * weights

    similarity = cosine_similarity(weighted_vector_1, weighted_vector_2)
    
    # Cost = 1 - similarity (minimize cost, maximize similarity)
    return 1 - similarity

# Initial guess for weights
initial_weights = np.ones(len(text_vector_1))

# Bounds to keep weights positive
bounds = [(0.1, 2.0) for _ in range(len(text_vector_1))]

# Run optimization
result = minimize(cost_function, initial_weights, bounds=bounds, method='L-BFGS-B')

# Optimized weights
optimized_weights = result.x

# Final similarity
final_vector_1 = text_vector_1 * optimized_weights
final_vector_2 = text_vector_2 * optimized_weights
final_similarity = cosine_similarity(final_vector_1, final_vector_2)

print("Initial Weights:", initial_weights)
print("Optimized Weights:", optimized_weights)
print("Minimum Cost:", result.fun)
print("Final Cosine Similarity:", final_similarity)