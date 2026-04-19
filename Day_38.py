from typing import List, Tuple

import faiss
import numpy as np


def load_sample_documents() -> List[str]:
    """
    Return a small set of sample documents.
    Each document will be represented by a dense NumPy vector.
    """
    return [
        "Python is widely used for data analysis and automation.",
        "FAISS is a library for efficient similarity search over vectors.",
        "Vector databases are useful in semantic search and RAG systems.",
        "Recommendation systems compare user and item embeddings.",
        "Chatbots with memory can provide more context-aware responses.",
    ]


def create_sample_vectors() -> np.ndarray:
    """
    Create example dense vectors for the sample documents.
    These are small numeric embeddings for demonstration purposes.
    """
    vectors = np.array(
        [
            [0.95, 0.10, 0.05, 0.15],
            [0.10, 0.95, 0.20, 0.10],
            [0.15, 0.85, 0.30, 0.20],
            [0.20, 0.30, 0.95, 0.10],
            [0.25, 0.20, 0.15, 0.90],
        ],
        dtype="float32",
    )
    return vectors


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors so inner product behaves like cosine similarity."""
    normalized_vectors = vectors.copy()
    faiss.normalize_L2(normalized_vectors)
    return normalized_vectors


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index for similarity search.
    IndexFlatIP uses inner product, which works well on normalized vectors.
    """
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def search_vectors(
    index: faiss.IndexFlatIP,
    query_vector: np.ndarray,
    documents: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Search the FAISS index and return the top matching documents."""
    query = query_vector.astype("float32").reshape(1, -1)
    normalized_query = normalize_vectors(query)

    scores, indices = index.search(normalized_query, top_k)

    results: List[Tuple[str, float]] = []
    for doc_index, score in zip(indices[0], scores[0]):
        results.append((documents[doc_index], float(score)))
    return results


def create_query_vector() -> np.ndarray:
    """
    Create a query vector similar to vector-search and semantic-search content.
    This helps demonstrate nearest-neighbor retrieval with FAISS.
    """
    return np.array([0.12, 0.92, 0.28, 0.18], dtype="float32")


def main() -> None:
    """
    Day 38 demo:
    1. Create document vectors with NumPy.
    2. Store them in a FAISS index.
    3. Search the index with a query vector.
    4. Retrieve the most similar documents.
    """
    documents = load_sample_documents()
    vectors = create_sample_vectors()
    normalized_vectors = normalize_vectors(vectors)
    index = build_faiss_index(normalized_vectors)

    query_vector = create_query_vector()
    results = search_vectors(index, query_vector, documents, top_k=3)

    print("FAISS Vector Search Demo\n")
    print(f"Total vectors stored in index: {index.ntotal}\n")
    print("Top matching documents:\n")

    for rank, (document, score) in enumerate(results, start=1):
        print(f"{rank}. Score: {score:.4f}")
        print(f"   Document: {document}\n")


if __name__ == "__main__":
    main()
