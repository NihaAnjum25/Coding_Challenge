from __future__ import annotations

from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass
class SearchResult:
    """Represents one retrieved document and its similarity score."""

    rank: int
    score: float
    document: str


def load_sample_documents() -> List[str]:
    """Return sample documents for semantic search."""
    return [
        "FAISS is optimized for fast nearest-neighbor search on dense vectors.",
        "Semantic search compares query meaning with document embeddings.",
        "RAG combines retrieval with LLM generation to improve factual accuracy.",
        "Recommendation engines use similarity between user and item vectors.",
        "SQL databases are useful for transactional workloads and structured records.",
        "Transformers create contextual embeddings that capture sentence meaning.",
        "Vector databases can store embeddings for documents, images, and audio.",
        "Prompt engineering improves output quality by giving better instructions.",
    ]


def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and return a sentence-transformers embedding model."""
    if SentenceTransformer is None:
        raise ModuleNotFoundError("sentence_transformers is not installed.")
    return SentenceTransformer(model_name)


def generate_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Encode text into float32 embeddings."""
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


class FallbackEmbedder:
    """
    Lightweight fallback when sentence-transformers is unavailable.
    Uses TF-IDF + SVD to create dense vectors for FAISS.
    """

    def __init__(self, n_components: int = 128) -> None:
        # Character n-grams help fallback retrieval stay robust to wording differences.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        tfidf = self.vectorizer.fit_transform(texts)
        components = min(self.svd.n_components, max(2, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=components, random_state=42)
        dense = self.svd.fit_transform(tfidf).astype("float32")
        faiss.normalize_L2(dense)
        self.is_fitted = True
        return dense

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("FallbackEmbedder must be fitted before transform().")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf).astype("float32")
        faiss.normalize_L2(dense)
        return dense


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index using inner product.
    With normalized embeddings, this is equivalent to cosine similarity.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def semantic_search(
    index: faiss.IndexFlatIP,
    model: SentenceTransformer | None,
    documents: List[str],
    query: str,
    top_k: int = 3,
    fallback_embedder: FallbackEmbedder | None = None,
) -> List[SearchResult]:
    """Run semantic search and return ranked results."""
    if model is not None:
        query_embedding = generate_embeddings(model, [query])
    elif fallback_embedder is not None:
        query_embedding = fallback_embedder.transform([query])
    else:
        raise ValueError("No embedder available for query encoding.")

    scores, indices = index.search(query_embedding, top_k)

    results: List[SearchResult] = []
    for rank, (doc_idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        results.append(
            SearchResult(
                rank=rank,
                score=float(score),
                document=documents[doc_idx],
            )
        )
    return results


def print_results(query: str, results: List[SearchResult]) -> None:
    """Pretty-print retrieval results."""
    print(f"\nQuery: {query}")
    print("Top Matches:")
    for item in results:
        print(f"{item.rank}. Score: {item.score:.4f}")
        print(f"   {item.document}")


def main() -> None:
    """
    Day 39 Challenge: Semantic Search with FAISS
    1. Create sample document corpus
    2. Convert documents into embeddings
    3. Store embeddings in a FAISS index
    4. Retrieve semantically similar documents for user queries
    """
    documents = load_sample_documents()
    model: SentenceTransformer | None = None
    fallback_embedder: FallbackEmbedder | None = None

    if SentenceTransformer is not None:
        model = load_embedding_model()
        doc_embeddings = generate_embeddings(model, documents)
        embedding_backend = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        fallback_embedder = FallbackEmbedder(n_components=64)
        doc_embeddings = fallback_embedder.fit_transform(documents)
        embedding_backend = "TF-IDF + SVD fallback"

    index = build_faiss_index(doc_embeddings)

    queries = [
        "How do we find similar documents using vectors?",
        "What helps LLMs answer questions using external knowledge?",
        "Which systems compare users with similar items?",
    ]

    print("Semantic Search Demo with FAISS")
    print(f"Documents indexed: {index.ntotal}")
    print(f"Embedding dimension: {doc_embeddings.shape[1]}")
    print(f"Embedding backend: {embedding_backend}")

    for query in queries:
        results = semantic_search(
            index=index,
            model=model,
            documents=documents,
            query=query,
            top_k=3,
            fallback_embedder=fallback_embedder,
        )
        print_results(query, results)


if __name__ == "__main__":
    main()
