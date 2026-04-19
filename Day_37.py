from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_sample_documents() -> List[Dict[str, str]]:
    """
    Return a small document collection for the embeddings demo.
    This keeps the example practical and easy to test.
    """
    return [
        {
            "id": "doc_1",
            "title": "Python for Data Analysis",
            "content": "Python is widely used for data analysis, automation, and machine learning workflows.",
        },
        {
            "id": "doc_2",
            "title": "Vector Databases",
            "content": "Vector databases store embeddings and support fast semantic search over high-dimensional data.",
        },
        {
            "id": "doc_3",
            "title": "Chatbots with Memory",
            "content": "A chatbot with memory can maintain conversation context and produce more relevant responses.",
        },
        {
            "id": "doc_4",
            "title": "Recommendation Systems",
            "content": "Recommendation systems use user behavior, item features, and similarity signals to suggest relevant content.",
        },
        {
            "id": "doc_5",
            "title": "Document Embeddings",
            "content": "Embeddings convert text into numerical vectors so machines can compare meaning instead of exact words.",
        },
    ]


def create_vectorizer() -> TfidfVectorizer:
    """
    Create a local text vectorizer.
    TF-IDF is a lightweight way to convert documents into numeric vectors.
    """
    return TfidfVectorizer(stop_words="english")


def build_vector_index(
    documents: List[Dict[str, str]],
) -> tuple[TfidfVectorizer, object, List[Dict[str, str]]]:
    """Convert documents into vector representations for local semantic-style search."""
    texts = [document["content"] for document in documents]
    vectorizer = create_vectorizer()
    document_vectors = vectorizer.fit_transform(texts)
    return vectorizer, document_vectors, documents


def semantic_search(
    query: str,
    vectorizer: TfidfVectorizer,
    document_vectors: object,
    documents: List[Dict[str, str]],
    top_k: int = 3,
) -> List[Dict[str, object]]:
    """
    Convert the query into a vector, compare it with document vectors,
    and return the most relevant matches.
    """
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()

    scored_results = []
    for document, score in zip(documents, similarity_scores):
        scored_results.append(
            {
                "id": document["id"],
                "title": document["title"],
                "content": document["content"],
                "similarity": float(score),
            }
        )

    return sorted(scored_results, key=lambda item: item["similarity"], reverse=True)[:top_k]


def print_results(query: str, results: List[Dict[str, object]]) -> None:
    """Display semantic search results in a readable format."""
    print(f"\nQuery: {query}")
    print("\nTop matching documents:\n")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['title']} ({result['id']})")
        print(f"   Similarity Score: {result['similarity']:.4f}")
        print(f"   Content: {result['content']}\n")


def main() -> None:
    """
    Day 37 mini demo:
    1. Convert a small set of documents into numeric vector representations.
    2. Convert the user query into the same vector space.
    3. Use cosine similarity to find the most relevant documents.
    """
    try:
        documents = load_sample_documents()
        vectorizer, document_vectors, indexed_documents = build_vector_index(documents)

        query = input("Enter a search query: ").strip()
        if not query:
            print("Please enter a non-empty query.")
            return

        results = semantic_search(query, vectorizer, document_vectors, indexed_documents)
        print_results(query, results)
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
