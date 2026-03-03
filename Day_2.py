import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def interpret(score: float) -> str:
    if score >= 0.85: return "Very High Similarity"
    if score >= 0.60: return "Moderate Similarity"
    if score >= 0.30: return "Low Similarity"
    return "Very Low / No Similarity"


def compute_text_similarity(doc1: str, doc2: str) -> float:
    cleaned = [preprocess(doc1), preprocess(doc2)]
    matrix = TfidfVectorizer().fit_transform(cleaned)
    score = round(float(cosine_similarity(matrix[0], matrix[1])[0][0]), 4)

    print(f"\n{'='*60}")
    print(f"{'TEXT SIMILARITY REPORT':^60}")
    print(f"{'='*60}")
    print(f"  Document 1 : {doc1[:50]!r}")
    print(f"  Document 2 : {doc2[:50]!r}")
    print(f"  Score      : {score}  |  {interpret(score)}")
    print(f"{'='*60}\n")
    return score


def run_demo() -> None:
    pairs = [
        ("Machine learning is a subset of artificial intelligence.",
         "AI encompasses machine learning and deep learning techniques."),
        ("The cat sat on the mat.",
         "Dogs love to play in the park."),
        ("Python is a versatile programming language used in data science.",
         "Python programming is widely adopted for data analysis and machine learning."),
    ]
    print("\n" + "="*60)
    print("  DEMO MODE — Three Sample Document Pairs")
    print("="*60)
    for i, (d1, d2) in enumerate(pairs, 1):
        print(f"\n  -- Pair {i} --")
        compute_text_similarity(d1, d2)


if __name__ == "__main__":
    print("\n  [1] Interactive   [2] Demo")
    choice = input("  Your choice: ").strip()

    if choice == "1":
        doc1 = input("\n  Document 1:\n  > ").strip()
        doc2 = input("  Document 2:\n  > ").strip()
        if doc1 and doc2:
            compute_text_similarity(doc1, doc2)
        else:
            print("  Both documents must be non-empty.")
    else:
        run_demo()