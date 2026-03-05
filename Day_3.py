"""
NLP Text Normalization — Stemming & Lemmatization
Tool: NLTK | Dataset: Product Reviews

Setup:
    pip install nltk
    python -c "import nltk; [nltk.download(x) for x in
               ('punkt_tab','stopwords','wordnet','averaged_perceptron_tagger_eng')]"
"""

import re
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# ── Sample Dataset ────────────────────────────────────────────────────────────

REVIEWS = [
    "The battery life is amazing! It lasted for days without needing a charge.",
    "The shoes were uncomfortable and the sizing was completely wrong.",
    "Absolutely loving this blender. It blends smoothies perfectly every morning.",
    "Great camera quality! The pictures are incredibly clear and sharp.",
    "The laptop runs slowly and crashes frequently. Very disappointed.",
    "These headphones are delivering outstanding sound quality. Highly recommended!",
]

# ── NLP Utilities ─────────────────────────────────────────────────────────────

STOP_WORDS  = set(stopwords.words("english"))
stemmer     = PorterStemmer()
lemmatizer  = WordNetLemmatizer()

def preprocess(text):
    """Clean → tokenize → remove stop-words."""
    text   = re.sub(r"[^a-z\s]", "", text.lower())
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOP_WORDS]

def get_wordnet_pos(tag):
    """Map Treebank POS tag to WordNet tag for accurate lemmatization."""
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)

def stem(tokens):
    return [stemmer.stem(t) for t in tokens]

def lemmatize(tokens):
    return [lemmatizer.lemmatize(t, get_wordnet_pos(p))
            for t, p in nltk.pos_tag(tokens)]

# ── Main ──────────────────────────────────────────────────────────────────────

def process_reviews(reviews):
    print(f"\n{'='*65}")
    print("  NLP TEXT NORMALIZATION — Stemming & Lemmatization")
    print(f"{'='*65}")

    for i, review in enumerate(reviews, 1):
        tokens     = preprocess(review)
        stemmed    = stem(tokens)
        lemmatized = lemmatize(tokens)

        print(f"\n  Review #{i}: {review}")
        print(f"  {'Tokens'    :12}: {tokens}")
        print(f"  {'Stemmed'   :12}: {stemmed}")
        print(f"  {'Lemmatized':12}: {lemmatized}")
        print(f"  {'-'*60}")

    print("\n  Done.\n")

if __name__ == "__main__":
    process_reviews(REVIEWS)