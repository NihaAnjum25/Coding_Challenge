# Text Similarity using Cosine Similarity
# ABTalksOnAI Global Coding Challenge - Day 11

# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Step 1: Define Two Sample Documents
# -------------------------------------------------
document1 = "Natural language processing helps machines understand human language."
document2 = "Machines use NLP techniques to understand and process human language."

documents = [document1, document2]

# -------------------------------------------------
# Step 2: Convert Text into TF-IDF Vectors
# -------------------------------------------------
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

# -------------------------------------------------
# Step 3: Compute Cosine Similarity
# -------------------------------------------------
similarity_matrix = cosine_similarity(tfidf_matrix)

# -------------------------------------------------
# Step 4: Display Results
# -------------------------------------------------
print("\n==============================")
print("Document 1:")
print(document1)

print("\nDocument 2:")
print(document2)

print("\n==============================")
print("Cosine Similarity Matrix")
print("==============================")

print(similarity_matrix)

# Extract similarity score between the two documents
similarity_score = similarity_matrix[0][1]

print("\n==============================")
print("Similarity Score Between Documents")
print("==============================")

print("Cosine Similarity:", round(similarity_score, 3))

# -------------------------------------------------
# Step 5: Interpretation
# -------------------------------------------------
if similarity_score > 0.7:
    print("The documents are highly similar.")
elif similarity_score > 0.4:
    print("The documents are moderately similar.")
else:
    print("The documents are not very similar.")