# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Sample dataset (
texts = [
    "Free money now!!!",
    "Hi, how are you?",
    "Win cash prizes",
    "Let's meet tomorrow",
    "Claim your reward",
    "Are you coming today?"
]

labels = [1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Step 2: Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# Step 4: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Test with new input
new_text = ["Congratulations! You won a free ticket"]
new_vector = vectorizer.transform(new_text)
prediction = model.predict(new_vector)

print("\nNew Text Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")