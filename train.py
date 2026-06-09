import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
data = pd.read_csv("data/indian_names_500.csv")

# Input names
X = data["name"]

# Output labels
y = data["gender"]


# Convert names into numerical features
vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(2, 3)
)


X_features = vectorizer.fit_transform(X)
data["gender"] = data["gender"].str.strip().str.capitalize()
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.25,
    random_state=42
)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy*100:.2f}%")
print("Model trained successfully")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print(X_features.shape)

user_name = input("Enter a name : ")
name_features = vectorizer.transform([user_name])

prediction = model.predict(name_features)

print("Prediction Gender:", prediction[0])