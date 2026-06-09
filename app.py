from pyexpat import features

from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
data = pd.read_csv("data/indian_names_500.csv")

X = data["name"]
y = data["gender"]

vectorizer = CountVectorizer(analyzer="char", ngram_range=(2,3))
X_features = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y, test_size=0.25, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        name = request.form["name"]

        features = vectorizer.transform([name])

        prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = max(probabilities) * 100   

    return render_template(
    "index.html",
    prediction=prediction,
    confidence=round(confidence, 2) if request.method == "POST" else "",
    name=name if request.method == "POST" else ""
)

if __name__ == "__main__":
    app.run(debug=True)