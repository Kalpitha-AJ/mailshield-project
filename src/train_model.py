import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("Loading training and validation datasets...")

# Load datasets
train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")

X_train = train["text"].fillna("")
y_train = train["label"]

X_val = val["text"].fillna("")
y_val = val["label"]

print("Training emails:", len(X_train))
print("Validation emails:", len(X_val))

# Create TF-IDF + SVM model
print("\nBuilding TF-IDF + SVM model...")

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english"
    )),
    ("svm", CalibratedClassifierCV(
        estimator=LinearSVC(class_weight="balanced"),
        cv=3
    ))
])

# Train model
print("\nTraining started. Please wait...")

model.fit(X_train, y_train)

# Validate model
print("\nEvaluating model...")

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

print("\nValidation Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_val,
    predictions,
    target_names=["Safe Email", "Phishing Email"]
))

# Save trained model
joblib.dump(model, "models/svm.joblib")

print("\nModel saved successfully!")
print("Location: models/svm.joblib")