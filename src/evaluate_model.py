import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("Loading trained MailShield model...")

# Load trained model
model = joblib.load("models/svm.joblib")

# Load independent test dataset
test = pd.read_csv("data/processed/test.csv")

X_test = test["text"].fillna("")
y_test = test["label"]

print("Test emails:", len(X_test))

# Make predictions
print("\nTesting model...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate performance
accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

print("\n========== MAILSHIELD RESULTS ==========")

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC Score:", round(roc_auc, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    target_names=["Safe Email", "Phishing Email"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nEvaluation completed successfully!")