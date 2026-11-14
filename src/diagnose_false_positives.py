import pandas as pd
import joblib

model = joblib.load("models/svm.joblib")

test = pd.read_csv("data/processed/test.csv")

X = test["text"].fillna("")
y = test["label"]

predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

results = pd.DataFrame({
    "email": X,
    "actual": y,
    "predicted": predictions,
    "phishing_probability": probabilities
})

false_positives = results[
    (results["actual"] == 0) &
    (results["predicted"] == 1)
]

print("Total false positives:", len(false_positives))

print("\nFALSE POSITIVE EXAMPLES:\n")

for _, row in false_positives.head(5).iterrows():
    print("Email:", str(row["email"])[:500])
    print("Phishing probability:",
          round(row["phishing_probability"] * 100, 2), "%")
    print("-" * 60)