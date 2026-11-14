import pandas as pd
import joblib

model = joblib.load("models/svm.joblib")

test = pd.read_csv("data/processed/test.csv")

X = test["text"].fillna("")
y = test["label"]

predictions = model.predict(X)

results = pd.DataFrame({
    "actual": y,
    "predicted": predictions
})

print("\nACTUAL EMAIL DISTRIBUTION")
print(results["actual"].value_counts())

print("\nPREDICTED EMAIL DISTRIBUTION")
print(results["predicted"].value_counts())

print("\nPREDICTION PERCENTAGES")
print(
    results["predicted"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nFALSE POSITIVES:",
      ((y == 0) & (predictions == 1)).sum())

print("FALSE NEGATIVES:",
      ((y == 1) & (predictions == 0)).sum())