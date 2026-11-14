import pandas as pd
import joblib

model = joblib.load("models/svm.joblib")

test = pd.read_csv("data/processed/test.csv")

X = test["text"].fillna("")
y = test["label"]

predictions = model.predict(X)

results = pd.DataFrame({
    "text": X,
    "actual": y,
    "predicted": predictions
})

false_positives = results[
    (results["actual"] == 0) &
    (results["predicted"] == 1)
]

keywords = [
    "newsletter",
    "unsubscribe",
    "advertisement",
    "offer",
    "promotion",
    "recruitment",
    "marketing",
    "sale",
    "discount"
]

print("FALSE POSITIVE CATEGORY ANALYSIS\n")

for word in keywords:
    count = false_positives["text"].str.contains(
        word, case=False, na=False
    ).sum()

    print(word, ":", count)

print("\nTotal false positives:", len(false_positives))