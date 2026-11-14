import joblib

model = joblib.load("models/svm.joblib")

tfidf = model.named_steps["tfidf"]
svm = model.named_steps["svm"]

feature_names = tfidf.get_feature_names_out()
classifier = svm.calibrated_classifiers_[0].estimator

coefficients = classifier.coef_[0]

top_phishing = coefficients.argsort()[-20:][::-1]
top_safe = coefficients.argsort()[:20]

print("\nTOP 20 PHISHING-ASSOCIATED WORDS")
for i in top_phishing:
    print(feature_names[i], round(coefficients[i], 4))

print("\nTOP 20 SAFE-ASSOCIATED WORDS")
for i in top_safe:
    print(feature_names[i], round(coefficients[i], 4))