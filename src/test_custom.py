import joblib

model = joblib.load("models/svm.joblib")

emails = [
    "Hello, thank you for contacting us. We will respond shortly.",
    "Your order has been successfully placed. Thank you for shopping with us.",
    "Your appointment has been confirmed. Please arrive 10 minutes early.",
    "Your college examination timetable is now available.",
    "Your bank account will be suspended. Click here to verify your password.",
    "Congratulations! You won a free prize. Click here to claim your reward."
]

for email in emails:
    probability = model.predict_proba([email])[0][1] * 100
    prediction = model.predict([email])[0]

    print("\nEmail:", email)
    print("Prediction:", "PHISHING" if prediction == 1 else "SAFE")
    print("Phishing probability:", round(probability, 2), "%")