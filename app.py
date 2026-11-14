import streamlit as st
import joblib
import re
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="MailShield",
    page_icon="🛡️",
    layout="centered"
)


# --------------------------------
# LOAD MODELS
# --------------------------------

@st.cache_resource
def load_svm():
    return joblib.load("models/svm.joblib")


@st.cache_resource
def load_modernbert():

    path = "models/modernbert"

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True
    )

    model.eval()

    return tokenizer, model


# --------------------------------
# PREDICTION FUNCTIONS
# --------------------------------

def predict_svm(text):

    model = load_svm()

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    return prediction, probabilities[1] * 100


def predict_modernbert(text):

    tokenizer, model = load_modernbert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    prediction = int(torch.argmax(probabilities))

    phishing_probability = float(
        probabilities[1].item()
    ) * 100

    return prediction, phishing_probability


# --------------------------------
# SUSPICIOUS INDICATORS
# --------------------------------

def find_indicators(text):

    indicators = []

    patterns = {
        "Suspicious links": r"https?://|www\.",
        "Urgent language": r"\b(urgent|immediately|suspended|expire|act now)\b",
        "Password requests": r"\b(password|login|credentials|verify account)\b",
        "Payment requests": r"\b(payment|bank details|credit card|transfer money)\b",
        "Sensitive information requests": r"\b(otp|pin|social security|personal information)\b"
    }

    for name, pattern in patterns.items():

        if re.search(pattern, text, re.IGNORECASE):

            indicators.append(name)

    return indicators


# --------------------------------
# CUSTOM STYLING
# --------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f8ff;
}

.title {
    text-align: center;
    color: #173b70;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# WEBSITE HEADER
# --------------------------------

st.markdown(
    '<div class="title">🛡️ MailShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Email Phishing Detection System</div>',
    unsafe_allow_html=True
)

st.write("---")


# --------------------------------
# MODEL SELECTION
# --------------------------------

st.subheader("⚙️ Select Detection Model")

selected_model = st.selectbox(
    "Choose a machine learning model:",
    ["ModernBERT", "SVM"]
)


# --------------------------------
# EMAIL INPUT
# --------------------------------

st.subheader("📩 Analyse Your Email")

email_text = st.text_area(
    "Paste your email content below:",
    height=220,
    placeholder="Paste the email you want MailShield to analyse..."
)


# --------------------------------
# ANALYSE EMAIL
# --------------------------------

if st.button("🔍 Analyse Email", use_container_width=True):

    if not email_text.strip():

        st.warning("Please paste an email before analysing.")

    else:

        try:

            with st.spinner(
                f"MailShield is analysing your email using {selected_model}..."
            ):

                if selected_model == "ModernBERT":

                    prediction, phishing_probability = predict_modernbert(
                        email_text
                    )

                else:

                    prediction, phishing_probability = predict_svm(
                        email_text
                    )

                indicators = find_indicators(email_text)

            # RESULTS

            st.write("---")

            st.subheader("📊 Analysis Results")

            if prediction == 1:

                st.error("🚨 PHISHING EMAIL DETECTED")

            else:

                st.success("✅ EMAIL CLASSIFIED AS SAFE")

            st.metric(
                "Phishing Probability",
                f"{phishing_probability:.2f}%"
            )

            st.progress(
                min(max(float(phishing_probability / 100), 0.0), 1.0)
            )

            # RISK LEVEL

            if phishing_probability >= 70:

                risk = "HIGH"

            elif phishing_probability >= 40:

                risk = "MEDIUM"

            else:

                risk = "LOW"

            st.write(f"### Risk Level: {risk}")

            # INDICATORS

            st.write("#### 🔎 Suspicious Indicators")

            if indicators:

                for indicator in indicators:

                    st.warning(indicator)

            else:

                st.info("No basic suspicious indicators found.")

            st.caption(
                "Model: " + selected_model
            )

            st.caption(
                "Results are model predictions, not a guarantee of email safety. "
                "Always verify suspicious links and requests independently."
            )

        except Exception as e:

            st.error(f"An error occurred: {e}")


# --------------------------------
# FOOTER
# --------------------------------

st.write("---")

st.markdown(
    "<center>MailShield | AI-Based Email Security Project</center>",
    unsafe_allow_html=True
)