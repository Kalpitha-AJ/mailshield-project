import csv
import io
import re

import joblib
import numpy as np
import streamlit as st
from supabase import create_client

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="MailShield | AI Email Security", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp { background: radial-gradient(ellipse at 12% 0%,rgba(24,102,145,.25),transparent 38%),linear-gradient(145deg,#07111f,#0a1728 55%,#0b1c30); color:#edf6ff; font-family:'DM Sans',sans-serif; }
header[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1320px;padding-top:1.8rem;padding-bottom:2.5rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;color:#edf6ff!important;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0b1b2d,#0a1625);border-right:1px solid rgba(125,180,220,.18);}
.brand {font-family:'Space Grotesk';font-size:34px;font-weight:700;color:#55d9f2;}
.hero {padding:26px 30px;border:1px solid rgba(85,217,242,.25);border-radius:22px;background:linear-gradient(115deg,rgba(17,48,72,.94),rgba(15,34,55,.88));margin-bottom:22px;}
.panel {background:linear-gradient(145deg,rgba(18,38,59,.96),rgba(12,29,47,.96));border:1px solid rgba(125,180,220,.18);border-radius:18px;padding:20px 22px;margin-bottom:14px;}
.metric-value {font-family:'Space Grotesk';font-size:30px;font-weight:700;color:#f4fbff;}
.eyebrow {color:#6ce6c0;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;}
.stButton button,.stDownloadButton button {border-radius:11px;border:1px solid rgba(85,217,242,.28);background:linear-gradient(100deg,#087e9f,#16a9bd);color:white;font-weight:700;}
.stTextArea textarea,.stTextInput input {background:#0d243b!important;color:white!important;border:1px solid #168eb8!important;border-radius:11px!important;}
</style>
""", unsafe_allow_html=True)

# -------------------- RISK LOGIC (single source of truth) --------------------
# Survey Table III: p < 0.30 Low / Legitimate, 0.30-0.70 Medium / Suspicious, >= 0.70 High / Phishing.
# Both the label and the risk level come from the SAME probability, so they can never disagree.
LOW_CUTOFF, HIGH_CUTOFF = 30.0, 70.0


def classify(probability):
    """probability is a percentage (0-100). Returns (label, risk_level)."""
    if probability < LOW_CUTOFF:
        return "LEGITIMATE", "LOW"
    if probability < HIGH_CUTOFF:
        return "SUSPICIOUS", "MEDIUM"
    return "PHISHING", "HIGH"


# MUST stay identical to clean() in train_lr.py
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


def clean(text):
    return URL_RE.sub(" urltoken ", str(text).lower())


# -------------------- SUPABASE / LOGIN --------------------
def get_client():
    """One client PER USER SESSION. (A cached global client is shared by everyone and mixes up logins.)"""
    if "sb_client" not in st.session_state:
        st.session_state["sb_client"] = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    return st.session_state["sb_client"]


def friendly_error(e):
    msg = str(e).lower()
    if "invalid login" in msg or "invalid credentials" in msg:
        return "Wrong email or password."
    if "not confirmed" in msg:
        return "Please confirm your email first (check your inbox and spam folder), then log in."
    if "rate limit" in msg or "too many" in msg:
        return "Too many attempts. Wait a minute and try again."
    if "network" in msg or "connection" in msg or "timed out" in msg:
        return "Could not reach the server. Check your internet and try again."
    return f"Something went wrong: {e}"


def start_session(result):
    st.session_state["logged_in"] = True
    st.session_state["user_id"] = result.user.id
    st.session_state["user_email"] = result.user.email
    try:  # make sure database calls are sent as this user
        get_client().postgrest.auth(result.session.access_token)
    except Exception:
        pass


def logout():
    try:
        get_client().auth.sign_out()
    except Exception:
        pass
    st.session_state.clear()
    st.rerun()


def auth_screen():
    st.markdown('<div class="hero"><div class="eyebrow">AI EMAIL SECURITY PLATFORM</div><h1>🛡️ MailShield</h1><p>Detect suspicious emails. Understand risk. Keep your analysis organised.</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Your email security workspace")
        st.markdown("- Phishing detection with Low / Medium / High risk\n- Logistic Regression, ModernBERT and SVM engines\n- Explainable red flags\n- Private analysis history with search, archive and export")
    with right:
        tab1, tab2 = st.tabs(["Login", "Create account"])
        with tab1:
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", use_container_width=True):
                if not email.strip() or not password:
                    st.warning("Enter your email and password.")
                else:
                    try:
                        result = get_client().auth.sign_in_with_password(
                            {"email": email.strip(), "password": password})
                        start_session(result)
                        st.rerun()
                    except Exception as e:
                        st.error(friendly_error(e))
        with tab2:
            new_email = st.text_input("Email address", key="signup_email")
            new_password = st.text_input("Create password", type="password", key="signup_password")
            confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
            if st.button("Create account", use_container_width=True):
                if not new_email.strip() or not new_password:
                    st.warning("Enter an email address and password.")
                elif new_password != confirm:
                    st.warning("The passwords do not match.")
                elif len(new_password) < 6:
                    st.warning("Use a password with at least 6 characters.")
                else:
                    try:
                        result = get_client().auth.sign_up({"email": new_email.strip(), "password": new_password})
                        if result.user is not None and not getattr(result.user, "identities", None):
                            st.warning("This email is already registered. Please log in instead.")
                        elif result.session:
                            start_session(result)
                            st.rerun()
                        else:
                            st.success("Account created. Check your email to confirm it, then log in.")
                    except Exception as e:
                        st.error(friendly_error(e))


if not st.secrets.get("SUPABASE_URL") or not st.secrets.get("SUPABASE_KEY"):
    st.error("Supabase settings are missing. Add SUPABASE_URL and SUPABASE_KEY in Streamlit Secrets.")
    st.stop()

if not st.session_state.get("logged_in"):
    auth_screen()
    st.stop()

supabase = get_client()
try:
    _session = supabase.auth.get_session()  # refreshes the token automatically when needed
except Exception:
    _session = None
if _session is None:
    st.session_state.clear()
    st.warning("Your session expired. Please log in again.")
    st.rerun()
try:
    supabase.postgrest.auth(_session.access_token)
except Exception:
    pass

uid = st.session_state["user_id"]


# -------------------- MODELS --------------------
@st.cache_resource
def load_lr():
    return joblib.load("models/lr.joblib")


@st.cache_resource
def load_svm():
    return joblib.load("models/svm.joblib")


@st.cache_resource
def load_modernbert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # imported lazily: faster start
    name = "kalpithaaj/mailshield-modernbert"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    return tokenizer, model


def predict_lr(text):
    """Calibrated LR probability (%) and the words pushing the email towards phishing."""
    bundle = load_lr()
    x = bundle["vectorizer"].transform([clean(text)])
    probability = float(bundle["calibrated"].predict_proba(x)[0, 1]) * 100
    coef = bundle["lr"].coef_[0]
    names = bundle["vectorizer"].get_feature_names_out()
    contrib = x.multiply(coef).tocoo()
    top = sorted(zip(contrib.col, contrib.data), key=lambda t: -t[1])[:6]
    # exp(w) = how many times a word multiplies the odds of phishing (survey Section IV-F)
    flags = [(names[c], float(np.exp(coef[c]))) for c, v in top if v > 0]
    return probability, flags


def predict_svm(text):
    model = load_svm()
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    idx = classes.index(1) if 1 in classes else 1
    return float(probs[idx]) * 100, []


def predict_modernbert(text):
    import torch
    tokenizer, model = load_modernbert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]
    # find the phishing class from the model config instead of assuming index 1
    idx = 1
    for i, label in model.config.id2label.items():
        if any(k in str(label).lower() for k in ("phish", "spam", "malicious", "unsafe")):
            idx = int(i)
            break
    return float(probs[idx].item()) * 100, []


def indicators(text):
    patterns = {
        "Contains links": r"https?://|www\.",
        "Urgent language": r"\b(urgent|immediately|suspended|expire|act now)\b",
        "Password requests": r"\b(password|login|credentials|verify account)\b",
        "Payment requests": r"\b(payment|bank details|credit card|transfer money)\b",
        "Sensitive information requests": r"\b(otp|pin|social security|personal information)\b",
    }
    return [label for label, pattern in patterns.items() if re.search(pattern, text, re.I)]


PREDICTORS = {"Logistic Regression": predict_lr, "ModernBERT": predict_modernbert, "SVM": predict_svm}

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown('<div class="brand">🛡️ MailShield</div>', unsafe_allow_html=True)
    st.caption("AI-powered email security")
    st.write(f"Signed in as **{st.session_state.get('user_email', '')}**")
    page = st.radio("WORKSPACE", ["Overview", "Email Scanner", "Analysis History", "Settings"])
    st.divider()
    if st.button("Log out", use_container_width=True):
        logout()

st.markdown('<div class="hero"><div class="eyebrow">EMAIL THREAT INTELLIGENCE</div><h1>MailShield Security Workspace</h1><p>Analyse email content, assess phishing risk and manage your private reports.</p></div>', unsafe_allow_html=True)


def fetch_history(archived=None):
    q = supabase.table("analysis_history").select("*").eq("user_id", uid).order("created_at", desc=True)
    if archived is not None:
        q = q.eq("is_archived", archived)
    return q.execute().data or []


# -------------------- PAGES --------------------
if page == "Overview":
    try:
        rows = fetch_history()
        total = len(rows)
        threats = sum(1 for r in rows if r.get("prediction") == "PHISHING")
        archived = len(fetch_history(True))
        a, b, c = st.columns(3)
        for col, label, value in [(a, "TOTAL ANALYSES", total), (b, "PHISHING DETECTED", threats), (c, "ARCHIVED REPORTS", archived)]:
            with col:
                st.markdown(f'<div class="panel"><div class="eyebrow">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        st.subheader("Recent activity")
        for row in rows[:5]:
            st.markdown(f"**{row.get('title') or 'Untitled analysis'}** · {row.get('prediction')} · {row.get('risk_level')} risk")
    except Exception as e:
        st.error(f"Could not load dashboard data: {e}")

elif page == "Email Scanner":
    st.subheader("📩 Analyse an email")
    model_choice = st.radio("Detection model", list(PREDICTORS), horizontal=True)
    email_text = st.text_area("Paste email content", height=230,
                              placeholder="Paste the email you want MailShield to analyse...")
    if st.button("Analyse email", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Paste an email first.")
        else:
            with st.spinner(f"Analysing with {model_choice}..."):
                try:
                    probability, flags = PREDICTORS[model_choice](email_text)
                    label, risk = classify(probability)
                    st.session_state["latest_analysis"] = {
                        "email_text": email_text, "prediction": label, "probability": round(probability, 2),
                        "risk_level": risk, "model_used": model_choice,
                        "indicators": indicators(email_text), "flags": flags,
                    }
                except FileNotFoundError:
                    st.error("Model file not found. Make sure models/lr.joblib (and svm.joblib) are in your repo.")
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    result = st.session_state.get("latest_analysis")
    if result:
        st.divider()
        st.subheader("📊 Analysis results")
        x, y, z = st.columns(3)
        x.metric("Detection result", result["prediction"])
        y.metric("Phishing probability", f'{result["probability"]:.2f}%')
        z.metric("Risk level", result["risk_level"])
        st.progress(min(max(result["probability"] / 100, 0), 1))
        st.caption(f"Bands: below {LOW_CUTOFF:.0f}% Low (Legitimate), {LOW_CUTOFF:.0f}-{HIGH_CUTOFF:.0f}% Medium (Suspicious), "
                   f"{HIGH_CUTOFF:.0f}% and above High (Phishing).")

        if result["flags"]:
            st.markdown("#### Why the model flagged this (top words)")
            for word, mult in result["flags"]:
                st.warning(f'**{word}**: multiplies the odds of phishing by about ×{mult:.1f}')

        st.markdown("#### Basic indicators found in the text")
        if result["indicators"]:
            for item in result["indicators"]:
                st.info(item)
        else:
            st.success("No basic indicators found.")
        st.caption(f'Model: {result["model_used"]}. Predictions are not a guarantee of email safety.')

        with st.form("save_analysis_form"):
            title = st.text_input("Report title", value=f'{result["prediction"].title()} email analysis')
            notes = st.text_area("Notes (optional)")
            save = st.form_submit_button("Save to my history")
        if save:
            try:
                supabase.table("analysis_history").insert({
                    "user_id": uid, "title": title.strip() or "Untitled analysis",
                    "email_preview": result["email_text"][:240],
                    "prediction": result["prediction"], "probability": result["probability"],
                    "risk_level": result["risk_level"], "model_used": result["model_used"],
                    "indicators": result["indicators"], "notes": notes, "is_archived": False,
                }).execute()
                st.success("Analysis saved to your private history.")
            except Exception as e:
                st.error(f"Could not save analysis: {e}")

elif page == "Analysis History":
    st.subheader("🗂️ Your analysis history")
    search = st.text_input("Search reports", placeholder="Search title, prediction, model or notes...")
    view = st.radio("Show", ["Active", "Archived"], horizontal=True)
    try:
        rows = fetch_history(view == "Archived")
        if search.strip():
            term = search.casefold()
            rows = [r for r in rows if term in " ".join(
                str(r.get(k, "")) for k in ["title", "prediction", "model_used", "notes", "email_preview"]).casefold()]
        st.caption(f"{len(rows)} report(s)")
        for row in rows:
            with st.expander(f'{row.get("title") or "Untitled"} · {row.get("prediction")} · {row.get("risk_level")}'):
                st.write(f'**Date:** {row.get("created_at", "")}')
                st.write(f'**Model:** {row.get("model_used", "")}')
                st.write(f'**Phishing probability:** {row.get("probability", "")}%')
                st.write("**Email preview:**")
                st.write(row.get("email_preview", ""))
                with st.form(f'edit_{row["id"]}'):
                    new_title = st.text_input("Title", value=row.get("title") or "")
                    new_notes = st.text_area("Notes", value=row.get("notes") or "")
                    save_edit = st.form_submit_button("Save changes")
                if save_edit:
                    try:
                        supabase.table("analysis_history").update({"title": new_title, "notes": new_notes}) \
                            .eq("id", row["id"]).eq("user_id", uid).execute()
                        st.success("Report updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Update failed: {e}")
                p, q = st.columns(2)
                with p:
                    label_btn = "Restore" if row.get("is_archived") else "Archive"
                    if st.button(label_btn, key=f'archive_{row["id"]}'):
                        try:
                            supabase.table("analysis_history").update({"is_archived": not row.get("is_archived", False)}) \
                                .eq("id", row["id"]).eq("user_id", uid).execute()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not update archive status: {e}")
                with q:
                    if st.button("Delete permanently", key=f'delete_{row["id"]}'):
                        st.session_state[f'confirm_delete_{row["id"]}'] = True
                if st.session_state.get(f'confirm_delete_{row["id"]}'):
                    st.warning("This permanently deletes the saved report.")
                    if st.button("Confirm delete", key=f'confirm_{row["id"]}'):
                        try:
                            supabase.table("analysis_history").delete().eq("id", row["id"]).eq("user_id", uid).execute()
                            st.session_state.pop(f'confirm_delete_{row["id"]}', None)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
        if rows:
            buf = io.StringIO()
            fields = ["title", "prediction", "probability", "risk_level", "model_used", "notes", "email_preview", "created_at"]
            writer = csv.DictWriter(buf, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
            st.download_button("⬇️ Export visible reports as CSV", buf.getvalue(), "mailshield_history.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not load history: {e}")

elif page == "Settings":
    st.subheader("⚙️ Settings & privacy")
    st.markdown(f"**Account:** {st.session_state.get('user_email', '')}")
    st.info("MailShield stores a short email preview and analysis metadata in your account. It does not save the complete email body.")
    st.markdown("Each user's history is protected by Supabase Row Level Security.")
    if st.button("Log out of MailShield"):
        logout()

st.markdown("---")
st.caption("MailShield · AI-Based Email Phishing Detection & Risk Assessment · Python · Streamlit · Logistic Regression · ModernBERT · SVM")