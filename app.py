import re
import io
import csv
from datetime import datetime, timezone

import joblib
import streamlit as st
import torch
from supabase import create_client
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="MailShield | AI Email Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- STYLE --------------------
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

# -------------------- SUPABASE --------------------
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def logout():
    for k in ["access_token", "refresh_token", "user_id", "user_email"]:
        st.session_state.pop(k, None)
    st.rerun()

def current_client():
    client = get_supabase()
    if st.session_state.get("access_token") and st.session_state.get("refresh_token"):
        try:
            response = client.auth.set_session(
                st.session_state["access_token"],
                st.session_state["refresh_token"]
            )
            if getattr(response, "session", None):
                st.session_state["access_token"] = response.session.access_token
                st.session_state["refresh_token"] = response.session.refresh_token
            user = client.auth.get_user()
            if user and user.user:
                st.session_state["user_id"] = user.user.id
                st.session_state["user_email"] = user.user.email
                return client
        except Exception:
            pass
    return None

def auth_screen():
    st.markdown('<div class="hero"><div class="eyebrow">AI EMAIL SECURITY PLATFORM</div><h1>🛡️ MailShield</h1><p>Detect suspicious emails. Understand risk. Keep your analysis organised.</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### Your email security workspace")
        st.markdown("- AI-powered phishing classification\n- ModernBERT and SVM detection engines\n- Private analysis history\n- Search, edit, archive and export reports")
    with right:
        tab1, tab2 = st.tabs(["Login", "Create account"])
        with tab1:
            email = st.text_input("Email address", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", use_container_width=True):
                try:
                    result = get_supabase().auth.sign_in_with_password({"email": email.strip(), "password": password})
                    st.session_state["access_token"] = result.session.access_token
                    st.session_state["refresh_token"] = result.session.refresh_token
                    st.session_state["user_id"] = result.user.id
                    st.session_state["user_email"] = result.user.email
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
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
                        result = get_supabase().auth.sign_up({"email": new_email.strip(), "password": new_password})
                        if result.session:
                            st.session_state["access_token"] = result.session.access_token
                            st.session_state["refresh_token"] = result.session.refresh_token
                            st.session_state["user_id"] = result.user.id
                            st.session_state["user_email"] = result.user.email
                            st.rerun()
                        else:
                            st.success("Account created. Check your email to confirm it, then log in.")
                    except Exception as e:
                        st.error(f"Sign-up failed: {e}")

if not st.secrets.get("SUPABASE_URL") or not st.secrets.get("SUPABASE_KEY"):
    st.error("Supabase settings are missing. Add SUPABASE_URL and SUPABASE_KEY in Streamlit Secrets.")
    st.stop()

if not st.session_state.get("access_token"):
    auth_screen()
    st.stop()

supabase = current_client()
if supabase is None:
    st.warning("Your session expired. Please log in again.")
    logout()

uid = st.session_state["user_id"]

# -------------------- MODEL FUNCTIONS --------------------
@st.cache_resource
def load_svm():
    return joblib.load("models/svm.joblib")

@st.cache_resource
def load_modernbert():
    name = "kalpithaaj/mailshield-modernbert"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    model.eval()
    return tokenizer, model

def predict_svm(email_text):
    model = load_svm()
    pred = int(model.predict([email_text])[0])
    probs = model.predict_proba([email_text])[0]
    return pred, float(probs[1]) * 100

def predict_modernbert(email_text):
    tokenizer, model = load_modernbert()
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    return int(torch.argmax(probs)), float(probs[1].item()) * 100

def indicators(email_text):
    patterns = {
        "Suspicious links": r"https?://|www\.",
        "Urgent language": r"\b(urgent|immediately|suspended|expire|act now)\b",
        "Password requests": r"\b(password|login|credentials|verify account)\b",
        "Payment requests": r"\b(payment|bank details|credit card|transfer money)\b",
        "Sensitive information requests": r"\b(otp|pin|social security|personal information)\b",
    }
    return [label for label, pattern in patterns.items() if re.search(pattern, email_text, re.I)]

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown('<div class="brand">🛡️ MailShield</div>', unsafe_allow_html=True)
    st.caption("AI-powered email security")
    st.write(f"Signed in as **{st.session_state.get('user_email','')}**")
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

if page == "Overview":
    try:
        rows = fetch_history()
        total = len(rows)
        threats = sum(1 for r in rows if r.get("prediction") == "PHISHING")
        archived = len(fetch_history(True))
        a,b,c = st.columns(3)
        for col, label, value in [(a,"TOTAL ANALYSES",total),(b,"PHISHING DETECTED",threats),(c,"ARCHIVED REPORTS",archived)]:
            with col:
                st.markdown(f'<div class="panel"><div class="eyebrow">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        st.subheader("Recent activity")
        for row in rows[:5]:
            st.markdown(f"**{row.get('title') or 'Untitled analysis'}** · {row.get('prediction')} · {row.get('risk_level')} risk")
    except Exception as e:
        st.error(f"Could not load dashboard data: {e}")

elif page == "Email Scanner":
    st.subheader("📩 Analyse an email")
    model_choice = st.radio("Detection model", ["ModernBERT", "SVM"], horizontal=True)
    email_text = st.text_area("Paste email content", height=230, placeholder="Paste the email you want MailShield to analyse...")
    if st.button("Analyse email", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Paste an email first.")
        else:
            with st.spinner(f"Analysing with {model_choice}..."):
                try:
                    pred, probability = predict_modernbert(email_text) if model_choice == "ModernBERT" else predict_svm(email_text)
                    found = indicators(email_text)
                    label = "PHISHING" if pred == 1 else "SAFE"
                    risk = "HIGH" if probability >= 70 else "MEDIUM" if probability >= 40 else "LOW"
                    st.session_state["latest_analysis"] = {
                        "email_text": email_text, "prediction": label, "probability": round(probability,2),
                        "risk_level": risk, "model_used": model_choice, "indicators": found
                    }
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    result = st.session_state.get("latest_analysis")
    if result:
        st.divider()
        st.subheader("📊 Analysis results")
        x,y,z = st.columns(3)
        x.metric("Detection result", result["prediction"])
        y.metric("Phishing probability", f'{result["probability"]:.2f}%')
        z.metric("Risk level", result["risk_level"])
        st.progress(min(max(result["probability"]/100,0),1))
        st.markdown("#### Suspicious indicators")
        if result["indicators"]:
            for item in result["indicators"]:
                st.warning(item)
        else:
            st.success("No basic suspicious indicators found.")
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
                    "indicators": result["indicators"], "notes": notes, "is_archived": False
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
            rows = [r for r in rows if term in " ".join(str(r.get(k,"")) for k in ["title","prediction","model_used","notes","email_preview"]).casefold()]
        st.caption(f"{len(rows)} report(s)")
        for row in rows:
            with st.expander(f'{row.get("title") or "Untitled"} · {row.get("prediction")} · {row.get("risk_level")}'):
                st.write(f'**Date:** {row.get("created_at","")}')
                st.write(f'**Model:** {row.get("model_used","")}')
                st.write(f'**Phishing probability:** {row.get("probability","")}%')
                st.write("**Email preview:**")
                st.write(row.get("email_preview",""))
                with st.form(f'edit_{row["id"]}'):
                    new_title = st.text_input("Title", value=row.get("title") or "")
                    new_notes = st.text_area("Notes", value=row.get("notes") or "")
                    save_edit = st.form_submit_button("Save changes")
                if save_edit:
                    try:
                        supabase.table("analysis_history").update({"title":new_title,"notes":new_notes}).eq("id",row["id"]).eq("user_id",uid).execute()
                        st.success("Report updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Update failed: {e}")
                p,q = st.columns(2)
                with p:
                    button_label = "Restore" if row.get("is_archived") else "Archive"
                    if st.button(button_label, key=f'archive_{row["id"]}'):
                        try:
                            supabase.table("analysis_history").update({"is_archived":not row.get("is_archived",False)}).eq("id",row["id"]).eq("user_id",uid).execute()
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
                            supabase.table("analysis_history").delete().eq("id",row["id"]).eq("user_id",uid).execute()
                            st.session_state.pop(f'confirm_delete_{row["id"]}',None)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
        if rows:
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["title","prediction","probability","risk_level","model_used","notes","email_preview","created_at"])
            writer.writeheader()
            for row in rows:
                writer.writerow({k:row.get(k,"") for k in writer.fieldnames})
            st.download_button("⬇️ Export visible reports as CSV", csv_buffer.getvalue(), "mailshield_history.csv", "text/csv")
    except Exception as e:
        st.error(f"Could not load history: {e}")

elif page == "Settings":
    st.subheader("⚙️ Settings & privacy")
    st.markdown(f"**Account:** {st.session_state.get('user_email','')}")
    st.info("MailShield stores a short email preview and analysis metadata in your account. It does not save the complete email body to Supabase.")
    st.markdown("Each user's history is protected by Supabase Row Level Security. Only the signed-in account should be able to access its own records.")
    if st.button("Log out of MailShield"):
        logout()

st.markdown("---")
st.caption("MailShield · AI-Based Email Phishing Detection & Risk Assessment · Python · Streamlit · ModernBERT · SVM")
