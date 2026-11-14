import streamlit as st
from supabase import create_client

st.set_page_config(
    page_title="MailShield | Account",
    page_icon="🛡️",
    layout="centered"
)

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

st.title("🛡️ MailShield")
st.subheader("Your Personal Email Security Dashboard")

login_tab, signup_tab = st.tabs(["Login", "Create Account"])

with login_tab:
    st.subheader("Welcome Back")

    with st.form("login_form"):
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")
        login = st.form_submit_button("Login")

        if login:
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })

                st.session_state.user = response.user
                st.success("Login successful! Welcome to MailShield.")

            except Exception as e:
                st.error(f"Login failed: {e}")

with signup_tab:
    st.subheader("Create Your Account")

    with st.form("signup_form"):
        email = st.text_input("Email Address", key="signup_email")
        password = st.text_input(
            "Create Password",
            type="password",
            key="signup_password"
        )
        signup = st.form_submit_button("Create Account")

        if signup:
            try:
                response = supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })

                st.success(
                    "Account created! Check your email if verification is required."
                )

            except Exception as e:
                st.error(f"Signup failed: {e}")
