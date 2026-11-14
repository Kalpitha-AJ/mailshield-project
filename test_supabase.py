import streamlit as st
from supabase import create_client

try:
    supabase = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

    st.success("MailShield connected to Supabase successfully!")
    st.write("Database connection is working.")

except Exception as e:
    st.error(f"Connection failed: {e}")