import re
import io
import csv
from datetime import datetime, timezone

import joblib
import streamlit as st
import torch
from supabase import create_client
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MailShield | AI Email Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #07111f;
        --panel: #101f32;
        --panel2: #13263c;
        --line: rgba(125, 180, 220, .18);
        --cyan: #55d9f2;
        --mint: #6ce6c0;
        --text: #edf6ff;
        --muted: #9db2c8;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(24, 102, 145, .25), transparent 38%),
            radial-gradient(ellipse at 95% 15%, rgba(46, 80, 135, .18), transparent 34%),
            linear-gradient(145deg, #07111f 0%, #0a1728 55%, #0b1c30 100%);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1320px; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; }
    p, label, li { color: #d5e4f2; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1b2d, #0a1625);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: #eaf5ff; }
    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 34px; font-weight: 700; letter-spacing: 1px;
        color: var(--cyan); margin-bottom: 0;
    }
    .tagline { color: var(--muted); font-size: 14px; margin-top: 2px; }
    .eyebrow { color: var(--mint); font-size: 12px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
    .hero {
        padding: 28px 30px; border: 1px solid rgba(85,217,242,.25);
        border-radius: 22px; background: linear-gradient(115deg, rgba(17,48,72,.94), rgba(15,34,55,.88));
        box-shadow: 0 18px 50px rgba(0,0,0,.18); margin-bottom: 22px;
    }
    .hero h1 { font-size: clamp(30px, 4vw, 48px); margin: 7px 0 8px; }
    .hero p { color: #b8cde0; font-size: 16px; margin-bottom: 0; }
    .panel {
        background: linear-gradient(145deg, rgba(18,38,59,.96), rgba(12,29,47,.96));
        border: 1px solid var(--line); border-radius: 18px; padding: 20px 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,.12); margin-bottom: 14px;
    }
    .metric-label { color: #9db6ca; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { color: #f4fbff; font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700; margin-top: 7px; }
    .metric-note { color: #8fa8bf; font-size: 12px; margin-top: 4px; }
    .pill {
        display: inline-block; padding: 5px 10px; border-radius: 999px;
        background: rgba(85,217,242,.12); border: 1px solid rgba(85,217,242,.25);
        color: var(--cyan); font-size: 12px; font-weight: 700;
    }
    .safe { color: #6ce6c0 !important; }
    .danger { color: #ff8296 !important; }
    .muted { color: var(--muted); }
    .stButton button, .stDownloadButton button {
        border-radius: 11px; border: 1px solid rgba(85,217,242,.28);
        background: linear-gradient(100deg, #087e9f, #16a9bd);
        color: white; font-weight: 700; transition: .2s ease;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        border-color: #70e6f8; color: white; box-shadow: 0 0 18px rgba(22,201,232,.22);
    }