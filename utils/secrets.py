# ─── utils/secrets.py ─────────────────────────────────────────────────────────
"""
Unified secret loader.
Priority: Streamlit Cloud secrets -> .env file (local) -> environment variable
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_openai_key() -> str | None:
    # 1. Streamlit Cloud secrets
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # 2. .env / OS environment variable
    return os.getenv("OPENAI_API_KEY")
