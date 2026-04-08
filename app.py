# ─── app.py ───────────────────────────────────────────────────────────────────
"""TradeXAI — Agentic AI Stock Market Platform"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from agents.data_agent import DataAgent
from agents.technical_agent import TechnicalAgent
from agents.sentiment_agent import SentimentAgent
from agents.prediction_agent import PredictionAgent
from agents.decision_agent import DecisionAgent
from agents.explanation_agent import ExplanationAgent
from agents.portfolio_agent import PortfolioAgent
from agents.alert_agent import AlertAgent
from agents.chat_agent import ChatAgent
from auth import login, register
from config import DEFAULT_STOCKS, RISK_LEVELS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TradeXAI — AI Stock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Images ─────────────────────────────────────────────────────────────────────
IMG_HERO     = "https://tse2.mm.bing.net/th/id/OIP.6-_YSSPlSHHAOwLApii-kQHaDa?pid=Api&P=0&h=180"
IMG_FEATURES = "https://tse2.mm.bing.net/th/id/OIP.vrCjSLrwE_7yQ4_o1lT97QHaEo?pid=Api&P=0&h=180"
IMG_ABOUT    = "https://tse4.mm.bing.net/th/id/OIP.tWz74LT10zeAZydj4-Sy9QHaEJ?pid=Api&P=0&h=180"

# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "home",
        "logged_in": False,
        "user": None,
        "dash_page": "analysis",
        "chat_history": [],
        "last_result": None,
        "last_ticker": "AAPL",
        "risk_level": "Moderate",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Navigation helpers ─────────────────────────────────────────────────────────
def go(page: str):
    st.session_state.page = page
    st.rerun()

def go_dash(sub: str):
    st.session_state.dash_page = sub
    st.rerun()

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.chat_history = []
    go("home")

# ── Agent cache ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agents():
    return {
        "data": DataAgent(), "technical": TechnicalAgent(),
        "sentiment": SentimentAgent(), "prediction": PredictionAgent(),
        "decision": DecisionAgent(), "explanation": ExplanationAgent(),
        "portfolio": PortfolioAgent(), "alert": AlertAgent(),
        "chat": ChatAgent(),
    }

agents = get_agents()

@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(ticker: str, risk_level: str) -> dict:
    data = agents["data"].run(ticker)
    if data["status"] != "ok":
        return {"status": "error", "ticker": ticker}
    tech  = agents["technical"].run(data)
    sent  = agents["sentiment"].run(data)
    pred  = agents["prediction"].run(data)
    dec   = agents["decision"].run(tech, sent, pred, risk_level, ticker)
    expl  = agents["explanation"].run(dec, tech, sent, pred, ticker)
    return {"status":"ok","ticker":ticker,"data":data,"technical":tech,
            "sentiment":sent,"prediction":pred,"decision":dec,"explanation":expl}

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Hide default Streamlit chrome on home/auth pages ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #f59e0b; border-radius: 3px; }

/* ── Buttons ── */
.btn-gold {
    display: inline-block; padding: 14px 38px; background: linear-gradient(135deg,#f59e0b,#d97706);
    color: #0a0e1a; font-weight: 700; font-size: 1rem; border-radius: 50px;
    text-decoration: none; cursor: pointer; border: none; transition: all .3s;
    box-shadow: 0 4px 20px rgba(245,158,11,.4);
}
.btn-gold:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(245,158,11,.6); }
.btn-outline {
    display: inline-block; padding: 13px 36px; background: transparent;
    color: #f59e0b; font-weight: 700; font-size: 1rem; border-radius: 50px;
    border: 2px solid #f59e0b; cursor: pointer; transition: all .3s;
}
.btn-outline:hover { background: #f59e0b; color: #0a0e1a; }

/* ── Hero ── */
.hero-section {
    position: relative; min-height: 100vh; display: flex; align-items: center;
    justify-content: center; text-align: center; overflow: hidden;
    background: #0a0e1a;
}
.hero-bg {
    position: absolute; inset: 0; z-index: 0;
    background-image: url('""" + IMG_HERO + """');
    background-size: cover; background-position: center;
    filter: brightness(0.25) saturate(1.2);
}
.hero-overlay {
    position: absolute; inset: 0; z-index: 1;
    background: linear-gradient(135deg, rgba(10,14,26,.85) 0%, rgba(245,158,11,.08) 100%);
}
.hero-content { position: relative; z-index: 2; padding: 2rem; max-width: 900px; }
.hero-badge {
    display: inline-block; padding: 6px 20px; background: rgba(245,158,11,.15);
    border: 1px solid rgba(245,158,11,.4); border-radius: 50px;
    color: #f59e0b; font-size: .85rem; font-weight: 600; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 1.5rem;
}
.hero-title {
    font-size: clamp(2.5rem, 6vw, 5rem); font-weight: 900; line-height: 1.1;
    color: #ffffff; margin: 0 0 1rem;
    text-shadow: 0 0 60px rgba(245,158,11,.3);
}
.hero-title span { color: #f59e0b; }
.hero-sub {
    font-size: clamp(1rem, 2vw, 1.3rem); color: rgba(255,255,255,.7);
    margin: 0 0 2.5rem; line-height: 1.6; max-width: 600px; margin-left: auto; margin-right: auto;
}
.hero-stats {
    display: flex; gap: 2rem; justify-content: center; margin: 3rem 0 2.5rem;
    flex-wrap: wrap;
}
.hero-stat { text-align: center; }
.hero-stat-val { font-size: 2rem; font-weight: 800; color: #f59e0b; }
.hero-stat-label { font-size: .8rem; color: rgba(255,255,255,.5); letter-spacing: 1px; text-transform: uppercase; }

/* ── Features ── */
.features-section {
    background: #0d1220; padding: 6rem 2rem;
}
.section-title {
    text-align: center; font-size: 2.5rem; font-weight: 800; color: #fff;
    margin-bottom: .5rem;
}
.section-subtitle {
    text-align: center; color: rgba(255,255,255,.5); font-size: 1.1rem;
    margin-bottom: 4rem;
}
.section-title span { color: #f59e0b; }
.features-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem; max-width: 1200px; margin: 0 auto;
}
.feature-card {
    background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px; padding: 2rem; transition: all .3s;
    backdrop-filter: blur(10px);
}
.feature-card:hover {
    border-color: rgba(245,158,11,.4); transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,.3), 0 0 30px rgba(245,158,11,.08);
}
.feature-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.feature-title { font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: .5rem; }
.feature-desc { font-size: .9rem; color: rgba(255,255,255,.5); line-height: 1.6; }

/* ── News Ticker ── */
.ticker-wrapper {
    background: rgba(245,158,11,.08); border-top: 1px solid rgba(245,158,11,.2);
    border-bottom: 1px solid rgba(245,158,11,.2); padding: 1rem 0;
    overflow: hidden; position: relative;
}
.ticker-label {
    position: absolute; left: 0; top: 0; bottom: 0; z-index: 10;
    background: #f59e0b; color: #0a0e1a; font-weight: 800; font-size: .8rem;
    letter-spacing: 1px; padding: 0 1.2rem; display: flex; align-items: center;
}
.ticker-track {
    display: flex; gap: 0; white-space: nowrap;
    animation: ticker-scroll 50s linear infinite;
    padding-left: 120px;
}
.ticker-track:hover { animation-play-state: paused; }
@keyframes ticker-scroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
.ticker-item {
    display: inline-flex; align-items: center; gap: .5rem;
    padding: 0 2rem; font-size: .9rem; color: rgba(255,255,255,.85);
}
.ticker-dot { color: #f59e0b; font-size: 1.2rem; }

/* ── About ── */
.about-section {
    background: #0a0e1a; padding: 6rem 2rem;
    display: flex; align-items: center; gap: 4rem;
    max-width: 1200px; margin: 0 auto;
    flex-wrap: wrap;
}
.about-img {
    flex: 1; min-width: 280px; border-radius: 20px; overflow: hidden;
    box-shadow: 0 30px 60px rgba(0,0,0,.5);
    border: 1px solid rgba(245,158,11,.2);
}
.about-img img { width: 100%; display: block; filter: brightness(.8) saturate(1.2); }
.about-text { flex: 1; min-width: 280px; }
.about-text h2 { font-size: 2.2rem; font-weight: 800; color: #fff; margin-bottom: 1rem; }
.about-text h2 span { color: #f59e0b; }
.about-text p { color: rgba(255,255,255,.6); line-height: 1.8; margin-bottom: 1.5rem; }
.about-pill {
    display: inline-block; padding: 6px 16px; background: rgba(245,158,11,.1);
    border: 1px solid rgba(245,158,11,.3); border-radius: 50px;
    color: #f59e0b; font-size: .8rem; font-weight: 600; margin: 3px;
}

/* ── Auth pages ── */
.auth-wrapper {
    min-height: 100vh; background: #0a0e1a; display: flex;
    align-items: center; justify-content: center; padding: 2rem;
}
.auth-card {
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.1);
    border-radius: 24px; padding: 3rem; width: 100%; max-width: 440px;
    backdrop-filter: blur(20px);
}
.auth-logo { text-align: center; margin-bottom: 2rem; }
.auth-logo .brand { font-size: 2rem; font-weight: 900; color: #f59e0b; }
.auth-logo p { color: rgba(255,255,255,.4); font-size: .9rem; }
.auth-title { font-size: 1.6rem; font-weight: 700; color: #fff; text-align: center; margin-bottom: .5rem; }
.auth-sub { color: rgba(255,255,255,.4); text-align: center; font-size: .9rem; margin-bottom: 2rem; }

/* ── Dashboard ── */
.dash-nav-item {
    width: 100%; text-align: left; padding: 12px 16px; border-radius: 10px;
    border: none; background: transparent; color: rgba(255,255,255,.6);
    font-size: .95rem; cursor: pointer; transition: all .2s; display: flex;
    align-items: center; gap: 10px; font-weight: 500;
}
.dash-nav-item:hover { background: rgba(245,158,11,.1); color: #f59e0b; }
.dash-nav-item.active { background: rgba(245,158,11,.15); color: #f59e0b; font-weight: 700;
    border-left: 3px solid #f59e0b; }

/* ── Decision cards ── */
.dec-buy  { background: rgba(16,185,129,.1); border-left: 5px solid #10b981; padding: 1.5rem; border-radius: 12px; }
.dec-sell { background: rgba(239,68,68,.1);  border-left: 5px solid #ef4444; padding: 1.5rem; border-radius: 12px; }
.dec-hold { background: rgba(245,158,11,.1); border-left: 5px solid #f59e0b; padding: 1.5rem; border-radius: 12px; }
.dec-label { font-size: 2.5rem; font-weight: 900; }

/* ── Metric card ── */
.metric-card {
    background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.08);
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.metric-val  { font-size: 1.6rem; font-weight: 800; color: #f59e0b; }
.metric-label { font-size: .8rem; color: rgba(255,255,255,.4); text-transform: uppercase; letter-spacing: 1px; }

/* ── Chat ── */
.chat-bubble-user { background: rgba(245,158,11,.15); border-radius: 18px 18px 4px 18px;
    padding: 12px 18px; margin: 8px 0; max-width: 75%; margin-left: auto;
    color: #fff; font-size: .95rem; }
.chat-bubble-bot  { background: rgba(255,255,255,.05); border-radius: 18px 18px 18px 4px;
    padding: 12px 18px; margin: 8px 0; max-width: 80%;
    color: rgba(255,255,255,.9); font-size: .95rem; }

/* ── Streamlit overrides for dashboard ── */
.dash-container .block-container { padding: 2rem 2rem 2rem 1rem !important; }
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid rgba(255,255,255,.07) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,.8); }

/* footer bar */
.footer-bar {
    background: #080c18; padding: 2rem; text-align: center;
    color: rgba(255,255,255,.3); font-size: .85rem; border-top: 1px solid rgba(255,255,255,.07);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: news ticker content
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner=False)
def get_ticker_news() -> list[str]:
    headlines = []
    da = DataAgent()
    for t in ["AAPL", "TSLA", "NVDA", "GOOGL", "AMZN", "MSFT"]:
        try:
            news = da.fetch_news(t, max_articles=3)
            for n in news:
                h = n.get("title", "").strip()
                if h:
                    headlines.append(f"{t}: {h}")
        except Exception:
            pass
    if not headlines:
        headlines = [
            "Markets open mixed amid global uncertainty",
            "Tech stocks lead gains in pre-market trading",
            "Fed signals cautious approach to rate decisions",
            "AI sector continues to attract investor interest",
        ]
    return headlines


def render_ticker():
    headlines = get_ticker_news()
    items_html = ""
    doubled = headlines * 2  # duplicate for seamless loop
    for h in doubled:
        items_html += f'<span class="ticker-item"><span class="ticker-dot">◆</span>{h}</span>'
    st.markdown(f"""
    <div class="ticker-wrapper">
        <div class="ticker-label">📡 LIVE NEWS</div>
        <div class="ticker-track">{items_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-bg"></div>
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-badge">🤖 Powered by GPT-4o + ML</div>
            <h1 class="hero-title">Welcome to <span>TradeXAI</span></h1>
            <p class="hero-sub">
                The next-generation Agentic AI platform for stock market intelligence.
                8 specialized AI agents analyze, predict, and explain every market move.
            </p>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-val">8</div>
                    <div class="hero-stat-label">AI Agents</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-val">GPT-4o</div>
                    <div class="hero-stat-label">Intelligence</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-val">Live</div>
                    <div class="hero-stat-label">Market Data</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-val">Free</div>
                    <div class="hero-stat-label">To Start</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero buttons — actual Streamlit buttons styled
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        bcol1, bcol2 = st.columns(2)
        if bcol1.button("🚀 Get Started", type="primary", use_container_width=True):
            go("register")
        if bcol2.button("🔑 Login", use_container_width=True):
            go("login")

    # ── Features ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="features-section">
        <h2 class="section-title">Everything You Need to <span>Trade Smart</span></h2>
        <p class="section-subtitle">8 specialized AI agents working in unison to power your decisions</p>
        <div style="text-align:center; margin-bottom:3rem;">
            <img src="{IMG_FEATURES}" style="max-width:600px; width:100%; border-radius:16px;
            border:1px solid rgba(245,158,11,.2); box-shadow:0 20px 40px rgba(0,0,0,.4);" />
        </div>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🗄️</div>
                <div class="feature-title">Data Agent</div>
                <div class="feature-desc">Real-time stock prices, OHLCV data, and live news from Yahoo Finance — no API key needed.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Technical Analysis</div>
                <div class="feature-desc">RSI, MACD, Bollinger Bands, Moving Average crossovers with automated BUY/SELL signals.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📰</div>
                <div class="feature-title">GPT-4o Sentiment</div>
                <div class="feature-desc">AI reads financial headlines and understands context, sarcasm, and market impact.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">ML Prediction</div>
                <div class="feature-desc">Random Forest + Gradient Boosting ensemble predicts stock price movement 7 days ahead.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">⚖️</div>
                <div class="feature-title">Decision Agent</div>
                <div class="feature-desc">GPT-4o combines all signals and your risk profile to generate a final BUY / SELL / HOLD.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">AI Explanation</div>
                <div class="feature-desc">Get a full professional investment report written by GPT-4o — not templates.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💼</div>
                <div class="feature-title">Portfolio Tracker</div>
                <div class="feature-desc">Track your positions, monitor live P&L, and visualize performance over time.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">AI Chatbot</div>
                <div class="feature-desc">Ask anything about any stock. Get live news, price data, and AI-powered analysis instantly.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live News Ticker ───────────────────────────────────────────────────────
    render_ticker()

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#0a0e1a; padding:6rem 2rem;">
        <div class="about-section">
            <div class="about-img">
                <img src="{IMG_ABOUT}" alt="Trading" />
            </div>
            <div class="about-text">
                <h2>About <span>TradeXAI</span></h2>
                <p>
                    TradeXAI is a full Agentic AI system where 8 specialized agents collaborate autonomously
                    to deliver institutional-grade stock analysis to individual investors.
                </p>
                <p>
                    Unlike traditional stock tools that use rigid rules, TradeXAI uses GPT-4o to
                    reason over technical signals, news sentiment, and ML predictions — just like
                    a professional portfolio manager would.
                </p>
                <div>
                    <span class="about-pill">GPT-4o</span>
                    <span class="about-pill">scikit-learn</span>
                    <span class="about-pill">yfinance</span>
                    <span class="about-pill">TextBlob NLP</span>
                    <span class="about-pill">Plotly Charts</span>
                    <span class="about-pill">Real-time Data</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer CTA ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#0d1220; padding:5rem 2rem; text-align:center;">
        <h2 style="font-size:2.2rem; font-weight:800; color:#fff; margin-bottom:.5rem;">
            Ready to trade <span style="color:#f59e0b;">smarter</span>?
        </h2>
        <p style="color:rgba(255,255,255,.5); margin-bottom:2rem;">
            Join TradeXAI — Free to start, no credit card required.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([3, 1, 3])
    with col_b:
        if st.button("🚀 Create Free Account", type="primary", use_container_width=True):
            go("register")

    st.markdown("""
    <div class="footer-bar">
        © 2026 TradeXAI · Built with Agentic AI · Not financial advice
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LOGIN
# ══════════════════════════════════════════════════════════════════════════════
def page_login():
    st.markdown('<div class="auth-wrapper">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="auth-card">
            <div class="auth-logo">
                <div class="brand">📈 TradeXAI</div>
                <p>AI-Powered Stock Intelligence</p>
            </div>
            <div class="auth-title">Welcome Back</div>
            <div class="auth-sub">Sign in to your account</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In →", type="primary", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please fill in all fields.")
            else:
                result = login(username.strip(), password)
                if result["ok"]:
                    st.session_state.logged_in = True
                    st.session_state.user = result["user"]
                    st.success(f"Welcome back, {result['user']['username']}!")
                    go("dashboard")
                else:
                    st.error(result["msg"])

        st.markdown("---")
        col_a, col_b = st.columns(2)
        if col_a.button("← Back to Home", use_container_width=True):
            go("home")
        if col_b.button("Register →", use_container_width=True):
            go("register")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: REGISTER
# ══════════════════════════════════════════════════════════════════════════════
def page_register():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="padding-top:3rem;">
        </div>
        """, unsafe_allow_html=True)
        st.markdown("## 📈 TradeXAI — Create Account")
        st.caption("Start your AI-powered trading journey")

        with st.form("register_form"):
            username = st.text_input("Username", placeholder="Choose a username")
            email    = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", placeholder="Min 6 characters")
            confirm  = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
            submitted = st.form_submit_button("Create Account →", type="primary", use_container_width=True)

        if submitted:
            if not all([username, email, password, confirm]):
                st.error("Please fill in all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                result = register(username.strip(), email.strip(), password)
                if result["ok"]:
                    st.success(result["msg"])
                    go("login")
                else:
                    st.error(result["msg"])

        st.markdown("---")
        col_a, col_b = st.columns(2)
        if col_a.button("← Back to Home", use_container_width=True):
            go("home")
        if col_b.button("Already have an account? Login", use_container_width=True):
            go("login")


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD — Sidebar
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem 0 1rem;">
            <div style="font-size:1.8rem; font-weight:900; color:#f59e0b;">📈 TradeXAI</div>
            <div style="font-size:.75rem; color:rgba(255,255,255,.3); letter-spacing:1px; text-transform:uppercase;">AI Stock Platform</div>
        </div>
        """, unsafe_allow_html=True)

        user = st.session_state.user or {}
        st.markdown(f"""
        <div style="background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.2);
             border-radius:10px; padding:.8rem 1rem; margin-bottom:1.5rem; text-align:center;">
            <div style="font-size:1.5rem;">👤</div>
            <div style="font-weight:700; color:#fff;">{user.get('username','User')}</div>
            <div style="font-size:.75rem; color:rgba(255,255,255,.4);">{user.get('email','')}</div>
        </div>
        """, unsafe_allow_html=True)

        nav_items = [
            ("analysis",   "📊", "Stock Analysis"),
            ("charts",     "📈", "Charts"),
            ("sentiment",  "📰", "News Sentiment"),
            ("portfolio",  "💼", "Portfolio"),
            ("alerts",     "🔔", "Alerts"),
            ("chatbot",    "🤖", "AI Chatbot"),
        ]

        st.markdown("<div style='margin-bottom:.5rem; font-size:.75rem; color:rgba(255,255,255,.3); text-transform:uppercase; letter-spacing:1px; padding-left:.5rem;'>Navigation</div>", unsafe_allow_html=True)
        for key, icon, label in nav_items:
            is_active = st.session_state.dash_page == key
            btn_style = "primary" if is_active else "secondary"
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True,
                         type=btn_style if is_active else "secondary"):
                go_dash(key)

        st.markdown("---")
        if st.button("🚪  Logout", use_container_width=True):
            logout()

        st.markdown("""
        <div style="padding:1rem; font-size:.7rem; color:rgba(255,255,255,.2); text-align:center; margin-top:2rem;">
            TradeXAI v2.0<br>Not financial advice
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD SUB-PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ── Charts helpers ─────────────────────────────────────────────────────────────
def _color(d): return {"BUY":"#10b981","SELL":"#ef4444","HOLD":"#f59e0b"}.get(d,"#aaa")
def _icon(d):  return {"BUY":"↑ BUY","SELL":"↓ SELL","HOLD":"→ HOLD"}.get(d,d)
def _cls(d):   return {"BUY":"dec-buy","SELL":"dec-sell","HOLD":"dec-hold"}.get(d,"dec-hold")

def price_chart(df, mas, bb, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55,0.25,0.20], vertical_spacing=0.04,
        subplot_titles=("Price & MAs","Volume","RSI"))
    fig.add_trace(go.Candlestick(x=df.index,open=df["Open"],high=df["High"],
        low=df["Low"],close=df["Close"],name="Price",
        increasing_line_color="#10b981",decreasing_line_color="#ef4444"),row=1,col=1)
    for col,clr in [("MA20","#60a5fa"),("MA50","#f59e0b"),("EMA20","#a78bfa")]:
        if col in mas.columns:
            fig.add_trace(go.Scatter(x=mas.index,y=mas[col],name=col,line=dict(color=clr,width=1.5)),row=1,col=1)
    if bb is not None and "BB_Upper" in bb.columns:
        fig.add_trace(go.Scatter(x=bb.index,y=bb["BB_Upper"],name="BB Upper",
            line=dict(color="rgba(150,150,150,.4)",dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=bb.index,y=bb["BB_Lower"],name="BB Lower",
            line=dict(color="rgba(150,150,150,.4)",dash="dot"),
            fill="tonexty",fillcolor="rgba(180,180,180,.05)"),row=1,col=1)
    vol_colors=["#10b981" if c>=o else "#ef4444" for c,o in zip(df["Close"],df["Open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["Volume"],name="Volume",marker_color=vol_colors,opacity=.6),row=2,col=1)
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index,y=df["RSI"],name="RSI",
            line=dict(color="#60a5fa",width=1.5)),row=3,col=1)
        fig.add_hline(y=70,line_color="#ef4444",line_dash="dash",row=3,col=1)
        fig.add_hline(y=30,line_color="#10b981",line_dash="dash",row=3,col=1)
    fig.update_layout(height=650,xaxis_rangeslider_visible=False,
        paper_bgcolor="#0d1220",plot_bgcolor="#0d1220",font=dict(color="#fafafa"),
        title=dict(text=f"{ticker} — Interactive Chart",font_size=15),
        legend=dict(bgcolor="rgba(0,0,0,0)",font_size=11))
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
    return fig

def macd_chart(macd_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd_df.index,y=macd_df["MACD"],name="MACD",line=dict(color="#60a5fa")))
    fig.add_trace(go.Scatter(x=macd_df.index,y=macd_df["Signal"],name="Signal",line=dict(color="#f59e0b")))
    colors=["#10b981" if h>=0 else "#ef4444" for h in macd_df["Histogram"]]
    fig.add_trace(go.Bar(x=macd_df.index,y=macd_df["Histogram"],name="Histogram",marker_color=colors,opacity=.7))
    fig.update_layout(height=280,title="MACD",paper_bgcolor="#0d1220",
        plot_bgcolor="#0d1220",font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="rgba(255,255,255,.05)"),yaxis=dict(gridcolor="rgba(255,255,255,.05)"))
    return fig


# ── Sub: Analysis ──────────────────────────────────────────────────────────────
def sub_analysis():
    st.markdown("## 📊 Stock Analysis")
    st.caption("Full 8-agent AI pipeline: data → technical → sentiment → prediction → decision → explanation")

    c1, c2, c3 = st.columns([2, 1, 1])
    ticker = c1.text_input("Stock Ticker", value=st.session_state.last_ticker, placeholder="AAPL").upper().strip()
    risk   = c2.selectbox("Risk Profile", RISK_LEVELS, index=RISK_LEVELS.index(st.session_state.risk_level))
    analyse_btn = c3.button("🔍 Analyse", type="primary", use_container_width=True)

    # Quick picks
    st.markdown("**Quick pick:**")
    qcols = st.columns(len(DEFAULT_STOCKS))
    for i, t in enumerate(DEFAULT_STOCKS):
        if qcols[i].button(t, key=f"q_{t}"):
            st.session_state.last_ticker = t
            st.rerun()

    if analyse_btn:
        st.session_state.last_ticker = ticker
        st.session_state.risk_level  = risk
        with st.spinner(f"Running all agents on {ticker}…"):
            st.session_state.last_result = run_analysis(ticker, risk)

    result = st.session_state.last_result
    if not result:
        st.info("Enter a ticker and click Analyse to begin.")
        return
    if result["status"] == "error":
        st.error(f"Could not fetch data for **{result['ticker']}**.")
        return

    dec_data = result["decision"]
    expl     = result["explanation"]
    data     = result["data"]
    pred     = result["prediction"]
    tech     = result["technical"]

    # Company header
    cinfo    = data["company_info"]
    pinfo    = data["price_info"]
    st.markdown(f"### {cinfo.get('name', ticker)} `{ticker}`")
    st.caption(f"{cinfo.get('sector','')} · {cinfo.get('industry','')} · {cinfo.get('country','')}")
    st.divider()

    # Metrics row
    cur  = pinfo.get("current_price", 0)
    prev = pinfo.get("previous_close", cur)
    chg  = cur - prev if isinstance(cur,(int,float)) else 0
    chg_pct = (chg/prev*100) if prev else 0
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Price",  f"${cur}", f"{chg:+.2f} ({chg_pct:+.1f}%)")
    m2.metric("52W High", f"${cinfo.get('52w_high','N/A')}")
    m3.metric("52W Low",  f"${cinfo.get('52w_low','N/A')}")
    m4.metric("P/E Ratio", cinfo.get("pe_ratio","N/A"))
    m5.metric("7d Forecast", f"${pred.get('predicted_price','N/A')}", f"{pred.get('change_pct',0):+.2f}%")
    st.divider()

    # Decision card
    d    = dec_data["decision"]
    conf = dec_data["confidence"]
    ws   = dec_data["weighted_score"]
    pw   = dec_data.get("powered_by","")
    badge = "🤖 GPT-4o" if "GPT" in pw else "⚙️ Rule-based"

    dc, sc = st.columns([1, 2])
    with dc:
        st.markdown(f"""
        <div class="{_cls(d)}">
            <div class="dec-label" style="color:{_color(d)}">{_icon(d)}</div>
            <div style="margin-top:.5rem"><b>Confidence:</b> {conf}</div>
            <div><b>Score:</b> {ws:+.3f}</div>
            <div><b>Risk:</b> {risk}</div>
            <div style="margin-top:.8rem; font-size:.8rem; color:rgba(255,255,255,.5);">{badge}</div>
        </div>
        """, unsafe_allow_html=True)
        reasoning = dec_data.get("reasoning","")
        if reasoning:
            st.info(reasoning)
        for f in dec_data.get("key_factors",[]):
            st.markdown(f"• {f}")

    with sc:
        cs = dec_data["component_scores"]
        csig = dec_data["component_signals"]
        st.markdown("**Agent Signal Breakdown**")
        rows = [
            {"Agent":"Technical","Score":cs.get("technical",0),
             "Signals":f"RSI:{csig['technical'].get('rsi','?')} · MACD:{csig['technical'].get('macd','?')} · MA:{csig['technical'].get('ma_cross','?')}"},
            {"Agent":"Sentiment","Score":cs.get("sentiment",0),"Signals":str(csig.get("sentiment","N/A"))},
            {"Agent":"Prediction","Score":cs.get("prediction",0),"Signals":str(csig.get("prediction","N/A"))},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for row in rows:
            norm = (row["Score"]+3)/6 if row["Agent"]=="Technical" else (row["Score"]+1)/2
            st.progress(float(max(0,min(1,norm))), text=f"{row['Agent']}: {row['Score']:+d}")

    st.divider()

    # Technical indicators table
    st.markdown("### Technical Indicators")
    signals = tech.get("signals",{})
    rows = []
    if signals.get("rsi"):
        r=signals["rsi"]; rows.append({"Indicator":"RSI","Value":r["value"],"Signal":r["signal"],"Reason":r["reason"]})
    if signals.get("macd"):
        r=signals["macd"]; rows.append({"Indicator":"MACD","Value":round(r["macd"],4),"Signal":r["signal"],"Reason":f"Histogram:{r['histogram']:+.4f}"})
    if signals.get("moving_average"):
        r=signals["moving_average"]; rows.append({"Indicator":"MA Cross","Value":f"MA20={r['MA20']}","Signal":r["signal"],"Reason":r["reason"]})
    if rows:
        def hl(v):
            if v=="BUY": return "background-color:#0d2e1a;color:#10b981;font-weight:700"
            if v=="SELL": return "background-color:#2e0d0d;color:#ef4444;font-weight:700"
            return "background-color:#2e2200;color:#f59e0b;font-weight:700"
        st.dataframe(pd.DataFrame(rows).style.map(hl,subset=["Signal"]),use_container_width=True,hide_index=True)

    ta, tb = st.columns(2)
    ta.info(f"**Volume:** {tech.get('volume_signal','N/A')}")
    tb.info(f"**Trend:** {signals.get('trend','N/A')}")
    st.divider()

    # AI Report
    exp_pw = expl.get("powered_by","")
    exp_badge = "🤖 GPT-4o Report" if "GPT" in exp_pw else "⚙️ Template Report"
    st.markdown(f"### AI Analysis Report &nbsp; `{exp_badge}`")
    st.markdown(expl.get("explanation_markdown",""))

    with st.expander("Company Overview"):
        st.write(cinfo.get("description","No description available."))


# ── Sub: Charts ───────────────────────────────────────────────────────────────
def sub_charts():
    st.markdown("## 📈 Interactive Charts")
    result = st.session_state.last_result
    if not result or result["status"] != "ok":
        st.info("Run an analysis first from the Stock Analysis page.")
        return

    tech    = result["technical"]
    df_ind  = tech.get("df_with_indicators", pd.DataFrame())
    mas     = tech.get("mas", pd.DataFrame())
    bb      = tech.get("bb",  pd.DataFrame())
    macd_df = tech.get("macd_df", pd.DataFrame())

    if not df_ind.empty:
        st.plotly_chart(price_chart(df_ind, mas, bb, result["ticker"]), use_container_width=True)
        st.plotly_chart(macd_chart(macd_df), use_container_width=True)
    else:
        st.warning("Not enough data to render charts.")

    pred = result["prediction"]
    if pred.get("individual_predictions"):
        st.markdown("### ML Model Predictions")
        preds = pred["individual_predictions"]
        pdf = pd.DataFrame([{"Model":k,"Predicted ($)":v} for k,v in preds.items()])
        pdf["Change %"] = ((pdf["Predicted ($)"]-pred.get("current_price",0))/pred.get("current_price",1)*100).round(2)
        c1, c2 = st.columns([1,2])
        c1.dataframe(pdf, use_container_width=True, hide_index=True)
        fig = px.bar(pdf, x="Model", y="Predicted ($)", color="Change %",
            color_continuous_scale=["#ef4444","#f59e0b","#10b981"],
            text="Predicted ($)", title="ML Ensemble Predictions")
        fig.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
        fig.update_layout(paper_bgcolor="#0d1220",plot_bgcolor="#0d1220",
            font=dict(color="#fafafa"),height=300)
        c2.plotly_chart(fig, use_container_width=True)


# ── Sub: Sentiment ────────────────────────────────────────────────────────────
def sub_sentiment():
    st.markdown("## 📰 News Sentiment Analysis")
    result = st.session_state.last_result
    if not result or result["status"] != "ok":
        st.info("Run an analysis first.")
        return

    sent     = result["sentiment"]
    summary  = sent.get("summary",{})
    analyzed = sent.get("analyzed_articles",[])

    pw = summary.get("powered_by","")
    st.markdown(f"`{'🤖 GPT-4o' if 'GPT' in pw else '⚙️ TextBlob'} Sentiment Engine`")

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Overall", summary.get("overall_label","N/A"))
    s2.metric("Polarity", f"{summary.get('overall_polarity',0):+.3f}")
    s3.metric("Positive", summary.get("positive_count",0))
    s4.metric("Negative", summary.get("negative_count",0))

    if summary.get("gpt_summary"):
        st.info(f"**GPT-4o:** {summary['gpt_summary']}")
    if summary.get("key_themes"):
        st.markdown("**Themes:** " + " · ".join(f"`{t}`" for t in summary["key_themes"]))

    if analyzed:
        df = pd.DataFrame(analyzed)
        df["short"] = df["title"].str[:55] + "…"
        colors = df["polarity"].apply(lambda p:"#10b981" if p>.1 else("#ef4444" if p<-.1 else "#f59e0b"))
        fig = go.Figure(go.Bar(y=df["short"],x=df["polarity"],orientation="h",
            marker_color=colors, text=df["label"], textposition="outside"))
        fig.update_layout(height=max(300,len(df)*42),title="Sentiment per Headline",
            xaxis=dict(range=[-1,1],gridcolor="rgba(255,255,255,.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,.05)"),
            paper_bgcolor="#0d1220",plot_bgcolor="#0d1220",font=dict(color="#fafafa"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Headlines")
        news_df = pd.DataFrame(analyzed)[["title","published","label","polarity"]]
        news_df.columns = ["Headline","Published","Sentiment","Polarity"]
        st.dataframe(news_df, use_container_width=True, hide_index=True)


# ── Sub: Portfolio ────────────────────────────────────────────────────────────
def sub_portfolio():
    st.markdown("## 💼 Portfolio Tracker")
    pa = agents["portfolio"]

    c_add, c_del = st.columns(2)
    with c_add:
        with st.form("pf_add"):
            st.markdown("**Add / Update Position**")
            pt = st.text_input("Ticker", placeholder="AAPL").upper().strip()
            pq = st.number_input("Shares", min_value=0.001, step=1.0, value=1.0)
            pp = st.number_input("Buy Price ($)", min_value=0.01, step=0.01, value=100.0)
            if st.form_submit_button("Add", type="primary", use_container_width=True):
                if pt:
                    r = pa.add_position(pt, pq, pp)
                    st.success(f"Position {r['status']}: {r['position']['ticker']}")
                    st.rerun()

    with c_del:
        with st.form("pf_del"):
            st.markdown("**Remove Position**")
            dt = st.text_input("Ticker to Remove", placeholder="AAPL").upper().strip()
            if st.form_submit_button("Remove", type="secondary", use_container_width=True):
                if dt:
                    r = pa.remove_position(dt)
                    st.success(f"Removed {dt}.") if r["status"]=="removed" else st.warning(f"{dt} not found.")
                    st.rerun()

    st.divider()
    summary  = pa.portfolio_summary()
    positions = summary["positions"]

    if not positions:
        st.info("No positions yet. Add some stocks above.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Invested",     f"${summary['total_invested']:,.2f}")
    c2.metric("Market Value", f"${summary['total_market_value']:,.2f}")
    c3.metric("Total P&L",    f"${summary['total_pnl']:,.2f}", f"{summary['total_pnl_pct']:+.2f}%")
    c4.metric("Positions",    len(positions))

    df = pd.DataFrame(positions)
    fig = px.bar(df, x="ticker", y="pnl", color="status",
        color_discrete_map={"Profit":"#10b981","Loss":"#ef4444"},
        text="pnl_pct", title="P&L by Position")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(paper_bgcolor="#0d1220",plot_bgcolor="#0d1220",
        font=dict(color="#fafafa"),height=320,showlegend=True,
        yaxis=dict(gridcolor="rgba(255,255,255,.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,.05)"))
    st.plotly_chart(fig, use_container_width=True)

    pos_df = pd.DataFrame(positions)[["ticker","qty","buy_price","current_price","cost_basis","market_value","pnl","pnl_pct","status"]]
    pos_df.columns = ["Ticker","Qty","Buy $","Current $","Cost","Value","P&L $","P&L %","Status"]
    def spnl(v):
        if isinstance(v,(int,float)): return f"color:{'#10b981' if v>=0 else '#ef4444'}"
        return ""
    st.dataframe(pos_df.style.map(spnl,subset=["P&L $","P&L %"]),use_container_width=True,hide_index=True)


# ── Sub: Alerts ───────────────────────────────────────────────────────────────
def sub_alerts():
    st.markdown("## 🔔 Price & Sentiment Alerts")
    aa = agents["alert"]

    with st.form("alert_add"):
        st.markdown("**Create Alert**")
        a1,a2,a3,a4 = st.columns(4)
        at = a1.text_input("Ticker", placeholder="AAPL").upper().strip()
        atype = a2.selectbox("Type", ["price_above","price_below","decision_change","sentiment_change"])
        athr  = a3.number_input("Threshold", min_value=0.0, step=1.0)
        anote = a4.text_input("Note", placeholder="Optional")
        if st.form_submit_button("Add Alert", type="primary"):
            if at:
                aa.add_alert(at, atype, float(athr), anote)
                st.success(f"Alert added: {at} {atype} @ {athr}")
                st.rerun()

    st.divider()
    alerts = aa.get_alerts()
    if not alerts:
        st.info("No alerts configured.")
    else:
        for alert in alerts:
            icon = "✅" if alert.get("triggered") else "⏳"
            c1,c2,c3,c4 = st.columns([.5, 3, 2, 1])
            c1.markdown(f"{icon} **#{alert['id']}**")
            c2.markdown(f"**{alert['ticker']}** — {alert['type']} @ {alert.get('threshold','N/A')}" + (f" _{alert['note']}_" if alert.get("note") else ""))
            if alert.get("triggered"):
                c3.markdown(f"🔴 {alert.get('triggered_at','')}")
                if c4.button("Reset", key=f"rst_{alert['id']}"):
                    aa.reset_alert(alert["id"]); st.rerun()
            else:
                c3.markdown(f"Created: {alert.get('created','')}")
                if c4.button("Delete", key=f"del_{alert['id']}"):
                    aa.delete_alert(alert["id"]); st.rerun()

    st.divider()
    if st.button("🔔 Check All Alerts Now", type="secondary"):
        with st.spinner("Checking live prices…"):
            triggered = aa.check_alerts()
        if triggered:
            for t in triggered: st.warning(f"**TRIGGERED:** {t.get('message','')}")
        else:
            st.success("No alerts triggered.")


# ── Sub: Chatbot ──────────────────────────────────────────────────────────────
def sub_chatbot():
    st.markdown("## 🤖 AI Stock Chatbot")
    ca = agents["chat"]

    st.caption(f"Powered by GPT-4o · Fetches live prices & news · Ask about any stock")

    if not ca.available:
        st.error("OpenAI API key not found. Add your key to `.env` file.")
        return

    # Suggestions
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        sug_cols = st.columns(3)
        for i, s in enumerate(ca.get_suggestions()):
            if sug_cols[i%3].button(s, key=f"sug_{i}"):
                st.session_state.chat_history.append({"role":"user","content":s})
                with st.spinner("Thinking…"):
                    reply = ca.chat(st.session_state.chat_history[:-1], s)
                st.session_state.chat_history.append({"role":"assistant","content":reply})
                st.rerun()

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"]=="user" else "🤖"):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about any stock… e.g. 'What's the latest on AAPL?'")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Fetching live data…"):
                reply = ca.chat(
                    [m for m in st.session_state.chat_history[:-1]],
                    user_input
                )
            st.markdown(reply)
        st.session_state.chat_history.append({"role":"assistant","content":reply})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    if not st.session_state.logged_in:
        go("login")
        return

    render_sidebar()

    sub_map = {
        "analysis":  sub_analysis,
        "charts":    sub_charts,
        "sentiment": sub_sentiment,
        "portfolio": sub_portfolio,
        "alerts":    sub_alerts,
        "chatbot":   sub_chatbot,
    }
    sub_map.get(st.session_state.dash_page, sub_analysis)()


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.page

if page == "home":
    page_home()
elif page == "login":
    page_login()
elif page == "register":
    page_register()
elif page == "dashboard":
    page_dashboard()
else:
    go("home")
