# ─── app.py ───────────────────────────────────────────────────────────────────
"""
FinanceAgent — Agentic AI Stock Market Decision System
Streamlit UI  |  Run: streamlit run app.py
"""

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
from config import DEFAULT_STOCKS, RISK_LEVELS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinanceAgent AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Modern dark professional theme ───────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.stApp { background: #0b0f1a; color: #e2e8f0; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(99,179,237,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #63b3ed, #76e4f7, #81e6d9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.hero-sub {
    color: #90cdf4; font-size: 1rem; margin-top: 0.5rem; opacity: 0.85;
}

/* ── Decision cards ── */
.card {
    border-radius: 14px; padding: 1.4rem 1.6rem;
    border: 1px solid; margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.card:hover { transform: translateY(-2px); }
.card-buy   { background: #0d2b1d; border-color: #38a169; }
.card-sell  { background: #2d1515; border-color: #e53e3e; }
.card-hold  { background: #2d2706; border-color: #d69e2e; }
.card-title { font-size: 2rem; font-weight: 800; }
.buy-text   { color: #68d391; }
.sell-text  { color: #fc8181; }
.hold-text  { color: #f6e05e; }

/* ── Info cards ── */
.info-card {
    background: #141928; border: 1px solid rgba(99,179,237,0.12);
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;
}
.info-card h4 { color: #63b3ed; margin: 0 0 0.5rem 0; font-size: 0.85rem;
                letter-spacing: 0.08em; text-transform: uppercase; }
.info-card p  { color: #e2e8f0; margin: 0; font-size: 1.1rem; font-weight: 600; }

/* ── Agent badges ── */
.badge {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 20px; font-size: 0.72rem; font-weight: 700;
    margin: 2px; letter-spacing: 0.04em;
}
.badge-blue   { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid rgba(99,179,237,0.3); }
.badge-green  { background: rgba(72,187,120,0.15); color: #68d391; border: 1px solid rgba(72,187,120,0.3); }
.badge-yellow { background: rgba(236,201,75,0.15);  color: #f6e05e; border: 1px solid rgba(236,201,75,0.3);  }
.badge-purple { background: rgba(159,122,234,0.15); color: #b794f4; border: 1px solid rgba(159,122,234,0.3); }

/* ── Chat UI ── */
.chat-bubble-user {
    background: linear-gradient(135deg, #2b4c7e, #1a365d);
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem; margin: 0.5rem 0 0.5rem 15%;
    color: #e2e8f0; border: 1px solid rgba(99,179,237,0.2);
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.chat-bubble-bot {
    background: linear-gradient(135deg, #141928, #1a202c);
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem; margin: 0.5rem 15% 0.5rem 0;
    color: #e2e8f0; border: 1px solid rgba(99,179,237,0.12);
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.chat-label { font-size: 0.72rem; opacity: 0.6; margin-bottom: 4px; }
.news-pill {
    background: #141928; border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
}
.news-pill:hover { border-color: rgba(99,179,237,0.4); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0d1117 !important; }
section[data-testid="stSidebar"] .stButton button {
    background: #141928 !important; color: #63b3ed !important;
    border: 1px solid rgba(99,179,237,0.25) !important;
    border-radius: 8px !important; font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(99,179,237,0.15) !important;
}
.primary-btn button {
    background: linear-gradient(135deg, #2b6cb0, #2c5282) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #141928; border-radius: 10px; padding: 4px; gap: 2px;
    border: 1px solid rgba(99,179,237,0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #718096 !important;
    font-weight: 600; font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.15) !important;
    color: #63b3ed !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #141928; border-radius: 12px; padding: 1rem;
    border: 1px solid rgba(99,179,237,0.1);
}
div[data-testid="stMetricValue"] { color: #e2e8f0 !important; }

/* ── Divider ── */
hr { border-color: rgba(99,179,237,0.1) !important; margin: 1.2rem 0 !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Agent singletons ───────────────────────────────────────────────────────────
@st.cache_resource
def get_agents():
    return {
        "data":        DataAgent(),
        "technical":   TechnicalAgent(),
        "sentiment":   SentimentAgent(),
        "prediction":  PredictionAgent(),
        "decision":    DecisionAgent(),
        "explanation": ExplanationAgent(),
        "portfolio":   PortfolioAgent(),
        "alert":       AlertAgent(),
        "chat":        ChatAgent(),
    }

agents = get_agents()

# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = "AAPL"

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(ticker: str, risk_level: str) -> dict:
    data     = agents["data"].run(ticker)
    if data["status"] != "ok":
        return {"status": "error", "ticker": ticker}
    tech     = agents["technical"].run(data)
    sent     = agents["sentiment"].run(data)
    pred     = agents["prediction"].run(data)
    decision = agents["decision"].run(tech, sent, pred, risk_level, ticker)
    expl     = agents["explanation"].run(decision, tech, sent, pred, ticker)
    return {"status":"ok","ticker":ticker,"data":data,"technical":tech,
            "sentiment":sent,"prediction":pred,"decision":decision,"explanation":expl}

def d_color(d): return {"BUY":"#68d391","SELL":"#fc8181","HOLD":"#f6e05e"}.get(d,"#e2e8f0")
def d_card(d):  return {"BUY":"card-buy","SELL":"card-sell","HOLD":"card-hold"}.get(d,"card-hold")
def d_icon(d):  return {"BUY":"↑ BUY","SELL":"↓ SELL","HOLD":"→ HOLD"}.get(d,d)
def d_text(d):  return {"BUY":"buy-text","SELL":"sell-text","HOLD":"hold-text"}.get(d,"hold-text")

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
      <div style="font-size:2rem">📈</div>
      <div style="font-size:1.2rem;font-weight:800;color:#63b3ed">FinanceAgent</div>
      <div style="font-size:0.75rem;color:#718096;margin-top:2px">Agentic AI Stock Advisor</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    ticker_input = st.text_input("Stock Ticker", value=st.session_state.last_ticker,
                                  placeholder="e.g. AAPL, TSLA, NVDA").strip().upper()

    st.markdown("<div style='font-size:0.78rem;color:#718096;margin-bottom:4px'>Quick Select</div>",
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, t in enumerate(DEFAULT_STOCKS[:6]):
        if cols[i % 3].button(t, key=f"qs_{t}", use_container_width=True):
            ticker_input = t

    st.divider()
    risk_level = st.selectbox("Risk Profile", RISK_LEVELS, index=1)
    st.caption("Conservative → needs stronger signals\nAggressive → acts on weaker signals")
    st.divider()

    with st.container():
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        analyse_btn = st.button("🔍  Analyse Stock", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:0.78rem;color:#718096;margin-bottom:6px'>Active Agents</div>",
                unsafe_allow_html=True)
    agent_badges = [
        ("🗄️","Data","blue"),("📊","Technical","blue"),("📰","Sentiment","green"),
        ("🤖","Prediction","purple"),("⚖️","Decision","yellow"),("💬","Explanation","green"),
        ("💼","Portfolio","blue"),("🔔","Alert","yellow"),("🧠","Chat AI","purple"),
    ]
    html = ""
    for icon, name, color in agent_badges:
        html += f'<span class="badge badge-{color}">{icon} {name}</span>'
    st.markdown(html, unsafe_allow_html=True)
    st.divider()
    st.caption("Powered by GPT-4o · yfinance · TextBlob · scikit-learn")

# ── Trigger analysis ───────────────────────────────────────────────────────────
if analyse_btn:
    st.session_state.last_ticker = ticker_input
    with st.spinner(f"Running all agents on {ticker_input} …"):
        result = run_analysis(ticker_input, risk_level)
    st.session_state.last_result = result
else:
    result = st.session_state.last_result

# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
ticker_display = st.session_state.last_ticker or "—"
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">FinanceAgent AI Platform</div>
  <div class="hero-sub">
    8-Agent Agentic AI · Real-time Stock Analysis · GPT-4o Powered Insights
    {'&nbsp;&nbsp;|&nbsp;&nbsp;Analysing: <b style="color:#76e4f7">' + ticker_display + '</b>' if result else ''}
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_home, tab_charts, tab_sentiment, tab_chat, tab_portfolio, tab_alerts = st.tabs([
    "🏠  Dashboard", "📊  Charts", "📰  Sentiment", "🧠  AI Chat", "💼  Portfolio", "🔔  Alerts"
])

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab_home:
    if result is None:
        # Landing state
        st.markdown("""
        <div class="info-card" style="text-align:center;padding:2.5rem">
          <div style="font-size:3rem;margin-bottom:1rem">📈</div>
          <h2 style="color:#63b3ed;margin:0 0 0.5rem">Welcome to FinanceAgent</h2>
          <p style="color:#718096;max-width:520px;margin:0 auto">
            Enter a stock ticker in the sidebar and click <b style="color:#63b3ed">Analyse Stock</b>
            to run all 8 AI agents and get a BUY / HOLD / SELL recommendation.
          </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        for col, icon, title, desc in [
            (c1, "📊", "Technical Analysis", "RSI, MACD, Bollinger Bands, Moving Average crossovers"),
            (c2, "📰", "Sentiment Analysis", "GPT-4o reads live news headlines and scores market mood"),
            (c3, "🤖", "ML Price Prediction", "Random Forest + Ridge + Gradient Boosting ensemble"),
        ]:
            col.markdown(f"""
            <div class="info-card" style="text-align:center">
              <div style="font-size:1.8rem">{icon}</div>
              <h4 style="font-size:0.9rem">{title}</h4>
              <p style="font-size:0.82rem;color:#718096;margin:0">{desc}</p>
            </div>""", unsafe_allow_html=True)

    elif result["status"] == "error":
        st.error(f"Could not fetch data for **{result['ticker']}**. Check the ticker symbol.")

    else:
        dec   = result["decision"]
        expl  = result["explanation"]
        data  = result["data"]
        pred  = result["prediction"]
        tech  = result["technical"]
        company    = data["company_info"]
        price_info = data["price_info"]

        # ── Company header ──────────────────────────────────────────────────
        st.markdown(f"""
        <div style="margin-bottom:1rem">
          <span style="font-size:1.5rem;font-weight:800;color:#e2e8f0">
            {company.get('name', result['ticker'])}
          </span>
          <span style="color:#718096;margin-left:10px">({result['ticker']})</span>
          <span style="color:#4a5568;margin-left:8px;font-size:0.85rem">
            {company.get('sector','')} · {company.get('industry','')}
          </span>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI metrics ─────────────────────────────────────────────────────
        current = price_info.get("current_price", 0)
        prev    = price_info.get("previous_close", current)
        chg     = current - prev if isinstance(current, float) else 0
        chg_pct = (chg / prev * 100) if prev else 0

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Current Price",   f"${current}",     f"{chg:+.2f} ({chg_pct:+.1f}%)")
        m2.metric("52W High",        f"${company.get('52w_high','N/A')}")
        m3.metric("52W Low",         f"${company.get('52w_low','N/A')}")
        m4.metric("P/E Ratio",       company.get("pe_ratio","N/A"))
        m5.metric("ML Forecast",
                  f"${pred.get('predicted_price','N/A')}",
                  f"{pred.get('change_pct',0):+.2f}% in {pred.get('predict_days',7)}d")
        st.divider()

        # ── Decision card + breakdown ────────────────────────────────────────
        col_dec, col_right = st.columns([1, 2])
        d = dec["decision"]
        with col_dec:
            powered  = dec.get("powered_by","")
            badge_lbl= "🤖 GPT-4o" if "GPT" in powered else "⚙️ Rule-based"
            reasoning = dec.get("reasoning","")
            key_factors = dec.get("key_factors",[])
            st.markdown(f"""
            <div class="card {d_card(d)}">
              <div class="card-title {d_text(d)}">{d_icon(d)}</div>
              <div style="margin-top:0.8rem;color:#a0aec0;font-size:0.88rem">
                <b style="color:#e2e8f0">Confidence:</b> {dec['confidence']}<br>
                <b style="color:#e2e8f0">Score:</b> {dec['weighted_score']:+.3f}<br>
                <b style="color:#e2e8f0">Risk Profile:</b> {risk_level}
              </div>
              <div style="margin-top:0.8rem">
                <span class="badge badge-purple">{badge_lbl}</span>
              </div>
            </div>""", unsafe_allow_html=True)
            if reasoning:
                st.info(reasoning)
            if key_factors:
                for f in key_factors:
                    st.markdown(f"• {f}")

        with col_right:
            # Signal cards
            signals = tech.get("signals",{})
            r = signals.get("rsi",{}); m = signals.get("macd",{}); ma = signals.get("moving_average",{})
            sent_sig = result["sentiment"].get("summary",{}).get("signal","N/A")
            sc1,sc2,sc3,sc4 = st.columns(4)
            for col, label, sig, val in [
                (sc1,"RSI", r.get("signal","—"), str(r.get("value","—"))),
                (sc2,"MACD",m.get("signal","—"), f"{m.get('histogram',0):+.4f}"),
                (sc3,"MA Cross",ma.get("signal","—"), ma.get("reason","")[:18]),
                (sc4,"Sentiment",sent_sig, result["sentiment"].get("summary",{}).get("overall_label","—")),
            ]:
                clr = {"BUY":"#68d391","SELL":"#fc8181","HOLD":"#f6e05e"}.get(sig,"#a0aec0")
                col.markdown(f"""
                <div class="info-card" style="text-align:center">
                  <h4>{label}</h4>
                  <p style="color:{clr};font-size:1.1rem">{sig}</p>
                  <div style="font-size:0.75rem;color:#718096">{val}</div>
                </div>""", unsafe_allow_html=True)

            st.divider()
            # Score bars
            comp = dec.get("component_scores",{})
            for agent_name, score, max_s in [
                ("Technical",  comp.get("technical",0),  3),
                ("Sentiment",  comp.get("sentiment",0),  1),
                ("Prediction", comp.get("prediction",0), 1),
            ]:
                norm = (score + max_s) / (2 * max_s)
                clr  = "normal" if score >= 0 else "inverse"
                st.progress(float(norm), text=f"{agent_name} Agent: {score:+d}")

        st.divider()

        # ── Technical table ──────────────────────────────────────────────────
        st.markdown("#### Technical Indicator Summary")
        rows = []
        if r: rows.append({"Indicator":"RSI","Value":r.get("value",""),"Signal":r.get("signal",""),"Reason":r.get("reason","")})
        if m: rows.append({"Indicator":"MACD","Value":m.get("macd",""),"Signal":m.get("signal",""),"Reason":f"Histogram {m.get('histogram',0):+.4f}"})
        if ma:rows.append({"Indicator":"MA Crossover","Value":f"MA20={ma.get('MA20','')}","Signal":ma.get("signal",""),"Reason":ma.get("reason","")})
        if rows:
            def hl(v):
                if v=="BUY":  return "background-color:#0d2b1d;color:#68d391;font-weight:700"
                if v=="SELL": return "background-color:#2d1515;color:#fc8181;font-weight:700"
                return "background-color:#2d2706;color:#f6e05e;font-weight:700"
            st.dataframe(
                pd.DataFrame(rows).style.map(hl,subset=["Signal"]),
                use_container_width=True, hide_index=True
            )

        col_a, col_b = st.columns(2)
        col_a.metric("Volume Signal", tech.get("volume_signal","N/A"))
        col_b.metric("Price Trend",   signals.get("trend","N/A"))
        st.divider()

        # ── GPT Report ──────────────────────────────────────────────────────
        exp_pw = expl.get("powered_by","")
        badge  = "🤖 GPT-4o Report" if "GPT" in exp_pw else "⚙️ Template Report"
        st.markdown(f"#### AI Analysis Report &nbsp;&nbsp;<span class='badge badge-{'purple' if 'GPT' in exp_pw else 'yellow'}'>{badge}</span>",
                    unsafe_allow_html=True)
        st.markdown(expl.get("explanation_markdown",""))

        with st.expander("Company Description"):
            st.write(company.get("description","N/A"))


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CHARTS
# ════════════════════════════════════════════════════════════════════════════
with tab_charts:
    if result and result["status"] == "ok":
        tech    = result["technical"]
        df_ind  = tech.get("df_with_indicators", pd.DataFrame())
        mas     = tech.get("mas", pd.DataFrame())
        bb      = tech.get("bb",  pd.DataFrame())
        macd_df = tech.get("macd_df", pd.DataFrame())
        ticker  = result["ticker"]

        if not df_ind.empty:
            # ── Candlestick + MA + BB + Volume + RSI ────────────────────────
            fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                                row_heights=[0.55,0.25,0.20],vertical_spacing=0.04,
                                subplot_titles=("Price & Moving Averages","Volume","RSI"))
            fig.add_trace(go.Candlestick(
                x=df_ind.index,open=df_ind["Open"],high=df_ind["High"],
                low=df_ind["Low"],close=df_ind["Close"],name="Price",
                increasing_line_color="#68d391",decreasing_line_color="#fc8181"),row=1,col=1)
            for col,clr in [("MA20","#63b3ed"),("MA50","#f6ad55"),("EMA20","#b794f4")]:
                if col in mas.columns:
                    fig.add_trace(go.Scatter(x=mas.index,y=mas[col],name=col,
                                            line=dict(color=clr,width=1.5)),row=1,col=1)
            if "BB_Upper" in bb.columns:
                fig.add_trace(go.Scatter(x=bb.index,y=bb["BB_Upper"],name="BB Upper",
                              line=dict(color="rgba(160,174,192,0.4)",dash="dot")),row=1,col=1)
                fig.add_trace(go.Scatter(x=bb.index,y=bb["BB_Lower"],name="BB Lower",
                              line=dict(color="rgba(160,174,192,0.4)",dash="dot"),
                              fill="tonexty",fillcolor="rgba(160,174,192,0.06)"),row=1,col=1)
            vol_clr=["#68d391" if c>=o else "#fc8181" for c,o in zip(df_ind["Close"],df_ind["Open"])]
            fig.add_trace(go.Bar(x=df_ind.index,y=df_ind["Volume"],name="Volume",
                                 marker_color=vol_clr,opacity=0.6),row=2,col=1)
            if "RSI" in df_ind.columns:
                fig.add_trace(go.Scatter(x=df_ind.index,y=df_ind["RSI"],name="RSI",
                              line=dict(color="#63b3ed",width=1.5)),row=3,col=1)
                fig.add_hline(y=70,line_color="#fc8181",line_dash="dash",row=3,col=1)
                fig.add_hline(y=30,line_color="#68d391",line_dash="dash",row=3,col=1)
            fig.update_layout(height=680,xaxis_rangeslider_visible=False,
                              paper_bgcolor="#0b0f1a",plot_bgcolor="#0b0f1a",
                              font=dict(color="#e2e8f0"),
                              legend=dict(bgcolor="rgba(0,0,0,0)"),
                              title=dict(text=f"{ticker} — Interactive Price Chart",font_size=15))
            fig.update_yaxes(gridcolor="rgba(99,179,237,0.06)")
            fig.update_xaxes(gridcolor="rgba(99,179,237,0.06)")
            st.plotly_chart(fig, use_container_width=True)

            # ── MACD chart ───────────────────────────────────────────────────
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=macd_df.index,y=macd_df["MACD"],name="MACD",line=dict(color="#63b3ed")))
            fig2.add_trace(go.Scatter(x=macd_df.index,y=macd_df["Signal"],name="Signal",line=dict(color="#f6ad55")))
            hist_clr=["#68d391" if h>=0 else "#fc8181" for h in macd_df["Histogram"]]
            fig2.add_trace(go.Bar(x=macd_df.index,y=macd_df["Histogram"],name="Histogram",
                                  marker_color=hist_clr,opacity=0.7))
            fig2.update_layout(height=280,title="MACD",paper_bgcolor="#0b0f1a",
                               plot_bgcolor="#0b0f1a",font=dict(color="#e2e8f0"),
                               xaxis=dict(gridcolor="rgba(99,179,237,0.06)"),
                               yaxis=dict(gridcolor="rgba(99,179,237,0.06)"))
            st.plotly_chart(fig2, use_container_width=True)

            # ── ML predictions bar ───────────────────────────────────────────
            pred = result["prediction"]
            if pred.get("individual_predictions"):
                st.markdown("#### ML Model Price Predictions")
                preds = pred["individual_predictions"]
                df_p  = pd.DataFrame([{"Model":k,"Predicted ($)":v} for k,v in preds.items()])
                df_p["Change %"] = ((df_p["Predicted ($)"]-pred.get("current_price",0))/pred.get("current_price",1)*100).round(2)
                cp1,cp2 = st.columns([1,2])
                cp1.dataframe(df_p,use_container_width=True,hide_index=True)
                fig3 = px.bar(df_p,x="Model",y="Predicted ($)",color="Change %",
                              color_continuous_scale=["#fc8181","#f6e05e","#68d391"],
                              text="Predicted ($)",title="Ensemble Predictions")
                fig3.update_traces(texttemplate="$%{text:.2f}",textposition="outside")
                fig3.update_layout(height=300,paper_bgcolor="#0b0f1a",plot_bgcolor="#0b0f1a",
                                   font=dict(color="#e2e8f0"))
                cp2.plotly_chart(fig3,use_container_width=True)
    else:
        st.info("Run an analysis from the sidebar to see charts.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SENTIMENT
# ════════════════════════════════════════════════════════════════════════════
with tab_sentiment:
    if result and result["status"] == "ok":
        sent     = result["sentiment"]
        summary  = sent.get("summary",{})
        analyzed = sent.get("analyzed_articles",[])
        powered  = summary.get("powered_by","")

        badge_s = "🤖 GPT-4o" if "GPT" in powered else "⚙️ TextBlob"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem">
          <span style="font-size:1.1rem;font-weight:700;color:#e2e8f0">Sentiment Analysis</span>
          <span class="badge badge-{'purple' if 'GPT' in powered else 'yellow'}">{badge_s}</span>
        </div>""", unsafe_allow_html=True)

        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Overall Sentiment", summary.get("overall_label","N/A"))
        s2.metric("Polarity Score",    f"{summary.get('overall_polarity',0):+.3f}")
        s3.metric("Positive News",     summary.get("positive_count",0))
        s4.metric("Negative News",     summary.get("negative_count",0))

        if summary.get("gpt_summary"):
            st.markdown(f"""
            <div class="info-card">
              <h4>GPT-4o Market Summary</h4>
              <p style="font-size:0.95rem;font-weight:400;color:#cbd5e0">{summary['gpt_summary']}</p>
            </div>""", unsafe_allow_html=True)

        if summary.get("key_themes"):
            st.markdown("**Key Themes:** " + " &nbsp;·&nbsp; ".join(
                f'<span class="badge badge-blue">{t}</span>' for t in summary["key_themes"]
            ), unsafe_allow_html=True)

        if analyzed:
            st.divider()
            # Horizontal bar chart
            df_s = pd.DataFrame(analyzed)
            df_s["short"] = df_s["title"].str[:55] + "…"
            clrs = df_s["polarity"].apply(lambda p:"#68d391" if p>0.1 else("#fc8181" if p<-0.1 else"#f6e05e"))
            fig_s = go.Figure(go.Bar(y=df_s["short"],x=df_s["polarity"],orientation="h",
                                     marker_color=clrs,text=df_s["label"],textposition="outside"))
            fig_s.update_layout(height=max(300,len(df_s)*38),title="Sentiment per Headline",
                                xaxis=dict(range=[-1,1],gridcolor="rgba(99,179,237,0.06)",title="Polarity"),
                                paper_bgcolor="#0b0f1a",plot_bgcolor="#0b0f1a",font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_s, use_container_width=True)

            st.markdown("#### News Headlines")
            for a in analyzed:
                clr = {"Positive":"#68d391","Negative":"#fc8181"}.get(a.get("label","Neutral"),"#f6e05e")
                impact = a.get("impact","")
                st.markdown(f"""
                <div class="news-pill">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <span style="font-size:0.88rem;color:#e2e8f0;flex:1">{a['title']}</span>
                    <span class="badge" style="background:rgba(0,0,0,0.3);color:{clr};border:1px solid {clr};margin-left:8px;white-space:nowrap">
                      {a.get('label','N/A')}
                    </span>
                  </div>
                  <div style="font-size:0.75rem;color:#718096;margin-top:4px">
                    {a.get('published','')[:16]}
                    {' &nbsp;|&nbsp; Impact: ' + impact if impact else ''}
                    {' &nbsp;|&nbsp; ' + a.get('reason','') if a.get('reason') else ''}
                  </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Run an analysis to see sentiment data.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — AI CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    chat_agent = agents["chat"]

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem">
      <div style="font-size:1.8rem">🧠</div>
      <div>
        <div style="font-size:1.2rem;font-weight:800;color:#e2e8f0">FinanceBot</div>
        <div style="font-size:0.78rem;color:#718096">
          GPT-4o · Fetches live stock news · Answers market questions in real time
        </div>
      </div>
      <span class="badge badge-purple" style="margin-left:auto">🤖 GPT-4o</span>
    </div>
    """, unsafe_allow_html=True)

    if not chat_agent.available:
        st.warning("Add your OpenAI API key to the `.env` file to enable the AI chatbot.")

    # ── Suggested questions ────────────────────────────────────────────────
    st.markdown("<div style='font-size:0.78rem;color:#718096;margin:0.5rem 0 0.3rem'>Suggested questions:</div>",
                unsafe_allow_html=True)
    suggestions = [
        "What's the latest news on AAPL?",
        "Should I buy TSLA right now?",
        "What is RSI and how does it work?",
        "Compare NVDA and AMD outlook",
        "What is the market sentiment today?",
        "Explain MACD in simple terms",
    ]
    sc = st.columns(3)
    for i, q in enumerate(suggestions):
        if sc[i % 3].button(q, key=f"sugg_{i}", use_container_width=True):
            if chat_agent.available:
                with st.spinner("FinanceBot is thinking …"):
                    resp = chat_agent.chat(q, st.session_state.chat_history)
                st.session_state.chat_history.append({"role":"user","content":q})
                st.session_state.chat_history.append({"role":"assistant","content":resp["reply"]})
                st.rerun()

    st.divider()

    # ── Live news feed in sidebar of chat ─────────────────────────────────
    col_chat, col_news = st.columns([2, 1])

    with col_news:
        st.markdown("<div style='font-size:0.85rem;font-weight:700;color:#63b3ed;margin-bottom:8px'>📡 Live Market News</div>",
                    unsafe_allow_html=True)
        current_ticker = st.session_state.last_ticker
        if st.button("🔄 Refresh News", use_container_width=True):
            st.cache_data.clear()
        news_list = chat_agent.get_market_news(
            ticker=current_ticker if current_ticker != "—" else None,
            max_articles=6
        )
        for article in news_list:
            st.markdown(f"""
            <div class="news-pill">
              <div style="font-size:0.78rem;color:#e2e8f0;line-height:1.4">{article['title'][:90]}…</div>
              <div style="font-size:0.68rem;color:#718096;margin-top:3px">{article.get('published','')[:16]}</div>
            </div>""", unsafe_allow_html=True)

    with col_chat:
        # ── Chat history ───────────────────────────────────────────────────
        chat_container = st.container(height=440)
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style="text-align:center;padding:3rem 0;color:#4a5568">
                  <div style="font-size:2.5rem;margin-bottom:0.5rem">💬</div>
                  <div>Ask me anything about stocks, markets, or news!</div>
                </div>""", unsafe_allow_html=True)
            else:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-bubble-user">
                          <div class="chat-label">You</div>
                          {msg['content']}
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-bubble-bot">
                          <div class="chat-label">🧠 FinanceBot</div>
                          {msg['content']}
                        </div>""", unsafe_allow_html=True)

        # ── Input ──────────────────────────────────────────────────────────
        with st.form("chat_form", clear_on_submit=True):
            ci1, ci2 = st.columns([5, 1])
            user_input = ci1.text_input("", placeholder="Ask about any stock, news, or market concept…",
                                         label_visibility="collapsed")
            send = ci2.form_submit_button("Send →", use_container_width=True)

        if send and user_input.strip():
            if chat_agent.available:
                with st.spinner("FinanceBot is researching …"):
                    resp = chat_agent.chat(user_input.strip(), st.session_state.chat_history)
                st.session_state.chat_history.append({"role":"user","content":user_input.strip()})
                st.session_state.chat_history.append({"role":"assistant","content":resp["reply"]})
                if resp.get("tickers_fetched"):
                    st.toast(f"📡 Live data fetched for: {', '.join(resp['tickers_fetched'])}")
                st.rerun()
            else:
                st.warning("OpenAI API key not configured.")

        # Clear button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    portfolio_agent = agents["portfolio"]
    st.markdown("### Portfolio Manager")

    col_add, col_del = st.columns(2)
    with col_add:
        with st.form("add_pos"):
            st.markdown('<div class="info-card"><h4>Add / Update Position</h4>', unsafe_allow_html=True)
            p_t = st.text_input("Ticker", placeholder="AAPL").upper().strip()
            p_q = st.number_input("Quantity", min_value=0.001, step=1.0, value=1.0)
            p_p = st.number_input("Buy Price ($)", min_value=0.01, step=0.01, value=100.0)
            st.markdown('</div>', unsafe_allow_html=True)
            if st.form_submit_button("➕ Add Position", type="primary"):
                if p_t:
                    r = portfolio_agent.add_position(p_t,p_q,p_p)
                    st.success(f"Position {r['status']}: {r['position']['ticker']}")
                    st.rerun()

    with col_del:
        with st.form("del_pos"):
            st.markdown('<div class="info-card"><h4>Remove Position</h4>', unsafe_allow_html=True)
            d_t = st.text_input("Ticker to Remove", placeholder="AAPL").upper().strip()
            st.markdown('</div>', unsafe_allow_html=True)
            if st.form_submit_button("🗑️ Remove", type="secondary"):
                if d_t:
                    r = portfolio_agent.remove_position(d_t)
                    st.success(f"Removed {d_t}.") if r["status"]=="removed" else st.warning(f"{d_t} not found.")
                    st.rerun()

    st.divider()
    summary   = portfolio_agent.portfolio_summary()
    positions = summary["positions"]

    if not positions:
        st.info("No positions yet. Add a stock above to start tracking.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Invested",  f"${summary['total_invested']:,.2f}")
        c2.metric("Market Value",    f"${summary['total_market_value']:,.2f}")
        c3.metric("Total P&L",       f"${summary['total_pnl']:,.2f}",
                                     f"{summary['total_pnl_pct']:+.2f}%")
        c4.metric("Positions",       len(positions))

        # P&L bar chart
        df_pos = pd.DataFrame(positions)
        fig_p = px.bar(df_pos, x="ticker", y="pnl",
                       color="status", color_discrete_map={"Profit":"#68d391","Loss":"#fc8181"},
                       text="pnl_pct", title="Portfolio P&L by Position")
        fig_p.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_p.update_layout(height=320, paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
                            font=dict(color="#e2e8f0"),
                            yaxis=dict(gridcolor="rgba(99,179,237,0.06)"))
        st.plotly_chart(fig_p, use_container_width=True)

        tbl = df_pos[["ticker","qty","buy_price","current_price","cost_basis","market_value","pnl","pnl_pct","status"]].copy()
        tbl.columns = ["Ticker","Qty","Buy $","Current $","Cost","Value","P&L $","P&L %","Status"]
        def spnl(v):
            if isinstance(v,(int,float)):
                return f"color:{'#68d391' if v>=0 else '#fc8181'}"
            return ""
        st.dataframe(tbl.style.map(spnl, subset=["P&L $","P&L %"]),
                     use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — ALERTS
# ════════════════════════════════════════════════════════════════════════════
with tab_alerts:
    alert_agent = agents["alert"]
    st.markdown("### Price & Event Alerts")

    with st.form("add_alert"):
        st.markdown('<div class="info-card"><h4>Create New Alert</h4>', unsafe_allow_html=True)
        a1,a2,a3,a4 = st.columns(4)
        a_t   = a1.text_input("Ticker", placeholder="AAPL").upper().strip()
        a_type= a2.selectbox("Type", ["price_above","price_below","decision_change","sentiment_change"])
        a_thr = a3.number_input("Threshold", min_value=0.0, step=1.0)
        a_note= a4.text_input("Note (optional)")
        st.markdown('</div>', unsafe_allow_html=True)
        if st.form_submit_button("🔔 Add Alert", type="primary"):
            if a_t:
                alert_agent.add_alert(a_t, a_type, float(a_thr), a_note)
                st.success(f"Alert added for {a_t}")
                st.rerun()

    st.divider()
    alerts = alert_agent.get_alerts()
    if not alerts:
        st.info("No alerts set. Create one above.")
    else:
        for alert in alerts:
            icon = "✅" if alert.get("triggered") else "⏳"
            c1,c2,c3,c4 = st.columns([0.5,3,2,1])
            c1.markdown(f"<div style='padding-top:8px'>{icon}</div>", unsafe_allow_html=True)
            c2.markdown(f"**{alert['ticker']}** · {alert['type']} @ {alert.get('threshold','N/A')}"
                        + (f" — _{alert['note']}_" if alert.get('note') else ""))
            c3.caption(f"{'Triggered: '+alert.get('triggered_at','') if alert.get('triggered') else 'Created: '+alert.get('created','')}")
            if alert.get("triggered"):
                if c4.button("Reset",key=f"rst_{alert['id']}"):
                    alert_agent.reset_alert(alert["id"]); st.rerun()
            else:
                if c4.button("Delete",key=f"del_{alert['id']}"):
                    alert_agent.delete_alert(alert["id"]); st.rerun()

    st.divider()
    if st.button("🔔 Check All Alerts Now", use_container_width=True):
        with st.spinner("Checking …"):
            triggered = alert_agent.check_alerts()
        if triggered:
            for t in triggered:
                st.warning(f"**TRIGGERED:** {t.get('message','')}")
        else:
            st.success("No alerts triggered.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.78rem;padding:0.5rem 0">
  ⚠️ FinanceAgent is for educational purposes only. Not financial advice.
  Always consult a qualified financial advisor before investing.
  <br>Built with Streamlit · GPT-4o · yfinance · scikit-learn
</div>
""", unsafe_allow_html=True)
