# ─── app.py ───────────────────────────────────────────────────────────────────
"""
Agentic AI Stock Market Decision System
Main Streamlit application — single file UI that orchestrates all agents.

Run with:  streamlit run app.py
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
from config import DEFAULT_STOCKS, RISK_LEVELS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinanceAgent - AI Stock Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .big-metric {font-size: 2rem; font-weight: 700;}
    .buy-card  {background:#e6f4ea; border-left:5px solid #1e8e3e; padding:1rem; border-radius:8px;}
    .sell-card {background:#fce8e6; border-left:5px solid #d93025; padding:1rem; border-radius:8px;}
    .hold-card {background:#fef7e0; border-left:5px solid #f9ab00; padding:1rem; border-radius:8px;}
    .agent-badge {display:inline-block; padding:0.2rem 0.7rem; border-radius:12px;
                  font-size:0.75rem; font-weight:600; margin:2px;}
    .stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Agent singletons (cached across reruns) ────────────────────────────────────
@st.cache_resource
def get_agents():
    return {
        "data": DataAgent(),
        "technical": TechnicalAgent(),
        "sentiment": SentimentAgent(),
        "prediction": PredictionAgent(),
        "decision": DecisionAgent(),
        "explanation": ExplanationAgent(),
        "portfolio": PortfolioAgent(),
        "alert": AlertAgent(),
    }

agents = get_agents()


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(ticker: str, risk_level: str) -> dict:
    """Full pipeline – cached for 5 min."""
    data = agents["data"].run(ticker)
    if data["status"] != "ok":
        return {"status": "error", "ticker": ticker}

    tech = agents["technical"].run(data)
    sent = agents["sentiment"].run(data)
    pred = agents["prediction"].run(data)
    decision = agents["decision"].run(tech, sent, pred, risk_level, ticker)
    expl = agents["explanation"].run(decision, tech, sent, pred, ticker)

    return {
        "status": "ok",
        "ticker": ticker,
        "data": data,
        "technical": tech,
        "sentiment": sent,
        "prediction": pred,
        "decision": decision,
        "explanation": expl,
    }


def decision_color(d: str) -> str:
    return {"BUY": "#1e8e3e", "SELL": "#d93025", "HOLD": "#f9ab00"}.get(d, "#555")

def decision_card_class(d: str) -> str:
    return {"BUY": "buy-card", "SELL": "sell-card", "HOLD": "hold-card"}.get(d, "hold-card")

def decision_icon(d: str) -> str:
    return {"BUY": "↑ BUY", "SELL": "↓ SELL", "HOLD": "→ HOLD"}.get(d, d)


# ── Charts ─────────────────────────────────────────────────────────────────────
def price_chart(df: pd.DataFrame, mas: pd.DataFrame, bb, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=("Price & Moving Averages", "Volume", "RSI"),
    )
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color="#1e8e3e", decreasing_line_color="#d93025",
        ),
        row=1, col=1,
    )
    # MAs
    colors = {"MA20": "#1a73e8", "MA50": "#fa7b17", "EMA20": "#9334e6"}
    for col in ["MA20", "MA50", "EMA20"]:
        if col in mas.columns:
            fig.add_trace(
                go.Scatter(x=mas.index, y=mas[col], name=col,
                           line=dict(color=colors[col], width=1.5)),
                row=1, col=1,
            )
    # Bollinger Bands
    if bb is not None and "BB_Upper" in bb.columns:
        fig.add_trace(
            go.Scatter(x=bb.index, y=bb["BB_Upper"], name="BB Upper",
                       line=dict(color="rgba(150,150,150,0.5)", dash="dot")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=bb.index, y=bb["BB_Lower"], name="BB Lower",
                       line=dict(color="rgba(150,150,150,0.5)", dash="dot"),
                       fill="tonexty", fillcolor="rgba(180,180,180,0.1)"),
            row=1, col=1,
        )
    # Volume bars
    colors_vol = ["#1e8e3e" if c >= o else "#d93025"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=colors_vol, opacity=0.6),
        row=2, col=1,
    )
    # RSI
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                       line=dict(color="#1a73e8", width=1.5)),
            row=3, col=1,
        )
        fig.add_hline(y=70, line_color="#d93025", line_dash="dash", row=3, col=1)
        fig.add_hline(y=30, line_color="#1e8e3e", line_dash="dash", row=3, col=1)

    fig.update_layout(
        height=650, showlegend=True,
        title=dict(text=f"{ticker} — Interactive Price Chart", font_size=16),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)")
    return fig


def macd_chart(macd_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["MACD"],
                             name="MACD", line=dict(color="#1a73e8")))
    fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["Signal"],
                             name="Signal", line=dict(color="#fa7b17")))
    colors = ["#1e8e3e" if h >= 0 else "#d93025" for h in macd_df["Histogram"]]
    fig.add_trace(go.Bar(x=macd_df.index, y=macd_df["Histogram"],
                         name="Histogram", marker_color=colors, opacity=0.7))
    fig.update_layout(
        height=300, title="MACD",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
    )
    return fig


def sentiment_bar_chart(analyzed: list[dict]) -> go.Figure:
    if not analyzed:
        return go.Figure()
    df = pd.DataFrame(analyzed)
    df["short_title"] = df["title"].str[:50] + "…"
    colors = df["polarity"].apply(
        lambda p: "#1e8e3e" if p > 0.1 else ("#d93025" if p < -0.1 else "#f9ab00")
    )
    fig = go.Figure(
        go.Bar(
            y=df["short_title"],
            x=df["polarity"],
            orientation="h",
            marker_color=colors,
            text=df["label"],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=max(300, len(df) * 40),
        title="News Sentiment per Headline",
        xaxis_title="Polarity Score",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)", range=[-1, 1]),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
    )
    return fig


def portfolio_chart(positions: list[dict]) -> go.Figure:
    if not positions:
        return go.Figure()
    df = pd.DataFrame(positions)
    fig = px.bar(
        df, x="ticker", y="pnl",
        color="status",
        color_discrete_map={"Profit": "#1e8e3e", "Loss": "#d93025"},
        text="pnl_pct",
        title="Portfolio P&L by Position",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=350, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        yaxis=dict(title="P&L ($)", gridcolor="rgba(255,255,255,0.07)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        showlegend=True,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📈 FinanceAgent")
    st.caption("Agentic AI Stock Advisor")
    st.divider()

    ticker_input = st.text_input(
        "Stock Ticker", value="AAPL",
        placeholder="e.g. AAPL, TSLA, NVDA"
    ).strip().upper()

    st.write("**Quick Select**")
    cols = st.columns(3)
    quick_tickers = DEFAULT_STOCKS[:6]
    for i, t in enumerate(quick_tickers):
        if cols[i % 3].button(t, key=f"quick_{t}", use_container_width=True):
            ticker_input = t

    st.divider()
    risk_level = st.selectbox("Risk Profile", RISK_LEVELS, index=1)
    st.caption("Conservative: needs stronger signals\nAggressive: acts on weaker signals")

    st.divider()
    analyze_btn = st.button("🔍 Analyse Stock", type="primary", use_container_width=True)

    st.divider()
    st.caption("**Active Agents**")
    agent_names = [
        ("🗄️", "Data"), ("📊", "Technical"), ("📰", "Sentiment"),
        ("🤖", "Prediction"), ("⚖️", "Decision"), ("💬", "Explanation"),
        ("💼", "Portfolio"), ("🔔", "Alert"),
    ]
    for icon, name in agent_names:
        st.markdown(
            f'<span class="agent-badge" style="background:#1e3a5f;color:#90caf9;">'
            f'{icon} {name} Agent</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("No API key needed · Powered by yfinance + TextBlob + scikit-learn")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT — Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_analysis, tab_charts, tab_sentiment, tab_portfolio, tab_alerts = st.tabs(
    ["🏠 Analysis", "📊 Charts", "📰 Sentiment", "💼 Portfolio", "🔔 Alerts"]
)

# ── Trigger analysis ─────────────────────────────────────────────────────────
result = None
if analyze_btn or "last_result" in st.session_state:
    if analyze_btn:
        with st.spinner(f"Running all agents on {ticker_input} …"):
            result = run_analysis(ticker_input, risk_level)
        st.session_state["last_result"] = result
        st.session_state["last_ticker"] = ticker_input
    else:
        result = st.session_state.get("last_result")
        ticker_input = st.session_state.get("last_ticker", ticker_input)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    if result is None:
        st.info("Enter a stock ticker in the sidebar and click **Analyse Stock** to begin.")
        st.markdown(
            """
            ### How it works
            1. **Data Agent** fetches live price & news from Yahoo Finance
            2. **Technical Agent** calculates RSI, MACD, Moving Averages
            3. **Sentiment Agent** scores news headlines using NLP
            4. **Prediction Agent** trains an ML ensemble and forecasts price
            5. **Decision Agent** combines all signals with your risk profile
            6. **Explanation Agent** generates a human-readable report
            7. **Portfolio Agent** tracks your investments and P&L
            8. **Alert Agent** monitors price & sentiment thresholds
            """
        )
    elif result["status"] == "error":
        st.error(f"Could not fetch data for **{result['ticker']}**. Check the ticker symbol.")
    else:
        decision_data = result["decision"]
        explanation = result["explanation"]
        data = result["data"]
        pred = result["prediction"]
        tech = result["technical"]

        # ── Header ──────────────────────────────────────────────────────────
        company = data["company_info"]
        price_info = data["price_info"]
        st.subheader(f"{company.get('name', ticker_input)}  ({ticker_input})")
        st.caption(
            f"{company.get('sector', '')}  ·  {company.get('industry', '')}  ·  "
            f"{company.get('country', '')}"
        )

        # ── Key metrics row ─────────────────────────────────────────────────
        m1, m2, m3, m4, m5 = st.columns(5)
        current = price_info.get("current_price", "N/A")
        prev = price_info.get("previous_close", current)
        if isinstance(current, (int, float)) and isinstance(prev, (int, float)):
            change = current - prev
            change_pct = (change / prev * 100) if prev else 0
        else:
            change = change_pct = 0

        m1.metric("Current Price", f"${current}", f"{change:+.2f} ({change_pct:+.1f}%)")
        m2.metric("52W High", f"${company.get('52w_high', 'N/A')}")
        m3.metric("52W Low", f"${company.get('52w_low', 'N/A')}")
        m4.metric("P/E Ratio", company.get("pe_ratio", "N/A"))
        m5.metric(
            "Predicted Price",
            f"${pred.get('predicted_price', 'N/A')}",
            f"{pred.get('change_pct', 0):+.2f}% in {pred.get('predict_days', 7)}d",
        )

        st.divider()

        # ── Decision card ────────────────────────────────────────────────────
        col_dec, col_scores = st.columns([1, 2])
        with col_dec:
            d = decision_data["decision"]
            conf = decision_data["confidence"]
            ws = decision_data["weighted_score"]
            powered_by = decision_data.get("powered_by", "")
            card_class = decision_card_class(d)
            badge = "🤖 GPT-4" if "GPT" in powered_by else "⚙️ Rule-based"
            st.markdown(
                f"""
                <div class="{card_class}">
                  <div class="big-metric" style="color:{decision_color(d)}">
                    {decision_icon(d)}
                  </div>
                  <div><b>Confidence:</b> {conf}</div>
                  <div><b>Weighted Score:</b> {ws:+.3f}</div>
                  <div><b>Risk Profile:</b> {risk_level}</div>
                  <div style="margin-top:8px"><span class="agent-badge" style="background:#1e3a5f;color:#90caf9;">{badge}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # GPT reasoning
            reasoning = decision_data.get("reasoning", "")
            key_factors = decision_data.get("key_factors", [])
            if reasoning:
                st.markdown(f"> {reasoning}")
            if key_factors:
                st.markdown("**Key Factors:**")
                for f in key_factors:
                    st.markdown(f"- {f}")

        with col_scores:
            comp_scores = decision_data["component_scores"]
            comp_signals = decision_data["component_signals"]
            st.markdown("**Signal Breakdown**")
            score_data = {
                "Agent": ["Technical", "Sentiment", "Prediction"],
                "Score": [
                    comp_scores.get("technical", 0),
                    comp_scores.get("sentiment", 0),
                    comp_scores.get("prediction", 0),
                ],
                "Signal": [
                    f"RSI:{comp_signals['technical'].get('rsi','?')} | "
                    f"MACD:{comp_signals['technical'].get('macd','?')} | "
                    f"MA:{comp_signals['technical'].get('ma_cross','?')}",
                    str(comp_signals.get("sentiment", "N/A")),
                    str(comp_signals.get("prediction", "N/A")),
                ],
            }
            st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)

            # Gauge-style score bars
            for agent, score in zip(score_data["Agent"], score_data["Score"]):
                norm = (score + 3) / 6 if agent == "Technical" else (score + 1) / 2
                color = "#1e8e3e" if score > 0 else ("#d93025" if score < 0 else "#f9ab00")
                st.progress(float(norm), text=f"{agent}: {score:+d}")

        st.divider()

        # ── Technical signals table ─────────────────────────────────────────
        st.subheader("Technical Indicators")
        signals = tech.get("signals", {})
        tech_rows = []
        if signals.get("rsi"):
            r = signals["rsi"]
            tech_rows.append({"Indicator": "RSI", "Value": r["value"],
                               "Signal": r["signal"], "Reason": r["reason"]})
        if signals.get("macd"):
            r = signals["macd"]
            tech_rows.append({"Indicator": "MACD", "Value": r["macd"],
                               "Signal": r["signal"], "Reason": f"Histogram: {r['histogram']:+.4f}"})
        if signals.get("moving_average"):
            r = signals["moving_average"]
            tech_rows.append({"Indicator": "MA Crossover", "Value": f"MA20={r['MA20']}",
                               "Signal": r["signal"], "Reason": r["reason"]})

        if tech_rows:
            tech_df = pd.DataFrame(tech_rows)

            def highlight_signal(val):
                if val == "BUY":
                    return "background-color: #1a3a2a; color: #1e8e3e; font-weight: bold"
                elif val == "SELL":
                    return "background-color: #3a1a1a; color: #d93025; font-weight: bold"
                return "background-color: #3a3a1a; color: #f9ab00; font-weight: bold"

            styled = tech_df.style.map(highlight_signal, subset=["Signal"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Volume Signal", tech.get("volume_signal", "N/A"))
        col_b.metric("Price Trend", signals.get("trend", "N/A"))

        st.divider()

        # ── Explanation ─────────────────────────────────────────────────────
        exp_powered = explanation.get("powered_by", "")
        exp_badge = "🤖 GPT-4 Report" if "GPT" in exp_powered else "⚙️ Template Report"
        st.subheader(f"AI Analysis Report")
        st.markdown(f'<span class="agent-badge" style="background:#1e3a5f;color:#90caf9;">{exp_badge}</span>', unsafe_allow_html=True)
        st.markdown(explanation.get("explanation_markdown", ""))

        st.divider()

        # ── Company info ────────────────────────────────────────────────────
        with st.expander("Company Description"):
            st.write(company.get("description", "No description available."))


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CHARTS
# ════════════════════════════════════════════════════════════════════════════
with tab_charts:
    if result and result["status"] == "ok":
        tech = result["technical"]
        data = result["data"]
        df_ind = tech.get("df_with_indicators", pd.DataFrame())
        mas = tech.get("mas", pd.DataFrame())
        bb = tech.get("bb", pd.DataFrame())
        macd_df = tech.get("macd_df", pd.DataFrame())

        if not df_ind.empty:
            st.plotly_chart(
                price_chart(df_ind, mas, bb, result["ticker"]),
                use_container_width=True,
            )
            st.plotly_chart(macd_chart(macd_df), use_container_width=True)
        else:
            st.warning("Not enough data to render charts.")

        # ── Prediction comparison ────────────────────────────────────────────
        pred = result["prediction"]
        if pred.get("individual_predictions"):
            st.subheader("ML Model Predictions")
            preds = pred["individual_predictions"]
            pred_df = pd.DataFrame(
                [{"Model": k, "Predicted Price ($)": v} for k, v in preds.items()]
            )
            pred_df["Change %"] = (
                (pred_df["Predicted Price ($)"] - pred.get("current_price", 0))
                / pred.get("current_price", 1) * 100
            ).round(2)

            col_p1, col_p2 = st.columns([1, 2])
            with col_p1:
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
            with col_p2:
                fig_pred = px.bar(
                    pred_df, x="Model", y="Predicted Price ($)",
                    color="Change %",
                    color_continuous_scale=["#d93025", "#f9ab00", "#1e8e3e"],
                    title="ML Ensemble Predictions",
                    text="Predicted Price ($)",
                )
                fig_pred.update_traces(texttemplate="$%{text:.2f}", textposition="outside")
                fig_pred.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font=dict(color="#fafafa"), height=300,
                )
                st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("Run an analysis from the sidebar to see charts.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — SENTIMENT
# ════════════════════════════════════════════════════════════════════════════
with tab_sentiment:
    if result and result["status"] == "ok":
        sent = result["sentiment"]
        summary = sent.get("summary", {})
        analyzed = sent.get("analyzed_articles", [])

        # Summary metrics
        powered = summary.get("powered_by", "")
        badge = "🤖 GPT-4 Sentiment" if "GPT" in powered else "⚙️ TextBlob Sentiment"
        st.markdown(f'<span class="agent-badge" style="background:#1e3a5f;color:#90caf9;">{badge}</span>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Overall Sentiment", summary.get("overall_label", "N/A"))
        s2.metric("Polarity Score", f"{summary.get('overall_polarity', 0):+.3f}")
        s3.metric("Positive Articles", summary.get("positive_count", 0))
        s4.metric("Negative Articles", summary.get("negative_count", 0))

        # GPT summary text
        gpt_summary = summary.get("gpt_summary", "")
        key_themes = summary.get("key_themes", [])
        if gpt_summary:
            st.info(f"**GPT-4 Analysis:** {gpt_summary}")
        if key_themes:
            st.markdown("**Key Themes:** " + " · ".join(f"`{t}`" for t in key_themes))

        st.plotly_chart(sentiment_bar_chart(analyzed), use_container_width=True)

        # News table
        if analyzed:
            st.subheader("News Headlines")
            news_df = pd.DataFrame(analyzed)[["title", "published", "label", "polarity"]]
            news_df.columns = ["Headline", "Published", "Sentiment", "Polarity"]
            st.dataframe(news_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run an analysis to see sentiment data.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    portfolio_agent = agents["portfolio"]

    # ── Add / Remove positions ────────────────────────────────────────────
    st.subheader("Manage Portfolio")
    col_add, col_del = st.columns(2)

    with col_add:
        with st.form("add_position_form"):
            st.markdown("**Add / Update Position**")
            p_ticker = st.text_input("Ticker", placeholder="AAPL").upper().strip()
            p_qty = st.number_input("Quantity (shares)", min_value=0.001, step=1.0, value=1.0)
            p_price = st.number_input("Buy Price ($)", min_value=0.01, step=0.01, value=100.0)
            if st.form_submit_button("Add Position", type="primary"):
                if p_ticker:
                    r = portfolio_agent.add_position(p_ticker, p_qty, p_price)
                    st.success(f"Position {r['status']}: {r['position']['ticker']}")
                    st.rerun()

    with col_del:
        with st.form("del_position_form"):
            st.markdown("**Remove Position**")
            del_ticker = st.text_input("Ticker to Remove", placeholder="AAPL").upper().strip()
            if st.form_submit_button("Remove", type="secondary"):
                if del_ticker:
                    r = portfolio_agent.remove_position(del_ticker)
                    if r["status"] == "removed":
                        st.success(f"Removed {del_ticker} from portfolio.")
                    else:
                        st.warning(f"{del_ticker} not found.")
                    st.rerun()

    st.divider()

    # ── Portfolio summary ────────────────────────────────────────────────
    st.subheader("Portfolio Overview")
    summary = portfolio_agent.portfolio_summary()
    positions = summary["positions"]

    if not positions:
        st.info("No positions yet. Add stocks above to track your portfolio.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Invested", f"${summary['total_invested']:,.2f}")
        c2.metric("Market Value", f"${summary['total_market_value']:,.2f}")
        c3.metric(
            "Total P&L",
            f"${summary['total_pnl']:,.2f}",
            f"{summary['total_pnl_pct']:+.2f}%",
        )
        c4.metric("Positions", len(positions))

        st.plotly_chart(portfolio_chart(positions), use_container_width=True)

        # Detailed table
        pos_df = pd.DataFrame(positions)[
            ["ticker", "qty", "buy_price", "current_price", "cost_basis", "market_value", "pnl", "pnl_pct", "status"]
        ]
        pos_df.columns = ["Ticker", "Qty", "Buy Price", "Current Price", "Cost Basis", "Market Value", "P&L ($)", "P&L %", "Status"]

        def style_pnl(val):
            if isinstance(val, (int, float)):
                return f"color: {'#1e8e3e' if val >= 0 else '#d93025'}"
            return ""

        styled_pos = pos_df.style.map(style_pnl, subset=["P&L ($)", "P&L %"])
        st.dataframe(styled_pos, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ALERTS
# ════════════════════════════════════════════════════════════════════════════
with tab_alerts:
    alert_agent = agents["alert"]

    st.subheader("Price & Sentiment Alerts")

    # ── Add alert ────────────────────────────────────────────────────────
    with st.form("add_alert_form"):
        st.markdown("**Create New Alert**")
        a1, a2, a3, a4 = st.columns(4)
        a_ticker = a1.text_input("Ticker", placeholder="AAPL").upper().strip()
        a_type = a2.selectbox(
            "Alert Type",
            ["price_above", "price_below", "decision_change", "sentiment_change"],
        )
        a_threshold = a3.number_input("Threshold (price/$)", min_value=0.0, step=1.0)
        a_note = a4.text_input("Note", placeholder="Optional note")
        if st.form_submit_button("Add Alert", type="primary"):
            if a_ticker:
                r = alert_agent.add_alert(a_ticker, a_type, float(a_threshold), a_note)
                st.success(f"Alert added for {a_ticker} ({a_type} @ {a_threshold})")
                st.rerun()

    st.divider()

    # ── Alert list ───────────────────────────────────────────────────────
    alerts = alert_agent.get_alerts()
    if not alerts:
        st.info("No alerts set. Create one above.")
    else:
        st.markdown(f"**{len(alerts)} Alert(s) configured**")
        for alert in alerts:
            status_icon = "✅" if alert.get("triggered") else "⏳"
            col_a, col_b, col_c, col_d = st.columns([1, 3, 2, 1])
            col_a.write(f"{status_icon} #{alert['id']}")
            col_b.write(
                f"**{alert['ticker']}** — {alert['type']} @ {alert.get('threshold', 'N/A')}"
                + (f"  _{alert['note']}_" if alert.get("note") else "")
            )
            if alert.get("triggered"):
                col_c.write(f"Triggered: {alert.get('triggered_at', '')}")
                if col_d.button("Reset", key=f"rst_{alert['id']}"):
                    alert_agent.reset_alert(alert["id"])
                    st.rerun()
            else:
                col_c.write(f"Created: {alert.get('created', '')}")
                if col_d.button("Delete", key=f"del_{alert['id']}"):
                    alert_agent.delete_alert(alert["id"])
                    st.rerun()

    st.divider()

    # ── Check now ────────────────────────────────────────────────────────
    if st.button("🔔 Check All Alerts Now", type="secondary"):
        with st.spinner("Checking alerts against live prices…"):
            triggered = alert_agent.check_alerts()
        if triggered:
            for t in triggered:
                st.warning(f"**ALERT TRIGGERED**: {t.get('message', '')}")
        else:
            st.success("No alerts triggered at this time.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer:** This application is for educational purposes only. "
    "It does NOT constitute financial advice. Always consult a financial advisor before investing."
)
