# ─── agents/explanation_agent.py ──────────────────────────────────────────────
"""
Explanation Agent  — GPT-4 powered
Generates rich, dynamic investment analysis reports using GPT-4.
Falls back to template-based generation if OpenAI is unavailable.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class ExplanationAgent:
    def __init__(self):
        self.name = "Explanation Agent (GPT-4)"
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.use_gpt = self.client is not None

    # ── GPT-4 report generation ─────────────────────────────────────────────────
    def _gpt_explain(
        self,
        ticker: str,
        decision: str,
        confidence: str,
        risk_level: str,
        signals: dict,
        sent_summary: dict,
        pred_result: dict,
        price_info: dict,
    ) -> str:
        context = f"""
Stock Ticker: {ticker}
Current Price: ${price_info.get('current_price', 'N/A')}
Final Decision: {decision}
Confidence: {confidence}
Risk Profile: {risk_level}

TECHNICAL SIGNALS:
- RSI: {signals.get('rsi', {}).get('value', 'N/A')} → {signals.get('rsi', {}).get('signal', 'N/A')} ({signals.get('rsi', {}).get('reason', '')})
- MACD: {signals.get('macd', {}).get('signal', 'N/A')} (Histogram: {signals.get('macd', {}).get('histogram', 'N/A')})
- Moving Average: {signals.get('moving_average', {}).get('signal', 'N/A')} — {signals.get('moving_average', {}).get('reason', '')}
- Price Trend: {signals.get('trend', 'N/A')}

SENTIMENT:
- Overall: {sent_summary.get('overall_label', 'N/A')} (Polarity: {sent_summary.get('overall_polarity', 0):+.3f})
- Positive News: {sent_summary.get('positive_count', 0)} | Negative: {sent_summary.get('negative_count', 0)}
- Key Themes: {', '.join(sent_summary.get('key_themes', []))}
- GPT Analysis: {sent_summary.get('gpt_summary', '')}

ML PRICE PREDICTION:
- Predicted Price: ${pred_result.get('predicted_price', 'N/A')} in {pred_result.get('predict_days', 7)} days
- Expected Change: {pred_result.get('change_pct', 0):+.2f}%
- ML Signal: {pred_result.get('signal', 'N/A')}
"""

        prompt = f"""You are an expert financial analyst and investment advisor.

Based on the following multi-agent AI analysis, write a comprehensive investment report:

{context}

Write a professional report in markdown format with these sections:
1. **Executive Summary** — 2-3 sentences with the key recommendation
2. **Technical Analysis** — interpret RSI, MACD, Moving Averages in plain English
3. **Market Sentiment** — explain news impact on the stock
4. **Price Forecast** — interpret the ML prediction
5. **Risk Assessment** — risks specific to this {risk_level} investor profile
6. **Action Plan** — specific steps the investor should take (entry point, stop loss, target)
7. **Disclaimer** — one line

Be specific, insightful, and use plain English. Avoid generic statements.
Do NOT repeat the raw numbers unnecessarily — interpret what they MEAN."""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior financial analyst. Write clear, actionable, professional investment reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()

    # ── Template fallback ───────────────────────────────────────────────────────
    def _template_explain(
        self, ticker, decision, confidence, risk_level, signals, sent_summary, pred_result
    ) -> str:
        lines = [
            f"### Analysis Report for {ticker}",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
            "#### Final Decision",
        ]
        icon = {"BUY": "↑", "SELL": "↓", "HOLD": "→"}.get(decision, "")
        lines.append(f"**{icon} {decision}** with **{confidence} confidence** | Risk: {risk_level}\n")
        lines.append("#### Technical Analysis")
        if signals.get("rsi"):
            r = signals["rsi"]
            lines.append(f"- RSI {r['value']:.1f} → **{r['signal']}** ({r['reason']})")
        if signals.get("macd"):
            m = signals["macd"]
            lines.append(f"- MACD **{m['signal']}** (Histogram: {m['histogram']:+.4f})")
        if signals.get("moving_average"):
            ma = signals["moving_average"]
            lines.append(f"- MA Crossover **{ma['signal']}** — {ma['reason']}")
        lines.append(f"\n#### Sentiment\n- {sent_summary.get('overall_label','N/A')} sentiment from {sent_summary.get('article_count',0)} articles")
        lines.append(f"\n#### ML Prediction\n- Price predicted to move **{pred_result.get('change_pct',0):+.2f}%** in {pred_result.get('predict_days',7)} days → ${pred_result.get('predicted_price','N/A')}")
        notes = {
            "Conservative": {"BUY": "Consider a small position with a tight stop-loss.", "SELL": "Reduce exposure to limit risk.", "HOLD": "Wait for a clearer signal."},
            "Moderate": {"BUY": "Balanced position recommended; set stop-loss ~5% below entry.", "SELL": "Consider reducing position.", "HOLD": "Monitor closely."},
            "Aggressive": {"BUY": "Strong signal — larger position if risk tolerance allows.", "SELL": "Exit and consider short.", "HOLD": "Situation may develop soon."},
        }
        lines.append(f"\n#### Investor Note\n{notes.get(risk_level,{}).get(decision,'')}")
        lines.append("\n> **Disclaimer:** AI-generated analysis for educational purposes only. Not financial advice.")
        return "\n".join(lines)

    # ── Main entry ──────────────────────────────────────────────────────────────
    def run(
        self,
        decision_result: dict,
        technical_result: dict,
        sentiment_result: dict,
        prediction_result: dict,
        ticker: str,
    ) -> dict:
        decision = decision_result.get("decision", "HOLD")
        confidence = decision_result.get("confidence", "Low")
        risk_level = decision_result.get("risk_level", "Moderate")
        signals = technical_result.get("signals", {})
        sent_summary = sentiment_result.get("summary", {})
        price_info = {}

        short_map = {
            "BUY": f"Bullish outlook on {ticker}. Multiple signals favor buying.",
            "SELL": f"Bearish outlook on {ticker}. Indicators suggest selling pressure.",
            "HOLD": f"Mixed signals on {ticker}. Best to wait for a clearer trend.",
        }

        if self.use_gpt:
            try:
                report = self._gpt_explain(
                    ticker, decision, confidence, risk_level,
                    signals, sent_summary, prediction_result, price_info
                )
                return {
                    "status": "ok",
                    "decision": decision,
                    "explanation_markdown": report,
                    "short_summary": short_map.get(decision, ""),
                    "powered_by": "GPT-4",
                }
            except Exception as e:
                print(f"[ExplanationAgent] GPT-4 failed ({e}), using template fallback")

        # Fallback
        report = self._template_explain(
            ticker, decision, confidence, risk_level, signals, sent_summary, prediction_result
        )
        return {
            "status": "ok",
            "decision": decision,
            "explanation_markdown": report,
            "short_summary": short_map.get(decision, ""),
            "powered_by": "Template (fallback)",
        }
