# ─── agents/decision_agent.py ─────────────────────────────────────────────────
"""
Decision Agent  — GPT-4 powered
Uses GPT-4 to reason over all agent signals like a financial analyst.
Falls back to weighted scoring if OpenAI is unavailable.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RISK_WEIGHTS = {
    "Conservative": {"technical": 0.50, "sentiment": 0.20, "prediction": 0.30},
    "Moderate":     {"technical": 0.40, "sentiment": 0.30, "prediction": 0.30},
    "Aggressive":   {"technical": 0.30, "sentiment": 0.20, "prediction": 0.50},
}
BUY_THRESHOLDS  = {"Conservative": 0.55, "Moderate": 0.35, "Aggressive": 0.20}
SELL_THRESHOLDS = {"Conservative": -0.55, "Moderate": -0.35, "Aggressive": -0.20}


class DecisionAgent:
    def __init__(self):
        self.name = "Decision Agent (GPT-4)"
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.use_gpt = self.client is not None

    # ── GPT-4 decision ──────────────────────────────────────────────────────────
    def _gpt_decide(
        self,
        ticker: str,
        tech_signals: dict,
        tech_score: int,
        sent_summary: dict,
        pred_result: dict,
        risk_level: str,
    ) -> dict:
        prompt = f"""You are a senior portfolio manager making investment decisions.

Stock: {ticker}
Investor Risk Profile: {risk_level}

TECHNICAL SIGNALS (Score: {tech_score}/3):
- RSI: {tech_signals.get('rsi', {}).get('value', 'N/A')} → {tech_signals.get('rsi', {}).get('signal', 'N/A')}
- MACD: {tech_signals.get('macd', {}).get('signal', 'N/A')} (Histogram: {tech_signals.get('macd', {}).get('histogram', 0):+.4f})
- Moving Average: {tech_signals.get('moving_average', {}).get('signal', 'N/A')} — {tech_signals.get('moving_average', {}).get('reason', '')}
- Trend: {tech_signals.get('trend', 'N/A')}

NEWS SENTIMENT:
- Overall: {sent_summary.get('overall_label', 'N/A')} (Polarity: {sent_summary.get('overall_polarity', 0):+.3f})
- Positive: {sent_summary.get('positive_count', 0)} | Negative: {sent_summary.get('negative_count', 0)} articles
- Market Impact: {sent_summary.get('market_impact', 'N/A')}
- Themes: {', '.join(sent_summary.get('key_themes', []))}

ML PREDICTION:
- Expected price change: {pred_result.get('change_pct', 0):+.2f}% in {pred_result.get('predict_days', 7)} days
- ML Signal: {pred_result.get('signal', 'N/A')}
- Predicted Price: ${pred_result.get('predicted_price', 'N/A')}

Based on ALL signals above, make an investment decision appropriate for a {risk_level} investor.

Respond with ONLY valid JSON (no markdown):
{{
  "decision": "BUY|SELL|HOLD",
  "confidence": "High|Medium|Low",
  "weighted_score": <float -1.0 to 1.0>,
  "reasoning": "<1-2 sentence explanation of why>",
  "key_factors": ["factor1", "factor2", "factor3"]
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a portfolio manager. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    # ── Rule-based fallback ─────────────────────────────────────────────────────
    def _normalize(self, score, min_val=-3, max_val=3):
        return max(-1.0, min(1.0, score / max(abs(min_val), abs(max_val))))

    def _rule_decide(self, tech_score, sent_score, pred_score, risk_level):
        w = RISK_WEIGHTS.get(risk_level, RISK_WEIGHTS["Moderate"])
        weighted = (
            w["technical"] * self._normalize(tech_score, -3, 3)
            + w["sentiment"] * self._normalize(sent_score, -1, 1)
            + w["prediction"] * self._normalize(pred_score, -1, 1)
        )
        weighted = round(weighted, 4)
        buy_t  = BUY_THRESHOLDS.get(risk_level, 0.35)
        sell_t = SELL_THRESHOLDS.get(risk_level, -0.35)
        decision = "BUY" if weighted >= buy_t else ("SELL" if weighted <= sell_t else "HOLD")
        abs_s = abs(weighted)
        confidence = "High" if abs_s >= 0.7 else ("Medium" if abs_s >= 0.4 else "Low")
        return decision, weighted, confidence

    # ── Main entry ──────────────────────────────────────────────────────────────
    def run(
        self,
        technical_result: dict,
        sentiment_result: dict,
        prediction_result: dict,
        risk_level: str = "Moderate",
        ticker: str = "",
    ) -> dict:
        tech_score  = technical_result.get("score", 0)
        sent_score  = sentiment_result.get("score", 0)
        pred_score  = prediction_result.get("score", 0)
        tech_signals = technical_result.get("signals", {})
        sent_summary = sentiment_result.get("summary", {})

        comp_signals = {
            "technical": {
                "rsi":      tech_signals.get("rsi", {}).get("signal", "N/A"),
                "macd":     tech_signals.get("macd", {}).get("signal", "N/A"),
                "ma_cross": tech_signals.get("moving_average", {}).get("signal", "N/A"),
                "trend":    tech_signals.get("trend", "N/A"),
            },
            "sentiment": sent_summary.get("signal", "N/A"),
            "prediction": prediction_result.get("signal", "N/A"),
        }

        if self.use_gpt:
            try:
                gpt = self._gpt_decide(
                    ticker, tech_signals, tech_score,
                    sent_summary, prediction_result, risk_level
                )
                decision   = gpt.get("decision", "HOLD")
                confidence = gpt.get("confidence", "Low")
                weighted   = gpt.get("weighted_score", 0.0)
                return {
                    "status": "ok",
                    "decision": decision,
                    "weighted_score": weighted,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "reasoning": gpt.get("reasoning", ""),
                    "key_factors": gpt.get("key_factors", []),
                    "component_scores": {"technical": tech_score, "sentiment": sent_score, "prediction": pred_score},
                    "component_signals": comp_signals,
                    "powered_by": "GPT-4",
                }
            except Exception as e:
                print(f"[DecisionAgent] GPT-4 failed ({e}), using rule-based fallback")

        # Fallback
        decision, weighted, confidence = self._rule_decide(tech_score, sent_score, pred_score, risk_level)
        return {
            "status": "ok",
            "decision": decision,
            "weighted_score": weighted,
            "confidence": confidence,
            "risk_level": risk_level,
            "reasoning": "",
            "key_factors": [],
            "component_scores": {"technical": tech_score, "sentiment": sent_score, "prediction": pred_score},
            "component_signals": comp_signals,
            "powered_by": "Rule-based (fallback)",
        }
