# ─── agents/sentiment_agent.py ────────────────────────────────────────────────
"""
Sentiment Analysis Agent  — GPT-4 powered
Uses GPT-4 to deeply understand news context, sarcasm, and market impact.
Falls back to TextBlob if OpenAI is unavailable.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from textblob import TextBlob
import json

load_dotenv()


class SentimentAgent:
    def __init__(self):
        self.name = "Sentiment Analysis Agent (GPT-4)"
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.use_gpt = self.client is not None

    # ── GPT-4 analysis ──────────────────────────────────────────────────────────
    def _gpt_analyze(self, headlines: list[str], ticker: str) -> dict:
        headlines_text = "\n".join(f"- {h}" for h in headlines[:10])
        prompt = f"""You are a financial news analyst specializing in stock market sentiment.

Analyze these news headlines for stock ticker: {ticker}

Headlines:
{headlines_text}

For each headline and overall, determine:
1. Sentiment: Positive / Negative / Neutral
2. Market Impact: High / Medium / Low
3. Key themes driving sentiment

Return a JSON response in this exact format:
{{
  "overall_sentiment": "Positive|Negative|Neutral",
  "overall_polarity": <float between -1.0 and 1.0>,
  "signal": "BUY|SELL|HOLD",
  "confidence": "High|Medium|Low",
  "positive_count": <int>,
  "negative_count": <int>,
  "neutral_count": <int>,
  "key_themes": ["theme1", "theme2", "theme3"],
  "market_impact": "High|Medium|Low",
  "summary": "<2-3 sentence analysis of how these headlines affect {ticker} stock>"
}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def _gpt_per_article(self, articles: list[dict], ticker: str) -> list[dict]:
        results = []
        for article in articles[:10]:
            text = article.get("title", "")
            try:
                prompt = f"""Analyze this financial headline for {ticker}:
"{text}"
Return JSON: {{"sentiment": "Positive|Negative|Neutral", "polarity": <-1.0 to 1.0>, "impact": "High|Medium|Low", "reason": "<brief>"}}"""
                resp = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Financial analyst. JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=150,
                )
                raw = resp.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw.strip())
                score_map = {"Positive": 1, "Negative": -1, "Neutral": 0}
                results.append({
                    "title": text,
                    "published": article.get("published", ""),
                    "link": article.get("link", ""),
                    "label": parsed.get("sentiment", "Neutral"),
                    "polarity": parsed.get("polarity", 0.0),
                    "impact": parsed.get("impact", "Medium"),
                    "reason": parsed.get("reason", ""),
                    "score": score_map.get(parsed.get("sentiment", "Neutral"), 0),
                    "subjectivity": 0.5,
                })
            except Exception:
                # Fallback to TextBlob for this article
                blob = TextBlob(text)
                p = blob.sentiment.polarity
                results.append({
                    "title": text,
                    "published": article.get("published", ""),
                    "link": article.get("link", ""),
                    "label": "Positive" if p > 0.1 else ("Negative" if p < -0.1 else "Neutral"),
                    "polarity": round(p, 4),
                    "impact": "Medium",
                    "reason": "TextBlob fallback",
                    "score": 1 if p > 0.1 else (-1 if p < -0.1 else 0),
                    "subjectivity": round(blob.sentiment.subjectivity, 4),
                })
        return results

    # ── TextBlob fallback ───────────────────────────────────────────────────────
    def _textblob_analyze(self, articles: list[dict]) -> tuple[list[dict], dict]:
        results = []
        for article in articles:
            text = article.get("title", "") + " " + article.get("summary", "")
            blob = TextBlob(text)
            p = blob.sentiment.polarity
            label = "Positive" if p > 0.1 else ("Negative" if p < -0.1 else "Neutral")
            score = 1 if p > 0.1 else (-1 if p < -0.1 else 0)
            results.append({
                "title": article.get("title", ""),
                "published": article.get("published", ""),
                "link": article.get("link", ""),
                "label": label, "polarity": round(p, 4),
                "subjectivity": round(blob.sentiment.subjectivity, 4),
                "score": score, "impact": "Medium", "reason": "TextBlob",
            })

        if not results:
            summary = {"overall_label": "Neutral", "overall_polarity": 0.0,
                       "signal": "HOLD", "overall_score": 0,
                       "positive_count": 0, "negative_count": 0,
                       "neutral_count": 0, "article_count": 0}
        else:
            avg_pol = sum(r["polarity"] for r in results) / len(results)
            pos = sum(1 for r in results if r["score"] == 1)
            neg = sum(1 for r in results if r["score"] == -1)
            neu = len(results) - pos - neg
            label = "Positive" if avg_pol > 0.05 else ("Negative" if avg_pol < -0.05 else "Neutral")
            signal = "BUY" if label == "Positive" else ("SELL" if label == "Negative" else "HOLD")
            summary = {
                "overall_label": label, "overall_polarity": round(avg_pol, 4),
                "signal": signal,
                "overall_score": 1 if signal == "BUY" else (-1 if signal == "SELL" else 0),
                "positive_count": pos, "negative_count": neg,
                "neutral_count": neu, "article_count": len(results),
            }
        return results, summary

    # ── Main entry ──────────────────────────────────────────────────────────────
    def run(self, stock_data: dict) -> dict:
        articles = stock_data.get("news", [])
        ticker = stock_data.get("ticker", "")

        if self.use_gpt and articles:
            try:
                headlines = [a.get("title", "") for a in articles]
                gpt_summary = self._gpt_analyze(headlines, ticker)
                analyzed = self._gpt_per_article(articles, ticker)

                signal = gpt_summary.get("signal", "HOLD")
                summary = {
                    "overall_label": gpt_summary.get("overall_sentiment", "Neutral"),
                    "overall_polarity": gpt_summary.get("overall_polarity", 0.0),
                    "signal": signal,
                    "overall_score": 1 if signal == "BUY" else (-1 if signal == "SELL" else 0),
                    "positive_count": gpt_summary.get("positive_count", 0),
                    "negative_count": gpt_summary.get("negative_count", 0),
                    "neutral_count": gpt_summary.get("neutral_count", 0),
                    "article_count": len(analyzed),
                    "confidence": gpt_summary.get("confidence", "Medium"),
                    "key_themes": gpt_summary.get("key_themes", []),
                    "market_impact": gpt_summary.get("market_impact", "Medium"),
                    "gpt_summary": gpt_summary.get("summary", ""),
                    "powered_by": "GPT-4",
                }
                return {
                    "status": "ok",
                    "analyzed_articles": analyzed,
                    "summary": summary,
                    "score": summary["overall_score"],
                }
            except Exception as e:
                print(f"[SentimentAgent] GPT-4 failed ({e}), falling back to TextBlob")

        # Fallback
        analyzed, summary = self._textblob_analyze(articles)
        summary["powered_by"] = "TextBlob (fallback)"
        return {"status": "ok", "analyzed_articles": analyzed, "summary": summary,
                "score": summary["overall_score"]}
