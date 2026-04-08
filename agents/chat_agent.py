# ─── agents/chat_agent.py ─────────────────────────────────────────────────────
"""
Chat Agent — GPT-4o powered conversational stock assistant
Fetches live news + price data and answers user questions in real time.
Maintains conversation memory within the Streamlit session.
"""

import json
from datetime import datetime
from openai import OpenAI
from agents.data_agent import DataAgent
from utils.secrets import get_openai_key


SYSTEM_PROMPT = """You are FinanceBot, an expert AI stock market analyst and financial advisor assistant.

You have access to real-time stock data and current news fetched from Yahoo Finance.
Your role is to:
1. Answer questions about stocks, markets, and investments clearly
2. Interpret live news and explain how it affects stock prices
3. Give BUY / HOLD / SELL opinions when asked, with clear reasoning
4. Explain financial concepts in simple language
5. Warn users when data is uncertain or when professional advice is needed

Always be:
- Specific (mention actual numbers, dates, companies)
- Balanced (mention both upside and risks)
- Clear (avoid jargon unless explained)

End every response with a one-line disclaimer if giving investment opinions.
Today's date: {date}
"""


class ChatAgent:
    def __init__(self):
        self.name = "Chat Agent (GPT-4o)"
        api_key = get_openai_key()
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.available = self.client is not None
        self.data_agent = DataAgent()

    # ── Ticker detection ────────────────────────────────────────────────────────
    def _extract_tickers(self, text: str) -> list[str]:
        """Detect stock tickers mentioned in user message."""
        import re
        # Match words that look like tickers (2-5 uppercase letters)
        words = re.findall(r'\b[A-Z]{2,5}\b', text.upper())
        # Common known tickers to validate against
        known = {
            "AAPL","GOOGL","GOOG","MSFT","AMZN","TSLA","NVDA","META","NFLX","AMD",
            "INTC","BABA","JPM","BAC","GS","MS","V","MA","PYPL","DIS","UBER","LYFT",
            "SNAP","TWTR","COIN","HOOD","PLTR","RBLX","ABNB","SHOP","SQ","ROKU",
            "ZM","PTON","DOCU","CRWD","OKTA","DDOG","NET","SNOW","MSTR","SPY",
            "QQQ","VTI","VOO","IWM","ARKK","GLD","SLV","BTC","ETH"
        }
        return list(set(w for w in words if w in known))[:3]  # max 3 tickers

    # ── Live context builder ────────────────────────────────────────────────────
    def _build_live_context(self, tickers: list[str]) -> str:
        if not tickers:
            return ""
        context_parts = []
        for ticker in tickers:
            try:
                price_info = self.data_agent.fetch_current_price(ticker)
                news = self.data_agent.fetch_news(ticker, max_articles=5)

                price_text = ""
                if price_info:
                    cur = price_info.get("current_price", "N/A")
                    prev = price_info.get("previous_close", "N/A")
                    if isinstance(cur, float) and isinstance(prev, float):
                        chg = cur - prev
                        chg_pct = (chg / prev * 100) if prev else 0
                        price_text = f"Current: ${cur:.2f} ({chg_pct:+.2f}% vs yesterday)"
                    else:
                        price_text = f"Current: ${cur}"

                news_text = ""
                if news:
                    headlines = [f"  - {a['title']} ({a.get('published','')[:16]})"
                                 for a in news[:5]]
                    news_text = "Latest News:\n" + "\n".join(headlines)

                context_parts.append(
                    f"=== {ticker} ===\n{price_text}\n{news_text}"
                )
            except Exception as e:
                context_parts.append(f"=== {ticker} === (data unavailable: {e})")

        return "\n\n".join(context_parts)

    # ── Main chat ───────────────────────────────────────────────────────────────
    def chat(self, user_message: str, history: list[dict]) -> dict:
        """
        Send a message and get a response.
        history: list of {"role": "user"|"assistant", "content": "..."}
        Returns: {"reply": str, "tickers_fetched": list, "context_used": bool}
        """
        if not self.available:
            return {
                "reply": "ChatAgent is unavailable — please add your OpenAI API key to the .env file.",
                "tickers_fetched": [],
                "context_used": False,
            }

        # Detect tickers and fetch live data
        tickers = self._extract_tickers(user_message)
        # Also check recent history for tickers
        if not tickers and history:
            for msg in history[-3:]:
                tickers = self._extract_tickers(msg.get("content", ""))
                if tickers:
                    break

        live_context = self._build_live_context(tickers)

        # Build system message
        system = SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d"))
        if live_context:
            system += f"\n\n--- LIVE MARKET DATA (fetched now) ---\n{live_context}\n---"

        # Build messages
        messages = [{"role": "system", "content": system}]
        # Add history (last 10 turns to stay within token limits)
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.5,
                max_tokens=800,
            )
            reply = response.choices[0].message.content.strip()
            return {
                "reply": reply,
                "tickers_fetched": tickers,
                "context_used": bool(live_context),
            }
        except Exception as e:
            # Fallback: answer without GPT
            return {
                "reply": f"Sorry, I couldn't connect to GPT-4o right now ({e}). Please check your API key or billing.",
                "tickers_fetched": tickers,
                "context_used": False,
            }

    # ── Quick news fetch ────────────────────────────────────────────────────────
    def get_market_news(self, ticker: str = None, max_articles: int = 8) -> list[dict]:
        """Fetch and return formatted news for display."""
        tickers_to_check = [ticker] if ticker else ["SPY", "QQQ"]
        all_news = []
        for t in tickers_to_check:
            news = self.data_agent.fetch_news(t, max_articles=max_articles)
            for article in news:
                article["ticker"] = t
            all_news.extend(news)
        return all_news[:max_articles]
