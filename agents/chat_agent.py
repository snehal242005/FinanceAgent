# ─── agents/chat_agent.py ─────────────────────────────────────────────────────
"""
ChatAgent — GPT-4o powered stock market chatbot.
Fetches live news + price data and answers user questions intelligently.
"""

import re
from datetime import datetime
from openai import OpenAI
from agents.data_agent import DataAgent
from utils.secrets import get_openai_key

COMMON_WORDS = {
    "I","A","AN","THE","AND","OR","IS","IN","ON","AT","TO","FOR","OF","WITH",
    "BUY","SELL","HOLD","WHAT","HOW","WHY","WHEN","WHO","CAN","WILL","DO",
    "MY","ME","US","WE","IT","BE","BY","IF","UP","AS","SO","NO","GO","GET",
    "ARE","WAS","HAS","HAD","NOW","NEW","ALL","TOP","GDP","IPO","ETF","USA",
    "AI","API","ML","AM","PM","CEO","CFO","CTO","PE","EPS","ATH","ATL","RSI",
}


class ChatAgent:
    def __init__(self):
        self.name = "TradeXAI Chatbot (GPT-4o)"
        api_key = get_openai_key()
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.data_agent = DataAgent()
        self.available = self.client is not None

    # ── Ticker detection ────────────────────────────────────────────────────────
    def extract_tickers(self, text: str) -> list[str]:
        """Pull likely stock tickers from the user message."""
        # $TICKER style
        dollar = re.findall(r'\$([A-Z]{1,5})', text.upper())
        # STANDALONE CAPS words 1-5 chars
        caps = re.findall(r'\b([A-Z]{1,5})\b', text.upper())
        candidates = list(dict.fromkeys(dollar + caps))  # dedupe, preserve order
        return [t for t in candidates if t not in COMMON_WORDS and len(t) >= 2][:3]

    # ── Live context builder ────────────────────────────────────────────────────
    def _build_context(self, message: str) -> str:
        tickers = self.extract_tickers(message)
        context_parts = []

        for ticker in tickers:
            try:
                # Price
                price = self.data_agent.fetch_current_price(ticker)
                if price and price.get("current_price"):
                    prev = price.get("previous_close", price["current_price"])
                    chg = price["current_price"] - prev
                    pct = (chg / prev * 100) if prev else 0
                    context_parts.append(
                        f"[LIVE] {ticker}: ${price['current_price']:.2f} "
                        f"({chg:+.2f} / {pct:+.2f}%)"
                    )

                # News
                news = self.data_agent.fetch_news(ticker, max_articles=5)
                if news:
                    context_parts.append(f"[LATEST NEWS for {ticker}]")
                    for n in news[:4]:
                        context_parts.append(f"  • {n['title']} ({n.get('published','')})")
            except Exception:
                pass

        return "\n".join(context_parts)

    # ── General market news ─────────────────────────────────────────────────────
    def _general_market_news(self) -> str:
        try:
            import feedparser
            feed = feedparser.parse("https://feeds.a.dj.com/rss/RSSMarketsMain.xml")
            lines = ["[GENERAL MARKET NEWS]"]
            for e in feed.entries[:5]:
                lines.append(f"  • {e.get('title','')}")
            return "\n".join(lines)
        except Exception:
            return ""

    # ── Chat ────────────────────────────────────────────────────────────────────
    def chat(self, history: list[dict], user_message: str) -> str:
        """
        history: list of {"role": "user"|"assistant", "content": "..."} dicts
        Returns assistant reply string.
        """
        if not self.available:
            return (
                "TradeXAI Chatbot requires an OpenAI API key. "
                "Please add your key to the .env file."
            )

        # Build live context
        live_ctx = self._build_context(user_message)
        if not live_ctx:
            live_ctx = self._general_market_news()

        system_prompt = f"""You are TradeXAI, a world-class AI stock market assistant.
You have access to live market data and the latest financial news.
Today: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

LIVE MARKET DATA:
{live_ctx if live_ctx else 'No specific ticker data fetched. Provide general market knowledge.'}

Guidelines:
- Answer concisely and professionally
- Use bullet points for lists
- Always mention when data is approximate or delayed
- Remind users this is NOT financial advice for investment decisions
- For specific tickers, refer to the live data above
- If asked for news, summarize the headlines above
- Be conversational but expert"""

        messages = [{"role": "system", "content": system_prompt}]
        # Include last 6 turns of history for context
        messages += history[-6:]
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.6,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)[:100]}. Please try again."

    # ── Suggested questions ─────────────────────────────────────────────────────
    def get_suggestions(self) -> list[str]:
        return [
            "What's the latest news on AAPL?",
            "Give me TSLA current price and recent headlines",
            "What are the top performing stocks today?",
            "Explain what RSI means in technical analysis",
            "Should I buy NVDA based on recent news?",
            "What's happening in the market today?",
        ]
