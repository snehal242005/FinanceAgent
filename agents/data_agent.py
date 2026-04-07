# ─── agents/data_agent.py ─────────────────────────────────────────────────────
"""
Data Agent
Responsibility: Fetch historical + real-time stock data (yfinance) and
                financial news headlines (Yahoo Finance RSS – no API key).
"""

import feedparser
import yfinance as yf
import pandas as pd
from datetime import datetime
from config import HISTORY_PERIOD


class DataAgent:
    def __init__(self):
        self.name = "Data Agent"

    # ── Stock data ──────────────────────────────────────────────────────────────
    def fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        """Return OHLCV DataFrame for *ticker* over HISTORY_PERIOD."""
        try:
            df = yf.download(ticker, period=HISTORY_PERIOD, auto_adjust=True, progress=False)
            if df.empty:
                return pd.DataFrame()
            # yfinance ≥0.2.x may return MultiIndex columns — flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"[DataAgent] Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_current_price(self, ticker: str) -> dict:
        """Return current price + basic info dict."""
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            return {
                "ticker": ticker,
                "current_price": round(float(info.last_price), 2),
                "previous_close": round(float(info.previous_close), 2),
                "market_cap": info.market_cap,
                "currency": info.currency,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            print(f"[DataAgent] Error fetching price for {ticker}: {e}")
            return {}

    def fetch_company_info(self, ticker: str) -> dict:
        """Return company metadata."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            return {
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "country": info.get("country", "N/A"),
                "description": info.get("longBusinessSummary", "")[:500],
                "pe_ratio": info.get("trailingPE", None),
                "52w_high": info.get("fiftyTwoWeekHigh", None),
                "52w_low": info.get("fiftyTwoWeekLow", None),
                "dividend_yield": info.get("dividendYield", None),
            }
        except Exception as e:
            print(f"[DataAgent] Error fetching company info for {ticker}: {e}")
            return {"name": ticker}

    # ── News data ───────────────────────────────────────────────────────────────
    def fetch_news(self, ticker: str, max_articles: int = 10) -> list[dict]:
        """
        Fetch news headlines from Yahoo Finance RSS for *ticker*.
        Returns list of {"title", "summary", "published", "link"}.
        """
        url = (
            f"https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={ticker}&region=US&lang=en-US"
        )
        articles = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_articles]:
                articles.append(
                    {
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", entry.get("title", "")),
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                    }
                )
        except Exception as e:
            print(f"[DataAgent] Error fetching news for {ticker}: {e}")

        # Fallback: generic market RSS if ticker-specific feed is empty
        if not articles:
            try:
                feed = feedparser.parse("https://feeds.a.dj.com/rss/RSSMarketsMain.xml")
                for entry in feed.entries[:max_articles]:
                    articles.append(
                        {
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", entry.get("title", "")),
                            "published": entry.get("published", ""),
                            "link": entry.get("link", ""),
                        }
                    )
            except Exception:
                pass

        return articles

    # ── Summary report ──────────────────────────────────────────────────────────
    def run(self, ticker: str) -> dict:
        """Full data collection – called by the orchestrator."""
        stock_df = self.fetch_stock_data(ticker)
        price_info = self.fetch_current_price(ticker)
        company_info = self.fetch_company_info(ticker)
        news = self.fetch_news(ticker)

        return {
            "ticker": ticker,
            "stock_df": stock_df,
            "price_info": price_info,
            "company_info": company_info,
            "news": news,
            "status": "ok" if not stock_df.empty else "error",
        }
