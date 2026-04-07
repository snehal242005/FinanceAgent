# ─── config.py ────────────────────────────────────────────────────────────────
# Central configuration for the FinanceAgent system

# Default stocks shown in UI
DEFAULT_STOCKS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]

# News RSS feeds (no API key required)
NEWS_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]

# Technical indicator settings
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_SHORT = 20
MA_LONG = 50

# Decision thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Risk levels → affect decision aggressiveness
RISK_LEVELS = ["Conservative", "Moderate", "Aggressive"]

# Portfolio storage file
PORTFOLIO_FILE = "data/portfolio.json"

# Alert storage file
ALERTS_FILE = "data/alerts.json"

# Historical data period for ML training
HISTORY_PERIOD = "2y"

# Prediction horizon (days ahead)
PREDICT_DAYS = 7
