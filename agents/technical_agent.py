# ─── agents/technical_agent.py ────────────────────────────────────────────────
"""
Technical Analysis Agent
Calculates RSI, MACD, Moving Averages, Bollinger Bands, and derives signals.
"""

import pandas as pd
import numpy as np
from config import RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, MA_SHORT, MA_LONG, RSI_OVERSOLD, RSI_OVERBOUGHT


class TechnicalAgent:
    def __init__(self):
        self.name = "Technical Analysis Agent"

    # ── Indicators ──────────────────────────────────────────────────────────────
    def compute_rsi(self, df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame(
            {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram},
            index=df.index,
        )

    def compute_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "MA20": df["Close"].rolling(MA_SHORT).mean(),
                "MA50": df["Close"].rolling(MA_LONG).mean(),
                "EMA20": df["Close"].ewm(span=MA_SHORT, adjust=False).mean(),
            },
            index=df.index,
        )

    def compute_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        ma = df["Close"].rolling(period).mean()
        std = df["Close"].rolling(period).std()
        return pd.DataFrame(
            {"BB_Upper": ma + 2 * std, "BB_Middle": ma, "BB_Lower": ma - 2 * std},
            index=df.index,
        )

    def compute_volume_signal(self, df: pd.DataFrame) -> str:
        avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
        last_vol = df["Volume"].iloc[-1]
        if last_vol > avg_vol * 1.5:
            return "High Volume (strong move)"
        elif last_vol < avg_vol * 0.5:
            return "Low Volume (weak move)"
        return "Normal Volume"

    # ── Signal derivation ───────────────────────────────────────────────────────
    def derive_signals(
        self, rsi: pd.Series, macd_df: pd.DataFrame, mas: pd.DataFrame, price: float
    ) -> dict:
        signals = {}

        # RSI signal
        rsi_val = rsi.iloc[-1]
        if rsi_val < RSI_OVERSOLD:
            signals["rsi"] = {"value": round(rsi_val, 2), "signal": "BUY", "reason": "Oversold"}
        elif rsi_val > RSI_OVERBOUGHT:
            signals["rsi"] = {"value": round(rsi_val, 2), "signal": "SELL", "reason": "Overbought"}
        else:
            signals["rsi"] = {"value": round(rsi_val, 2), "signal": "HOLD", "reason": "Neutral"}

        # MACD signal
        macd_val = macd_df["MACD"].iloc[-1]
        sig_val = macd_df["Signal"].iloc[-1]
        hist_val = macd_df["Histogram"].iloc[-1]
        prev_hist = macd_df["Histogram"].iloc[-2] if len(macd_df) > 1 else 0
        if macd_val > sig_val and hist_val > 0 and prev_hist <= 0:
            macd_signal = "BUY"
        elif macd_val < sig_val and hist_val < 0 and prev_hist >= 0:
            macd_signal = "SELL"
        elif macd_val > sig_val:
            macd_signal = "BUY"
        else:
            macd_signal = "SELL"
        signals["macd"] = {
            "macd": round(float(macd_val), 4),
            "signal_line": round(float(sig_val), 4),
            "histogram": round(float(hist_val), 4),
            "signal": macd_signal,
        }

        # Moving average crossover
        ma20 = mas["MA20"].iloc[-1]
        ma50 = mas["MA50"].iloc[-1]
        prev_ma20 = mas["MA20"].iloc[-2] if len(mas) > 1 else ma20
        prev_ma50 = mas["MA50"].iloc[-2] if len(mas) > 1 else ma50
        if ma20 > ma50 and prev_ma20 <= prev_ma50:
            ma_signal = "BUY"
            ma_reason = "Golden Cross (MA20 crossed above MA50)"
        elif ma20 < ma50 and prev_ma20 >= prev_ma50:
            ma_signal = "SELL"
            ma_reason = "Death Cross (MA20 crossed below MA50)"
        elif ma20 > ma50:
            ma_signal = "BUY"
            ma_reason = "MA20 above MA50 (uptrend)"
        else:
            ma_signal = "SELL"
            ma_reason = "MA20 below MA50 (downtrend)"
        signals["moving_average"] = {
            "MA20": round(float(ma20), 2),
            "MA50": round(float(ma50), 2),
            "signal": ma_signal,
            "reason": ma_reason,
        }

        # Price vs MA20 (trend filter)
        if price > float(ma20):
            signals["trend"] = "Bullish (price above MA20)"
        else:
            signals["trend"] = "Bearish (price below MA20)"

        return signals

    def score_signals(self, signals: dict) -> int:
        """
        Convert signals to a numeric score: +1 BUY, -1 SELL, 0 HOLD.
        Range: -3 to +3.
        """
        score = 0
        for key in ["rsi", "macd", "moving_average"]:
            s = signals.get(key, {}).get("signal", "HOLD")
            if s == "BUY":
                score += 1
            elif s == "SELL":
                score -= 1
        return score

    # ── Main entry ──────────────────────────────────────────────────────────────
    def run(self, stock_data: dict) -> dict:
        df = stock_data.get("stock_df", pd.DataFrame())
        if df.empty or len(df) < MA_LONG + 5:
            return {"status": "insufficient_data", "signals": {}, "score": 0}

        price = stock_data.get("price_info", {}).get("current_price", float(df["Close"].iloc[-1].item()))

        rsi = self.compute_rsi(df)
        macd_df = self.compute_macd(df)
        mas = self.compute_moving_averages(df)
        bb = self.compute_bollinger_bands(df)
        vol_signal = self.compute_volume_signal(df)
        signals = self.derive_signals(rsi, macd_df, mas, price)
        score = self.score_signals(signals)

        return {
            "status": "ok",
            "rsi": rsi,
            "macd_df": macd_df,
            "mas": mas,
            "bb": bb,
            "signals": signals,
            "volume_signal": vol_signal,
            "score": score,
            "df_with_indicators": df.join(mas).join(macd_df).join(bb).assign(RSI=rsi),
        }
