# ─── agents/prediction_agent.py ───────────────────────────────────────────────
"""
Prediction Agent
Trains a Random Forest + Linear Regression ensemble on historical OHLCV +
technical features, then predicts the next N-day closing price.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from config import PREDICT_DAYS


class PredictionAgent:
    def __init__(self):
        self.name = "Prediction Agent"
        self.models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Linear Regression": Ridge(alpha=1.0),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        self.scaler = StandardScaler()
        self._trained = False

    # ── Feature engineering ─────────────────────────────────────────────────────
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        close = df["Close"]

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            feat[f"lag_{lag}"] = close.shift(lag)

        # Rolling statistics
        for w in [5, 10, 20]:
            feat[f"roll_mean_{w}"] = close.rolling(w).mean()
            feat[f"roll_std_{w}"] = close.rolling(w).std()

        # Returns
        feat["return_1d"] = close.pct_change(1)
        feat["return_5d"] = close.pct_change(5)

        # OHLCV derived
        feat["hl_ratio"] = (df["High"] - df["Low"]) / close
        feat["oc_ratio"] = (df["Close"] - df["Open"]) / df["Open"]
        feat["volume_ma"] = df["Volume"].rolling(10).mean()

        # RSI (inline)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        feat["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        feat["macd"] = (
            close.ewm(span=12, adjust=False).mean()
            - close.ewm(span=26, adjust=False).mean()
        )

        return feat

    # ── Training ────────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame) -> dict:
        """
        Train all models; return accuracy metrics.
        Target = Close price shifted back by PREDICT_DAYS (predict N days ahead).
        """
        features = self._build_features(df)
        target = df["Close"].shift(-PREDICT_DAYS)  # future close

        combined = features.join(target.rename("target")).dropna()
        if len(combined) < 60:
            return {"status": "insufficient_data"}

        X = combined.drop(columns=["target"])
        y = combined["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        metrics = {}
        for name, model in self.models.items():
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            mape = mean_absolute_percentage_error(y_test, preds)
            metrics[name] = {"MAPE": round(mape * 100, 2), "accuracy": round((1 - mape) * 100, 2)}

        self._trained = True
        self._feature_cols = list(X.columns)
        return {"status": "ok", "metrics": metrics}

    # ── Prediction ──────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict closing price PREDICT_DAYS ahead; ensemble average of all models.
        """
        if not self._trained:
            train_result = self.train(df)
            if train_result.get("status") != "ok":
                return {"status": "insufficient_data", "predicted_price": None}

        features = self._build_features(df)
        latest = features.dropna().iloc[[-1]]
        latest_s = self.scaler.transform(latest[self._feature_cols])

        preds = {}
        for name, model in self.models.items():
            preds[name] = round(float(model.predict(latest_s)[0]), 2)

        ensemble = round(float(np.mean(list(preds.values()))), 2)
        current = float(df["Close"].iloc[-1])
        change_pct = round((ensemble - current) / current * 100, 2)

        if change_pct > 2:
            signal = "BUY"
        elif change_pct < -2:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "status": "ok",
            "current_price": round(current, 2),
            "predicted_price": ensemble,
            "change_pct": change_pct,
            "predict_days": PREDICT_DAYS,
            "individual_predictions": preds,
            "signal": signal,
            "score": 1 if signal == "BUY" else (-1 if signal == "SELL" else 0),
        }

    # ── Main entry ──────────────────────────────────────────────────────────────
    def run(self, stock_data: dict) -> dict:
        df = stock_data.get("stock_df", pd.DataFrame())
        if df.empty or len(df) < 60:
            return {
                "status": "insufficient_data",
                "predicted_price": None,
                "signal": "HOLD",
                "score": 0,
            }
        train_result = self.train(df)
        pred_result = self.predict(df)
        return {**train_result, **pred_result}
