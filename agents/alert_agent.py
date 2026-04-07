# ─── agents/alert_agent.py ────────────────────────────────────────────────────
"""
Alert Agent
Stores user-defined alerts (price threshold, sentiment change, decision change).
Checks alerts against live data and returns triggered alerts.
"""

import json
import os
from datetime import datetime
import yfinance as yf
from config import ALERTS_FILE


def _load() -> list[dict]:
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save(alerts: list[dict]) -> None:
    os.makedirs(os.path.dirname(ALERTS_FILE), exist_ok=True)
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=2)


class AlertAgent:
    def __init__(self):
        self.name = "Alert Agent"

    # ── CRUD ──────────────────────────────────────────────────────────────────
    def get_alerts(self) -> list[dict]:
        return _load()

    def add_alert(
        self,
        ticker: str,
        alert_type: str,          # "price_above" | "price_below" | "decision_change" | "sentiment_change"
        threshold: float | None = None,
        note: str = "",
    ) -> dict:
        alerts = _load()
        alert = {
            "id": len(alerts) + 1,
            "ticker": ticker.upper(),
            "type": alert_type,
            "threshold": threshold,
            "note": note,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "triggered": False,
            "last_checked": None,
        }
        alerts.append(alert)
        _save(alerts)
        return {"status": "added", "alert": alert}

    def delete_alert(self, alert_id: int) -> dict:
        alerts = _load()
        original = len(alerts)
        alerts = [a for a in alerts if a.get("id") != alert_id]
        _save(alerts)
        return {"status": "deleted" if len(alerts) < original else "not_found"}

    def reset_alert(self, alert_id: int) -> dict:
        alerts = _load()
        for a in alerts:
            if a.get("id") == alert_id:
                a["triggered"] = False
                a["last_checked"] = None
        _save(alerts)
        return {"status": "reset"}

    # ── Checking ───────────────────────────────────────────────────────────────
    def check_alerts(
        self,
        current_prices: dict[str, float] | None = None,
        current_decisions: dict[str, str] | None = None,
        current_sentiments: dict[str, str] | None = None,
    ) -> list[dict]:
        """
        Check all active alerts against provided live data.
        Returns list of triggered alert dicts with a 'message' field.
        """
        alerts = _load()
        triggered = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for alert in alerts:
            if alert.get("triggered"):
                continue  # skip already-triggered alerts

            ticker = alert["ticker"]
            atype = alert["type"]
            threshold = alert.get("threshold")
            fired = False
            message = ""

            # Price alerts — fetch live if not provided
            if atype in ("price_above", "price_below"):
                price = None
                if current_prices and ticker in current_prices:
                    price = current_prices[ticker]
                else:
                    try:
                        price = float(yf.Ticker(ticker).fast_info.last_price)
                    except Exception:
                        pass

                if price is not None and threshold is not None:
                    if atype == "price_above" and price >= threshold:
                        fired = True
                        message = f"{ticker} price ${price:.2f} crossed ABOVE ${threshold:.2f}"
                    elif atype == "price_below" and price <= threshold:
                        fired = True
                        message = f"{ticker} price ${price:.2f} dropped BELOW ${threshold:.2f}"

            elif atype == "decision_change" and current_decisions:
                decision = current_decisions.get(ticker)
                expected = threshold  # repurposed field for expected decision string
                if decision and str(decision) != str(expected):
                    fired = True
                    message = f"{ticker} decision changed to {decision} (was {expected})"

            elif atype == "sentiment_change" and current_sentiments:
                sentiment = current_sentiments.get(ticker)
                expected = threshold
                if sentiment and str(sentiment) != str(expected):
                    fired = True
                    message = f"{ticker} sentiment changed to {sentiment} (was {expected})"

            alert["last_checked"] = now
            if fired:
                alert["triggered"] = True
                alert["triggered_at"] = now
                alert["message"] = message
                triggered.append({**alert})

        _save(alerts)
        return triggered

    def check_portfolio_alerts(self, portfolio_positions: list[dict]) -> list[dict]:
        """Check price alerts for all portfolio positions and return any triggered."""
        prices = {p["ticker"]: p.get("current_price", 0) for p in portfolio_positions}
        return self.check_alerts(current_prices=prices)
