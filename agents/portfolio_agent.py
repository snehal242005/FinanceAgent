# ─── agents/portfolio_agent.py ────────────────────────────────────────────────
"""
Portfolio Agent
Persists a simple JSON portfolio (ticker, qty, buy_price, buy_date).
Calculates current P&L using live yfinance prices.
"""

import json
import os
from datetime import datetime
import yfinance as yf
from config import PORTFOLIO_FILE


def _load() -> list[dict]:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save(portfolio: list[dict]) -> None:
    os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)


class PortfolioAgent:
    def __init__(self):
        self.name = "Portfolio Agent"

    # ── CRUD ─────────────────────────────────────────────────────────────────────
    def get_portfolio(self) -> list[dict]:
        return _load()

    def add_position(self, ticker: str, qty: float, buy_price: float) -> dict:
        portfolio = _load()
        # Check if ticker already exists → update average
        for pos in portfolio:
            if pos["ticker"].upper() == ticker.upper():
                old_cost = pos["qty"] * pos["buy_price"]
                new_cost = qty * buy_price
                total_qty = pos["qty"] + qty
                pos["buy_price"] = round((old_cost + new_cost) / total_qty, 4)
                pos["qty"] = total_qty
                pos["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _save(portfolio)
                return {"status": "updated", "position": pos}

        position = {
            "ticker": ticker.upper(),
            "qty": qty,
            "buy_price": buy_price,
            "buy_date": datetime.now().strftime("%Y-%m-%d"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        portfolio.append(position)
        _save(portfolio)
        return {"status": "added", "position": position}

    def remove_position(self, ticker: str) -> dict:
        portfolio = _load()
        original_len = len(portfolio)
        portfolio = [p for p in portfolio if p["ticker"].upper() != ticker.upper()]
        _save(portfolio)
        removed = original_len - len(portfolio)
        return {"status": "removed" if removed else "not_found", "ticker": ticker}

    def update_quantity(self, ticker: str, new_qty: float) -> dict:
        portfolio = _load()
        for pos in portfolio:
            if pos["ticker"].upper() == ticker.upper():
                if new_qty <= 0:
                    return self.remove_position(ticker)
                pos["qty"] = new_qty
                pos["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _save(portfolio)
                return {"status": "updated", "position": pos}
        return {"status": "not_found", "ticker": ticker}

    # ── P&L calculation ──────────────────────────────────────────────────────────
    def calculate_pnl(self) -> list[dict]:
        portfolio = _load()
        if not portfolio:
            return []

        enriched = []
        for pos in portfolio:
            ticker = pos["ticker"]
            try:
                info = yf.Ticker(ticker).fast_info
                current_price = round(float(info.last_price), 2)
            except Exception:
                current_price = pos["buy_price"]

            cost_basis = pos["qty"] * pos["buy_price"]
            market_value = pos["qty"] * current_price
            pnl = market_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0

            enriched.append(
                {
                    **pos,
                    "current_price": current_price,
                    "cost_basis": round(cost_basis, 2),
                    "market_value": round(market_value, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "status": "Profit" if pnl >= 0 else "Loss",
                }
            )

        return enriched

    def portfolio_summary(self) -> dict:
        positions = self.calculate_pnl()
        if not positions:
            return {
                "total_invested": 0,
                "total_market_value": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "positions": [],
            }
        total_invested = sum(p["cost_basis"] for p in positions)
        total_value = sum(p["market_value"] for p in positions)
        total_pnl = total_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0
        return {
            "total_invested": round(total_invested, 2),
            "total_market_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "positions": positions,
        }
