import uuid
from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel
import ast

from src.schemas.researcher_schemas import DebateResult
from src.schemas.trader_agent_schemas import TradeProposal


class TraderAgent:
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def safe_float(self, prompt: str, allow_empty=False, default=None) -> float:
        while True:
            try:
                value = input(prompt).strip()
                if value == "" and allow_empty:
                    return default
                return float(value)
            except ValueError:
                print("❌ Please enter a valid number.")

    def safe_literal(self, prompt: str, expected_type):
        while True:
            try:
                val = input(prompt).strip()
                if not val:
                    return expected_type()
                parsed = ast.literal_eval(val)
                if isinstance(parsed, expected_type):
                    return parsed
                else:
                    raise ValueError()
            except Exception:
                print(f"❌ Please enter a valid {expected_type.__name__} (e.g., {{'AAPL': 5000}} or [0.01, -0.002]).")

    def get_user_input(self) -> Dict:
        print("📊 --- Provide Portfolio Info ---")
        ticker = input("Ticker: ").strip().upper() or "NVDA"
        portfolio_value = self.safe_float("Total portfolio value ($): ")
        cash_balance = self.safe_float("Available cash ($): ")
        holdings = self.safe_literal("Current holdings (e.g., {'AAPL': 5000}): ", dict)
        sector_exposure = self.safe_literal("Sector exposure (e.g., {'tech': 0.5, 'energy': 0.2}): ", dict)
        daily_returns = self.safe_literal("Recent daily returns (e.g., [0.01, -0.002, 0.003]): ", list)

        print("\n📈 --- Market Context (Optional, press Enter to skip) ---")
        volatility = self.safe_float("Volatility (annualized): ", allow_empty=True, default=None)
        avg_volume = self.safe_float("Average volume: ", allow_empty=True, default=None)
        correlation = self.safe_float("Correlation with portfolio: ", allow_empty=True, default=None)
        events = input("Upcoming events (comma-separated): ").strip()
        sentiment = input("Market sentiment (positive / neutral / negative): ").strip().lower()
        sentiment = sentiment if sentiment in ["positive", "neutral", "negative"] else None

        return {
            "ticker": ticker,
            "portfolio_value": portfolio_value,
            "cash_balance": cash_balance,
            "holdings": holdings,
            "sector_exposure": sector_exposure,
            "daily_returns": daily_returns,
            "volatility": volatility,
            "avg_volume": avg_volume,
            "correlation_with_portfolio": correlation,
            "upcoming_events": [e.strip() for e in events.split(",") if e.strip()],
            "sentiment": sentiment
        }

    def generate_proposal(self, debate: DebateResult, user_info: Dict) -> TradeProposal:
        winner = debate.winner
        action = "buy" if winner == "Bullish" else "sell" if winner == "Bearish" else "hold"

        price = self.safe_float(f"Current price of {user_info['ticker']}: ")

        if action == "buy":
            max_affordable = user_info["cash_balance"] // price
            quantity = self.safe_float(f"How many shares to buy? (max {max_affordable}): ", allow_empty=True, default=max_affordable)
        elif action == "sell":
            current_holding = user_info["holdings"].get(user_info["ticker"], 0)
            quantity = self.safe_float(f"How many shares to sell? (max {current_holding}): ", allow_empty=True, default=current_holding)
        else:
            quantity = 0.0

        estimated_cost = quantity * price
        new_cash_balance = user_info["cash_balance"] - estimated_cost if action == "buy" else user_info["cash_balance"] + estimated_cost
        new_position_size = (
            (user_info["holdings"].get(user_info["ticker"], 0) + (quantity if action == "buy" else -quantity)) * price
        ) / user_info["portfolio_value"]

        return TradeProposal(
            trade_id=str(uuid.uuid4())[:8],
            ticker=user_info["ticker"],
            action=action,
            quantity=quantity,
            price=price,
            estimated_cost=estimated_cost,
            portfolio_value=user_info["portfolio_value"],
            cash_balance=user_info["cash_balance"],
            holdings=user_info["holdings"],
            sector_exposure=user_info["sector_exposure"],
            daily_returns=user_info["daily_returns"],
            new_position_size=round(new_position_size, 4),
            new_cash_balance=round(new_cash_balance, 2),
            volatility=user_info.get("volatility"),
            avg_volume=user_info.get("avg_volume"),
            correlation_with_portfolio=user_info.get("correlation_with_portfolio"),
            upcoming_events=user_info.get("upcoming_events"),
            sentiment=user_info.get("sentiment"),
            reason_for_trade=f"Based on debate result: {debate.winner} argument was stronger.",
            timestamp=datetime.now().isoformat()
        )

    def run(self, debate_result: DebateResult):
        user_info = self.get_user_input()
        proposal = self.generate_proposal(debate_result, user_info)
        print("\n✅ Final Trade Proposal:\n")
        print(proposal.model_dump_json(indent=2))

