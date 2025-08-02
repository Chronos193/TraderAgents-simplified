import uuid
import yfinance as yf
from datetime import datetime
from typing import Dict, List

from src.schemas.researcher_schemas import DebateResult
from src.schemas.trader_agent_schemas import TradeProposal


class TraderAgent:
    def __init__(self):
        pass

    def fetch_price_from_yfinance(self, ticker: str) -> float:
        try:
            stock = yf.Ticker(ticker)
            price = stock.info.get("regularMarketPrice")
            if price is None:
                raise ValueError("Price not available for ticker.")
            return float(price)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to fetch price for {ticker}: {e}")

    def generate_proposal(self, debate: DebateResult, user_info: Dict) -> TradeProposal:
        winner = debate.winner
        
        # 🎯 IMPROVED DECISION LOGIC: Less conservative, more action-oriented
        if winner == "Bullish":
            action = "buy"
        elif winner == "Bearish":
            action = "sell" 
        else:  # winner == "Tie"
            # NEW: Instead of always defaulting to "hold", check debate content
            bullish_mentions = sum(1 for turn in debate.turns if 
                                 any(word in turn.message.lower() for word in 
                                     ['strong', 'growth', 'positive', 'bullish', 'buy', 'undervalued']))
            bearish_mentions = sum(1 for turn in debate.turns if 
                                 any(word in turn.message.lower() for word in 
                                     ['risk', 'overvalued', 'decline', 'bearish', 'sell', 'expensive']))
            
            # Tie-breaker: favor action when there's strong evidence on either side
            if bullish_mentions > bearish_mentions * 1.2:  # 20% bias toward action
                action = "buy"
                print(f"[INFO] Tie broken in favor of BUY based on debate content ({bullish_mentions} vs {bearish_mentions})")
            elif bearish_mentions > bullish_mentions * 1.2:
                action = "sell"
                print(f"[INFO] Tie broken in favor of SELL based on debate content ({bearish_mentions} vs {bullish_mentions})")
            else:
                action = "hold"

        # ✅ Fetch price
        price = self.fetch_price_from_yfinance(user_info["ticker"])
        print(f"💰 Current price of {user_info['ticker']}: ${price:.2f}")

        current_shares = user_info["holdings"].get(user_info["ticker"], 0)

        # 🔧 Position-aware logic fix
        if action == "sell" and current_shares == 0:
            print(f"[INFO] Cannot sell {user_info['ticker']} - no shares owned. Changing to HOLD.")
            action = "hold"

        # ✅ Quantity logic
        if "quantity" in user_info and user_info["quantity"] is not None:
            quantity = float(user_info["quantity"])
        elif action == "buy":
            max_shares = user_info["cash_balance"] // price
            if max_shares == 0:
                print(f"[INFO] Cannot buy {user_info['ticker']} - insufficient cash. Changing to HOLD.")
                action = "hold"
                quantity = 0.0
            else:
                quantity = max_shares
        elif action == "sell":
            quantity = min(current_shares, user_info["quantity"] or current_shares)
        else:
            quantity = 0.0

        # ✅ Estimate cost and update state
        estimated_cost = quantity * price

        if action == "buy":
            new_cash_balance = user_info["cash_balance"] - estimated_cost
            new_position_shares = current_shares + quantity
        elif action == "sell":
            quantity = min(current_shares, quantity)
            estimated_cost = quantity * price  # Recalculate in case clamped
            new_cash_balance = user_info["cash_balance"] + estimated_cost
            new_position_shares = max(current_shares - quantity, 0)
        else:  # hold
            new_cash_balance = user_info["cash_balance"]
            new_position_shares = current_shares

        new_position_size = (new_position_shares * price) / user_info["portfolio_value"]

        return TradeProposal(
            trade_id=str(uuid.uuid4())[:8],
            ticker=user_info["ticker"],
            action=action,
            quantity=quantity,
            price=round(price, 2),
            estimated_cost=round(estimated_cost, 2),
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
            timestamp=datetime.now().isoformat(),
        )



    def __call__(self, state: dict) -> dict:
        if "research_debate_result" not in state:
            print("[WARN] TraderAgent: research_debate_result missing.")
            return {**state, "trade_proposal": None}

        try:
            debate_result: DebateResult = state["research_debate_result"]

            # Ensure all required fields are present
            required_keys = [
                "ticker", "portfolio_value", "cash_balance",
                "holdings", "sector_exposure", "daily_returns"
            ]
            missing_keys = [k for k in required_keys if k not in state]
            if missing_keys:
                print(f"[ERROR] TraderAgent missing required keys: {missing_keys}")
                return {**state, "trade_proposal": None}

            user_info = {
                "ticker": state["ticker"],
                "portfolio_value": state["portfolio_value"],
                "cash_balance": state["cash_balance"],
                "holdings": state["holdings"],
                "sector_exposure": state["sector_exposure"],
                "daily_returns": state["daily_returns"],
                "volatility": state.get("volatility"),
                "avg_volume": state.get("avg_volume"),
                "correlation_with_portfolio": state.get("correlation_with_portfolio"),
                "upcoming_events": state.get("upcoming_events"),
                "sentiment": state.get("sentiment"),
                "quantity": state.get("quantity"),  # optional
            }

            proposal = self.generate_proposal(debate_result, user_info)
            return {**state, "trade_proposal": proposal}

        except Exception as e:
            print(f"[ERROR] TraderAgent failed: {e}")
            return {**state, "trade_proposal": None}
