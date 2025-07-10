from src.agents.risk_management_team import DebateCoordinatorAgent
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# ---------------------------
# Sample TradeProposal Input
# ---------------------------
sample_input = {
    "trade_id": "TP-001",
    "ticker": "NVDA",
    "action": "buy",
    "quantity": 100,
    "price": 125.0,
    "estimated_cost": 12500.0,

    # --- Portfolio snapshot ---
    "portfolio_value": 100000.0,
    "cash_balance": 30000.0,
    "holdings": {"AAPL": 25000.0, "MSFT": 20000.0},
    "sector_exposure": {"tech": 0.6, "energy": 0.3, "healthcare": 0.1},
    "daily_returns": [0.001, -0.002, 0.003, -0.001, 0.002],
    "unrealized_pnl": 1500.0,

    # --- Simulated impact ---
    "new_position_size": 0.125,  # 12.5% of portfolio
    "new_cash_balance": 17500.0,
    "simulated_drawdown": 0.03,

    # --- Market context ---
    "volatility": 0.28,
    "avg_volume": 5000000,
    "correlation_with_portfolio": 0.75,
    "upcoming_events": ["NVDA earnings call", "Fed rate announcement"],
    "sentiment": "positive",

    # --- Meta ---
    "reason_for_trade": "Capitalizing on NVIDIA's dominant position in AI chip market",
    "timestamp": "2025-07-10T12:30:00Z"
}

# ---------------------------
# Run Coordinator Agent
# ---------------------------
coordinator = DebateCoordinatorAgent(n_rounds=4)
result = coordinator.run_debate(sample_input)

# ---------------------------
# Output Results
# ---------------------------
print("\n📄 Analyst Final Arguments:")
for analyst in result.analyst_responses:
    print(f"\n🧠 {analyst.role.capitalize()} Analyst:\n{analyst.final_argument}")

print("\n📜 Final Decision:")
final = result.final_decision
print(f"decision: {final.decision}")
print(f"reason: {final.reason}")
print(f"recommendation: {final.recommendation}")
print(f"confidence: {final.confidence}")
print(f"notes: {final.notes}")

print(f"\n🧮 Total Tokens Used: {coordinator.get_total_tokens_used()}")
