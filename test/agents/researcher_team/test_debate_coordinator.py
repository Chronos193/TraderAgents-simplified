from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.agents.risk_management_team import DebateCoordinatorAgent
import os
from dotenv import load_dotenv

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("🧪 Running DebateCoordinatorAgent with mock data...")

# Define the state schema for LangGraph
class State(TypedDict):
    ticker: str
    fundamentals: str
    technical: str
    news: str
    sentiment: str
    reason_for_trade: str
    portfolio_value: float
    cash_balance: float
    quantity: int
    price: float
    simulated_drawdown: str
    correlation_with_portfolio: str
    action: str

# Mock input
mock_state: State = {
    "ticker": "MSFT",
    "fundamentals": """
- Revenue growth slowed to 8% YoY
- Slight increase in long-term debt
- Cloud segment growth decelerating
""",
    "technical": """
- RSI: 71 (overbought)
- MACD weakening
- Support at $320, resistance at $350
""",
    "news": """
- SEC investigating cloud contract disclosures
- EU pushing antitrust charges against Microsoft
- Positive coverage on AI integration with OpenAI
""",
    "sentiment": """
- Retail bullishness remains strong
- Analyst targets range from $360 to $400
""",
    "reason_for_trade": "Buy MSFT due to expected AI-driven growth and continued cloud dominance.",
    "portfolio_value": 100000,
    "cash_balance": 25000,
    "quantity": 50,
    "price": 340,
    "simulated_drawdown": "-4.8%",
    "correlation_with_portfolio": "0.63",
    "action": "buy"
}

# Build and run LangGraph
agent = DebateCoordinatorAgent(n_rounds=1)
graph = StateGraph(State)
graph.add_node("debate", agent.as_runnable_node())
graph.set_entry_point("debate")
graph.add_edge("debate", END)
compiled = graph.compile()

# Run the pipeline
result = compiled.invoke(mock_state)

# Retrieve and inspect the output
output = result.get("debate_coordinator_output")

print("\n✅ Final Parsed Output:")
if output is None or agent.error_occurred():
    print("❌ DebateCoordinatorAgent returned None.")
    print("🪵 Raw LLM Output:\n", agent.last_summary_raw_output)
else:
    print("Final Decision:", output.final_decision)
    print("Confidence:", output.confidence)
    print("Recommendation:", output.recommendation)
    print("Rationale:", output.rationale)
