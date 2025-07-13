from src.agents.trader_agents import TraderAgent
from src.schemas.researcher_schemas import DebateResult, DebateTurn
from langgraph.graph import StateGraph, END

# -------------------------------
# ✅ Mock DebateResult
# -------------------------------
mock_debate_result = DebateResult(
    turns=[
        DebateTurn(speaker="Bullish", message="MSFT will grow with AI and cloud.", tokens_used=100),
        DebateTurn(speaker="Bearish", message="Regulatory issues pose a threat.", tokens_used=90)
    ],
    summary="Bullish arguments emphasize AI growth; Bearish raises regulation risks.",
    winner="Bullish",
    total_tokens=190
)

# -------------------------------
# ✅ Mock State (no price needed!)
# -------------------------------
mock_state = {
    "research_debate_result": mock_debate_result,
    "ticker": "MSFT",
    "portfolio_value": 100000,
    "cash_balance": 25000,
    "holdings": {"MSFT": 50},
    "sector_exposure": {"tech": 0.5, "finance": 0.2},
    "daily_returns": [0.01, -0.005, 0.003],
    "volatility": 0.25,
    "avg_volume": 1000000,
    "correlation_with_portfolio": 0.6,
    "upcoming_events": ["Earnings call", "AI developer day"],
    "sentiment": "positive",
    "quantity": 10,  # Optional — auto-calculated if omitted
}

# -------------------------------
# ✅ Build and Compile LangGraph
# -------------------------------
agent = TraderAgent()
graph = StateGraph(dict)
graph.add_node("trade", agent)
graph.set_entry_point("trade")
graph.add_edge("trade", END)
compiled = graph.compile()

# -------------------------------
# ✅ Run and Inspect Output
# -------------------------------
print("\n🚀 Running TraderAgent Test with Mock Debate + Portfolio State...\n")
result = compiled.invoke(mock_state)

proposal = result.get("trade_proposal")
print("\n🪵 Full Returned State:")
print(result)

if proposal:
    print("\n✅ Final Trade Proposal:")
    print(proposal.model_dump_json(indent=2))
else:
    print("❌ No trade proposal generated.")
