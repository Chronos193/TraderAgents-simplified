from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from src.agents.risk_management_team import AggressiveDebatorAgent

# ------------------------------
# ✅ Load environment + Groq key
# ------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY in .env")

llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.3)

# ------------------------------
# ✅ Create agent instance
# ------------------------------
aggressive_node = AggressiveDebatorAgent(llm=llm)

# ------------------------------
# ✅ LangGraph setup
# ------------------------------
graph = StateGraph(state_schema=dict)
graph.add_node("aggressive", aggressive_node)
graph.set_entry_point("aggressive")
graph.set_finish_point("aggressive")
pipeline = graph.compile()

# ------------------------------
# ✅ Mock input state
# ------------------------------
mock_state = {
    "ticker": "TSLA",
    "action": "BUY",
    "quantity": 10,
    "price": 265.5,
    "portfolio_value": 150000,
    "cash_balance": 25000,
    "simulated_drawdown": "Moderate",
    "correlation_with_portfolio": "Low",
    "reason_for_trade": "Positive delivery numbers, expansion in India market",
    "volatility": "High",
    "avg_volume": "32M",
    "upcoming_events": "AI day next week",
    "sentiment": "Bullish",
    "sector_exposure": "Tech / Auto",
    "holdings": "AAPL, NVDA, MSFT",
    "new_position_size": "$2655",
    "new_cash_balance": "$22345",
    "key_risks": "High valuation, regulatory scrutiny",
    "risk_opportunities": "Robotaxi, FSD adoption, global EV leadership",
    "volatility_indicators": "MACD rising, strong RSI",
    "financial_flags": "Insider sales, elevated PE",
    "negative_news_themes": "CEO controversy, China EV price war",
    "overall_risk_assessment": "High but strategic",
    "current_safe_response": "- Market overvalued\n- Wait for dip",
    "current_neutral_response": "- Promising long term, but uncertain near term",
    "history": "- Last trade on TSLA made 12% in 2 weeks"
}

# ------------------------------
# ✅ Run and test
# ------------------------------
if __name__ == "__main__":
    print("🚀 Running AggressiveDebatorAgent in LangGraph...")
    result = pipeline.invoke(mock_state)

    print("\n✅ Aggressive Response:")
    print(result.get("aggressive_debate_response", "❌ No output returned."))
