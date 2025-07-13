from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from src.agents.risk_management_team import NeutralDebatorAgent

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
neutral_node = NeutralDebatorAgent(llm=llm)

# ------------------------------
# ✅ LangGraph setup
# ------------------------------
graph = StateGraph(state_schema=dict)
graph.add_node("neutral", neutral_node)
graph.set_entry_point("neutral")
graph.set_finish_point("neutral")
pipeline = graph.compile()

# ------------------------------
# ✅ Mock input state
# ------------------------------
mock_state = {
    "ticker": "TSLA",
    "action": "BUY",
    "quantity": 15,
    "price": 270.0,
    "portfolio_value": 120000,
    "cash_balance": 15000,
    "simulated_drawdown": "Medium",
    "correlation_with_portfolio": "Low",

    "reason_for_trade": "TSLA expanding charging network, strong EV tailwinds, potential AI catalyst",
    "key_risks": "Competition in China, valuation premium, leadership controversies",
    "risk_opportunities": "FSD progress, margin expansion, battery breakthroughs",
    "volatility_indicators": "High beta, strong RSI",
    "financial_flags": "Insider selling, high PE",
    "negative_news_themes": "Tesla sued over Autopilot ad claims, recalls in Europe",
    "overall_risk_assessment": "Aggressive but strategic",

    "current_risky_response": "- Massive global upside if Tesla leads EV + AI convergence\n- Ignoring TSLA now is like ignoring AMZN in 2005",
    "current_safe_response": "- Valuation unsustainable if growth slows\n- Safer opportunities exist in diversified ETFs or blue chips",
    "history": "- Previous TSLA swing trades were volatile but profitable in short bursts"
}

# ------------------------------
# ✅ Run and test
# ------------------------------
if __name__ == "__main__":
    print("🔍 Running NeutralDebatorAgent in LangGraph...")
    result = pipeline.invoke(mock_state)

    print("\n✅ Neutral Response:")
    print(result.get("neutral_debate_response", "❌ No output returned."))
