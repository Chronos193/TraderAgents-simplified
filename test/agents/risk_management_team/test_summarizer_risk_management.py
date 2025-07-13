from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from src.agents.risk_management_team import RiskSummarizerAgent

# --------------------------
# ✅ Load .env + Groq key
# --------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.3)

# --------------------------
# ✅ Create summarizer node
# --------------------------
summarizer_node = RiskSummarizerAgent(llm=llm)

# --------------------------
# ✅ Create LangGraph wrapper
# --------------------------
graph = StateGraph(state_schema=dict)
graph.add_node("summarize_risk", summarizer_node)
graph.set_entry_point("summarize_risk")
graph.set_finish_point("summarize_risk")
pipeline = graph.compile()

# --------------------------
# ✅ Mock input state
# --------------------------
mock_state = {
    "ticker": "MSFT",
    "fundamentals": (
        "Revenue grew 7%, but cloud segment slowed. Free cash flow stable. "
        "R&D up 12%, but debt slightly increased. PE ratio at 29."
    ),
    "technical": (
        "RSI at 71, indicating overbought. MACD shows weakening bullish momentum. "
        "Stock above 50 and 200 DMA. Slight divergence noted."
    ),
    "news": (
        "SEC is investigating MSFT's cloud contracts. Antitrust concerns in EU. "
        "Positive press on AI collaboration with OpenAI."
    ),
    "sentiment": (
        "Analyst tone is optimistic. Retail sentiment high. Insider selling activity observed. "
        "Hedge fund exposure dropped 3%."
    )
}

# --------------------------
# ✅ Run test
# --------------------------
if __name__ == "__main__":
    print("🧪 Running RiskSummarizerAgent with mock data...")
    result = pipeline.invoke(mock_state)

    risk_summary = result.get("risk_summary")
    if risk_summary:
        print("\n✅ Risk Summary Output:")
        print(risk_summary.model_dump_json(indent=2))
    else:
        print("❌ No risk summary returned.")
