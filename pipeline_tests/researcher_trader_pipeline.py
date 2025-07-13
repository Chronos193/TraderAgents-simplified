from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from src.agents.analyst_team.fundamentals_analyst import FundamentalsAnalyst
from src.agents.analyst_team.news_analyst import NewsAnalyst
from src.agents.analyst_team.sentiment_analyst import SentimentAnalyst
from src.agents.analyst_team.technical_analyst import TechnicalAnalystAgent

from src.agents.researcher_team.bullish_researcher import BullishResearcher
from src.agents.researcher_team.bearish_researcher import BearishResearcher
from src.agents.researcher_team.debate_coordinator import DebateCoordinator

from src.agents.trader_agents import TraderAgent  # ✅ NEW

# ----------------------
# ✅ Load API keys
# ----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env")
if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY missing in .env")

# ----------------------
# ✅ Init LLM
# ----------------------
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.3)

# ----------------------
# ✅ Instantiate Agents
# ----------------------
fundamentals = FundamentalsAnalyst(llm=llm)
news = NewsAnalyst(llm=llm)
sentiment = SentimentAnalyst(llm=llm)
technical = TechnicalAnalystAgent(llm=llm)

bullish = BullishResearcher(llm=llm)
bearish = BearishResearcher(llm=llm)
coordinator = DebateCoordinator(llm=llm)
trader = TraderAgent()  # ✅ NEW

# ----------------------
# ✅ LangGraph Build
# ----------------------
graph = StateGraph(state_schema=dict)

# Nodes
graph.add_node("fundamentals", fundamentals)
graph.add_node("news", news)
graph.add_node("sentiment", sentiment)
graph.add_node("technical", technical)
graph.add_node("bullish", bullish)
graph.add_node("bearish", bearish)
graph.add_node("debate", coordinator)
graph.add_node("trader", trader)  # ✅ NEW

# Flow
graph.set_entry_point("fundamentals")
graph.add_edge("fundamentals", "news")
graph.add_edge("news", "sentiment")
graph.add_edge("sentiment", "technical")
graph.add_edge("technical", "bullish")
graph.add_edge("bullish", "bearish")
graph.add_edge("bearish", "debate")
graph.add_edge("debate", "trader")  # ✅ NEW
graph.add_edge("trader", END)

# Compile
pipeline = graph.compile()

# ----------------------
# ✅ Run
# ----------------------
if __name__ == "__main__":
    state = {
        "ticker": "AAPL",
        "portfolio_value": 100000,
        "cash_balance": 20000,
        "holdings": {"AAPL": 100},
        "sector_exposure": {"tech": 0.5, "health": 0.2},
        "daily_returns": [0.01, -0.003, 0.002],
        # price: ❌ REMOVED – will now be fetched via yfinance
        "quantity": 20,
        "volatility": 0.22,
        "avg_volume": 1200000,
        "correlation_with_portfolio": 0.5,
        "upcoming_events": ["Earnings next week", "Product launch"],
        "sentiment": "positive"
    }

    print("\n🚨 Running FULL Research + Trade Pipeline...")
    final = pipeline.invoke(state)

    # -------------------
    # ✅ Show Results
    # -------------------
    debate_result = final.get("research_debate_result")
    proposal = final.get("trade_proposal")

    if debate_result:
        print("\n🧠 Debate Result:")
        for turn in debate_result.turns:
            print(f"{turn.speaker}: {turn.message}")
        print("📋 Summary:", debate_result.summary)
        print("🏆 Winner:", debate_result.winner)
        print("🔢 Tokens Used:", debate_result.total_tokens)
    else:
        print("❌ DebateCoordinator failed.")

    if proposal:
        print("\n✅ Final Trade Proposal:")
        print(proposal.model_dump_json(indent=2))
    else:
        print("❌ TraderAgent failed to generate proposal.")
