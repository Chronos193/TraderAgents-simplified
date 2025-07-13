from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnableLambda

# === Analyst + Research ===
from src.agents.analyst_team.fundamentals_analyst import FundamentalsAnalyst
from src.agents.analyst_team.news_analyst import NewsAnalyst
from src.agents.analyst_team.sentiment_analyst import SentimentAnalyst
from src.agents.analyst_team.technical_analyst import TechnicalAnalystAgent
from src.agents.researcher_team.bullish_researcher import BullishResearcher
from src.agents.researcher_team.bearish_researcher import BearishResearcher
from src.agents.researcher_team.debate_coordinator import DebateCoordinator

# === Trader Agent ===
from src.agents.trader_agents import TraderAgent

# === Risk Management Team ===
from src.agents.risk_management_team.summarizer_risk_management import RiskSummarizerAgent
from src.agents.risk_management_team.aggressive_debater import AggressiveDebatorAgent
from src.agents.risk_management_team.conservative_debater import ConservativeDebatorAgent
from src.agents.risk_management_team.neutral_debater import NeutralDebatorAgent
from src.agents.risk_management_team.debate_coordinator_agent import DebateCoordinatorAgent

# === Load API Keys ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env")
if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY missing in .env")

# === Init LLM ===
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY, temperature=0.3)

# === Inject Trade + Risk Output ===
def inject_trade_and_risk(state: dict) -> dict:
    trade = state.get("trade_proposal")
    risk = state.get("risk_summary")

    trade_dict = trade.model_dump() if trade else {}
    risk_dict = risk.model_dump() if risk else {}

    return {**state, **trade_dict, **risk_dict}


# === Instantiate Agents ===
fundamentals = FundamentalsAnalyst(llm=llm)
news = NewsAnalyst(llm=llm)
sentiment = SentimentAnalyst(llm=llm)
technical = TechnicalAnalystAgent(llm=llm)

bullish = BullishResearcher(llm=llm)
bearish = BearishResearcher(llm=llm)
debate = DebateCoordinator(llm=llm)

trader = TraderAgent()

# ✅ New Risk Agents
risk_summarizer = RiskSummarizerAgent(llm=llm)
risk_aggressive = AggressiveDebatorAgent(llm=llm)
risk_conservative = ConservativeDebatorAgent(llm=llm)
risk_neutral = NeutralDebatorAgent(llm=llm)
risk_coordinator = DebateCoordinatorAgent()

injector = RunnableLambda(inject_trade_and_risk)

# === LangGraph Build ===
graph = StateGraph(state_schema=dict)

# === Analyst Flow ===
graph.add_node("fundamentals", fundamentals)
graph.add_node("news", news)
graph.add_node("sentiment", sentiment)
graph.add_node("technical", technical)

# === Researcher Flow ===
graph.add_node("bullish", bullish)
graph.add_node("bearish", bearish)
graph.add_node("debate", debate)

# === Trader Agent ===
graph.add_node("trader", trader)

# === Risk Management Flow ===
graph.add_node("risk_summarizer", risk_summarizer)
graph.add_node("inject_trade_and_risk", injector)
graph.add_node("risk_aggressive", risk_aggressive)
graph.add_node("risk_conservative", risk_conservative)
graph.add_node("risk_neutral", risk_neutral)
graph.add_node("risk_coordinator", risk_coordinator)

# === Flow Logic ===
graph.set_entry_point("fundamentals")
graph.add_edge("fundamentals", "news")
graph.add_edge("news", "sentiment")
graph.add_edge("sentiment", "technical")
graph.add_edge("technical", "bullish")
graph.add_edge("bullish", "bearish")
graph.add_edge("bearish", "debate")
graph.add_edge("debate", "trader")

# === Risk Flow (fixed order to resolve dependency errors) ===
graph.add_edge("trader", "risk_summarizer")
graph.add_edge("risk_summarizer", "inject_trade_and_risk")

# ➤ Run neutral first: it gives `current_neutral_response`
graph.add_edge("inject_trade_and_risk", "risk_neutral")

# ➤ Run conservative next: needs `current_neutral_response`, outputs `current_safe_response`
graph.add_edge("risk_neutral", "risk_conservative")

# ➤ Run aggressive last: needs both `current_neutral_response` and `current_safe_response`
graph.add_edge("risk_conservative", "risk_aggressive")

# ➤ Final decision
graph.add_edge("risk_aggressive", "risk_coordinator")
graph.add_edge("risk_coordinator", END)

# === Compile ===
pipeline = graph.compile()

# === Run ===
if __name__ == "__main__":
    state = {
        "ticker": "AAPL",
        "portfolio_value": 100000,
        "cash_balance": 20000,
        "holdings": {"AAPL": 100},
        "sector_exposure": {"tech": 0.5, "health": 0.2},
        "daily_returns": [0.01, -0.003, 0.002],
        "volatility": 0.22,
        "avg_volume": 1200000,
        "correlation_with_portfolio": 0.5,
        "upcoming_events": ["Earnings next week", "Product launch"],
        "sentiment": "positive",
        # 🧠 PREVENT MISSING VARIABLE ERRORS
        "current_risky_response": "No input yet.",
        "current_safe_response": "No input yet.",
        "current_neutral_response": "No input yet.",
        "history": [],
    }

    print("\n🚨 Running Full Research + Trading + Risk Management Pipeline...")
    final = pipeline.invoke(state)

    # === Output ===
    from pprint import pprint
    print("\n✅ Final State Output:\n")
    pprint(final)
