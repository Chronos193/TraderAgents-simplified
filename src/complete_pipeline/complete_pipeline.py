import os
import json
import ast
from dotenv import load_dotenv
from typing import Dict, List

from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from src.schemas.final_pipeline_schema.state_schema import TradingState

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

# --- Agent Imports ---
from src.agents.analyst_team.fundamentals_analyst import FundamentalsAnalyst
from src.agents.analyst_team.technical_analyst import TechnicalAnalystAgent
from src.agents.analyst_team.news_analyst import NewsAnalyst
from src.agents.analyst_team.sentiment_analyst import SentimentAnalyst
from src.agents.researcher_team.bullish_researcher import BullishResearcher
from src.agents.researcher_team.bearish_researcher import BearishResearcher
from src.agents.researcher_team.debate_coordinator import DebateCoordinator as ResearchDebateCoordinator
from src.agents.trader_agents.trader_agent import TraderAgent
from src.agents.risk_management_team.summarizer_risk_management import RiskSummarizerAgent
from src.agents.risk_management_team.aggressive_debater import AggressiveDebatorAgent
from src.agents.risk_management_team.conservative_debater import ConservativeDebatorAgent
from src.agents.risk_management_team.neutral_debater import NeutralDebatorAgent
from src.agents.risk_management_team.debate_coordinator_agent import DebateCoordinatorAgent as RiskDebateCoordinator

# --- Instantiate Agents ---
fundamentals = FundamentalsAnalyst(llm)
technical = TechnicalAnalystAgent(llm)
news = NewsAnalyst(llm)
sentiment = SentimentAnalyst(llm)
bullish = BullishResearcher(llm)
bearish = BearishResearcher(llm)
research_debate = ResearchDebateCoordinator(llm)
trader = TraderAgent()
risk_summarizer = RiskSummarizerAgent(llm)
aggressive = AggressiveDebatorAgent()
conservative = ConservativeDebatorAgent()
neutral = NeutralDebatorAgent()
risk_debate = RiskDebateCoordinator()

# --- Build LangGraph ---
workflow = StateGraph(state_schema=TradingState)

# Add nodes
workflow.add_node("fundamentals", fundamentals)
workflow.add_node("technical", technical)
workflow.add_node("news", news)
workflow.add_node("sentiment", sentiment)
workflow.add_node("bullish", bullish)
workflow.add_node("bearish", bearish)
workflow.add_node("research_debate", research_debate)
workflow.add_node("trader", trader)
workflow.add_node("risk_summarizer", risk_summarizer)
workflow.add_node("aggressive", aggressive)
workflow.add_node("conservative", conservative)
workflow.add_node("neutral", neutral)

# ✅ Manual join node using RunnableLambda
risk_join = RunnableLambda(lambda state: state)
workflow.add_node("risk_join", risk_join)

workflow.add_node("risk_debate", risk_debate)

# Define execution order
workflow.set_entry_point("fundamentals")
workflow.add_edge("fundamentals", "technical")
workflow.add_edge("technical", "news")
workflow.add_edge("news", "sentiment")
workflow.add_edge("sentiment", "bullish")
workflow.add_edge("bullish", "bearish")
workflow.add_edge("bearish", "research_debate")
workflow.add_edge("research_debate", "trader")
workflow.add_edge("trader", "risk_summarizer")

# Risk debaters
workflow.add_edge("risk_summarizer", "aggressive")
workflow.add_edge("risk_summarizer", "conservative")
workflow.add_edge("risk_summarizer", "neutral")

# Manual join logic
workflow.add_edge("aggressive", "risk_join")
workflow.add_edge("conservative", "risk_join")
workflow.add_edge("neutral", "risk_join")
workflow.add_edge("risk_join", "risk_debate")

workflow.add_edge("risk_debate", END)

# Compile graph
graph = workflow.compile()

# --- 🔁 User Input Helper ---
def get_user_portfolio_input() -> Dict:
    def safe_float(prompt: str) -> float:
        while True:
            try:
                return float(input(prompt).strip())
            except ValueError:
                print("❌ Please enter a valid number.")

    def safe_literal(prompt: str, expected_type):
        while True:
            try:
                val = input(prompt).strip()
                parsed = ast.literal_eval(val)
                if isinstance(parsed, expected_type):
                    return parsed
                raise ValueError()
            except Exception:
                print(f"❌ Please enter a valid {expected_type.__name__}.")

    print("\n📊 --- Provide Required Portfolio Info ---")
    return {
        "ticker": input("Ticker: ").strip().upper(),
        "portfolio_value": safe_float("Total portfolio value ($): "),
        "cash_balance": safe_float("Available cash ($): "),
        "holdings": safe_literal("Current holdings (e.g., {'AAPL': 5000}): ", dict),
        "sector_exposure": safe_literal("Sector exposure (e.g., {'tech': 0.5}): ", dict),
        "daily_returns": safe_literal("Recent daily returns (e.g., [0.01, -0.002]): ", list),
    }

# --- 🚀 Run Pipeline ---
if __name__ == "__main__":
    user_input_state: TradingState = get_user_portfolio_input()

    print("\n🚀 Starting Trading Agent Workflow...\n")
    result = graph.invoke(user_input_state)

    print("\n✅ Final pipeline result:")
    print(json.dumps(result, indent=2, default=str))
