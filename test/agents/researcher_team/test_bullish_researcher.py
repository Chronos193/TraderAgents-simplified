from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.agents.researcher_team.bullish_researcher import BullishResearcher
from src.schemas.analyst_schemas import (
    FundamentalsAnalysisOutput,
    NewsAnalysisOutput,
    SentimentAnalysisOutput,
    TechnicalAnalysisOutput
)
import json
import os
from dotenv import load_dotenv
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Step 1: Mock outputs from Analyst agents
mock_state = {
    "ticker": "AAPL",
    "fundamentals_analysis": FundamentalsAnalysisOutput(
        ticker="AAPL",
        financials_summary={"Revenue": 300e9},
        balance_summary={"Assets": 350e9},
        cashflow_summary={"Operating": 100e9},
        analysis_text="Strong revenue and assets, with a solid cashflow base and market dominance."
    ),
    "news_analysis": NewsAnalysisOutput(
        ticker="AAPL",
        headlines=["Apple unveils new iPhone", "Apple tops Wall Street estimates"],
        summary_text="Apple continues to impress with new product launches and earnings beats.",
        themes_text="Product innovation, strong demand, global expansion"
    ),
    "sentiment_analysis": SentimentAnalysisOutput(
        ticker="AAPL",
        headlines=["Investors optimistic", "Positive market reaction"],
        summary_text="Investors are generally positive.",
        sentiment_text="Positive sentiment supported by strong headlines and market confidence."
    ),
    "technical_analysis": TechnicalAnalysisOutput(
        ticker="AAPL",
        macd=1.5,
        signal=1.2,
        histogram=0.3,
        rsi=65.4,
        recommendation="Buy – MACD and RSI indicate upward momentum."
    ),
}

# Step 2: Initialize LLM and researcher
llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)
bull = BullishResearcher(llm=llm)

# Step 3: Create LangGraph to run just the BullishResearcher
graph = StateGraph(state_schema=dict)
graph.add_node("bullish", bull)
graph.set_entry_point("bullish")
graph.add_edge("bullish", END)
workflow = graph.compile()

# Step 4: Run and display output
if __name__ == "__main__":
    print("🚀 Running BullishResearcher Test...\n")
    result = workflow.invoke(mock_state)

    thesis = result.get("bullish_thesis")
    if thesis:
        print("✅ Bullish Thesis Output:")
        print(json.dumps(thesis.dict(), indent=2))
    else:
        print("❌ No thesis was returned.")
