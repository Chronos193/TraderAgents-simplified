from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.agents.researcher_team.bearish_researcher import BearishResearcher
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
# Step 1: Mock state simulating output from analysts
mock_state = {
    "ticker": "AAPL",
    "fundamentals_analysis": FundamentalsAnalysisOutput(
        ticker="AAPL",
        financials_summary={"Revenue": 300e9},
        balance_summary={"Assets": 350e9},
        cashflow_summary={"Operating": 100e9},
        analysis_text="Apple is heavily reliant on a few product lines and has growing R&D costs."
    ),
    "news_analysis": NewsAnalysisOutput(
        ticker="AAPL",
        headlines=["Apple faces antitrust lawsuit", "iPhone demand slows in China"],
        summary_text="Regulatory challenges and weak international demand raise concerns.",
        themes_text="Lawsuits, demand decline, regulatory pressure"
    ),
    "sentiment_analysis": SentimentAnalysisOutput(
        ticker="AAPL",
        headlines=["Investor fears on Apple's growth", "Negative press surrounding Apple"],
        summary_text="Sentiment leans negative amid market uncertainty.",
        sentiment_text="Negative sentiment due to regulatory risk and growth slowdown."
    ),
    "technical_analysis": TechnicalAnalysisOutput(
        ticker="AAPL",
        macd=-0.7,
        signal=-0.4,
        histogram=-0.3,
        rsi=38.1,
        recommendation="Sell – Bearish MACD crossover and RSI indicates weakness."
    ),
}

# Step 2: Initialize LLM and BearishResearcher
llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)
bear = BearishResearcher(llm=llm)

# Step 3: Create LangGraph
graph = StateGraph(state_schema=dict)
graph.add_node("bearish", bear)
graph.set_entry_point("bearish")
graph.add_edge("bearish", END)
workflow = graph.compile()

# Step 4: Run and display output
if __name__ == "__main__":
    print("🐻 Running BearishResearcher Test...\n")
    result = workflow.invoke(mock_state)

    thesis = result.get("bearish_thesis")
    if thesis:
        print("✅ Bearish Thesis Output:")
        print(json.dumps(thesis.dict(), indent=2))
    else:
        print("❌ No thesis was returned.")
