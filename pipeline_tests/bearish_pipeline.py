from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from src.agents.analyst_team.fundamentals_analyst import FundamentalsAnalyst
from src.agents.analyst_team.news_analyst import NewsAnalyst
from src.agents.analyst_team.sentiment_analyst import SentimentAnalyst
from src.agents.analyst_team.technical_analyst import TechnicalAnalystAgent
from src.agents.researcher_team.bearish_researcher import BearishResearcher  # 👈 This one is different
import os
from dotenv import load_dotenv
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Step 1: Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.3)

# Step 2: Instantiate agents
fundamentals = FundamentalsAnalyst(llm=llm)
news = NewsAnalyst(llm=llm)
sentiment = SentimentAnalyst(llm=llm)
technical = TechnicalAnalystAgent(llm=llm)
bearish = BearishResearcher(llm=llm)

# Step 3: Build LangGraph
graph = StateGraph(state_schema=dict)

# Add nodes
graph.add_node("fundamentals", fundamentals)
graph.add_node("news", news)
graph.add_node("sentiment", sentiment)
graph.add_node("technical", technical)
graph.add_node("bearish", bearish)

# Define edges
graph.set_entry_point("fundamentals")
graph.add_edge("fundamentals", "news")
graph.add_edge("news", "sentiment")
graph.add_edge("sentiment", "technical")
graph.add_edge("technical", "bearish")
graph.add_edge("bearish", END)

# Compile
workflow = graph.compile()

# Step 4: Run with real input
if __name__ == "__main__":
    state = {"ticker": "AAPL"}  # You can replace with any valid ticker like "NVDA"
    print("🚨 Running BearishResearcher End-to-End Pipeline...\n")
    final_result = workflow.invoke(state)

    if thesis := final_result.get("bearish_thesis"):
        print("❗ Final Bearish Thesis:")
        print(f"\n🧠 Thesis: {thesis.thesis}")
        print("\n📌 Supporting Points:")
        for point in thesis.supporting_points:
            print(f" - {point}")
        print(f"\n📉 Confidence: {thesis.confidence}")
    else:
        print("❌ Bearish thesis generation failed.")
