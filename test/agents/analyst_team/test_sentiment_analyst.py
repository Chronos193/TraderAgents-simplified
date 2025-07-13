from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.agents.analyst_team.sentiment_analyst import SentimentAnalyst
import json
import os
from dotenv import load_dotenv
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Step 1: Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

# Step 2: Instantiate SentimentAnalyst agent
sentiment_agent = SentimentAnalyst(llm=llm)

# Step 3: Build minimal LangGraph with just sentiment node
workflow = StateGraph(state_schema=dict)
workflow.add_node("sentiment", sentiment_agent)
workflow.set_entry_point("sentiment")
workflow.add_edge("sentiment", END)
graph = workflow.compile()

# Step 4: Provide test input and run
if __name__ == "__main__":
    test_state = {
        "ticker": "AAPL"
    }

    print("🚀 Running SentimentAnalyst test...\n")
    result = graph.invoke(test_state)

    print("\n✅ Final SentimentAnalysis Output:")
    print(json.dumps(result["sentiment_analysis"].dict(), indent=2))
