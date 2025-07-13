from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from src.agents.analyst_team.technical_analyst import TechnicalAnalystAgent
import json
import os
from dotenv import load_dotenv
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Step 1: Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

# Step 2: Instantiate the agent
technical = TechnicalAnalystAgent(llm=llm)

# Step 3: Build mini LangGraph
workflow = StateGraph(state_schema=dict)
workflow.add_node("technical", technical)
workflow.set_entry_point("technical")
workflow.add_edge("technical", END)
graph = workflow.compile()

# Step 4: Run test
if __name__ == "__main__":
    test_state = {
        "ticker": "AAPL"
    }

    print("🚀 Running TechnicalAnalystAgent test...\n")
    result = graph.invoke(test_state)

    print("\n✅ Final TechnicalAnalysis Output:")
    print(json.dumps(result["technical_analysis"].dict(), indent=2))
