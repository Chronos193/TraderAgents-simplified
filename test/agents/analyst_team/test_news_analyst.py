from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from src.agents.analyst_team.news_analyst import NewsAnalyst
from src.schemas.analyst_schemas import NewsAnalysisOutput
from langgraph.graph import END
import json
import os
from dotenv import load_dotenv
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ---- Initialize LLM ----
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)

# ---- Instantiate NewsAnalyst agent ----
news_agent = NewsAnalyst(llm=llm)

# ---- Build a minimal test graph ----
workflow = StateGraph(state_schema=dict)
workflow.add_node("news_analysis", news_agent)
workflow.set_entry_point("news_analysis")
workflow.add_edge("news_analysis", END)
graph = workflow.compile()

# ---- Provide test input ----
if __name__ == "__main__":
    test_state = {
        "ticker": "AAPL"
    }

    print("🚀 Running NewsAnalyst test workflow...\n")
    result = graph.invoke(test_state)

    print("\n✅ Final output:")
    print(json.dumps(result["news_analysis"].dict(), indent=2))
