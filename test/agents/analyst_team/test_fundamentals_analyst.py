from langgraph.graph import StateGraph
from src.agents.analyst_team.fundamentals_analyst import FundamentalsAnalyst
from langchain_groq import ChatGroq  # Or any other LLM you use
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Use your real or test LLM here
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
fundamentals_node = FundamentalsAnalyst(llm)

# Create a mini LangGraph to test it
workflow = StateGraph(state_schema=dict)
workflow.add_node("fundamentals", fundamentals_node)
workflow.set_entry_point("fundamentals")
workflow.set_finish_point("fundamentals")
graph = workflow.compile()

# Provide test input
mock_state = {
    "ticker": "AAPL"
}

if __name__ == "__main__":
    print("🔍 Running FundamentalsAnalyst node test...")
    result = graph.invoke(mock_state)

    print("\n✅ Test Completed. Output:")
    from pprint import pprint
    pprint(result["fundamentals_analysis"].model_dump(), sort_dicts=False)
