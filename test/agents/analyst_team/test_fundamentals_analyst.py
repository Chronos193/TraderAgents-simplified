from src.agents.analyst_team.fundamentals_analyst import FundamentalsAnalyst
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: LangGraph State
class GraphState(BaseModel):
    fundamentals: Optional[dict] = None

# Step 2: LangGraph Node
def wrap_fundamentals_output(state: GraphState) -> GraphState:
    output = analyst.structured_analyze()
    return GraphState(fundamentals=output.model_dump())

# Step 3: LangGraph App
llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
analyst = FundamentalsAnalyst("AAPL", llm)

graph = StateGraph(GraphState)
graph.add_node("fundamentals_analysis", wrap_fundamentals_output)
graph.set_entry_point("fundamentals_analysis")
graph.add_edge("fundamentals_analysis", END)

app = graph.compile()

# Step 4: Run and print
result = app.invoke(GraphState())
fundamentals = result["fundamentals"]  # ✅ FIXED
print("📊 Fundamentals Analysis:", fundamentals)
print("🧾 Tokens Used:", analyst.get_last_token_count())
