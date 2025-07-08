from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import Optional
from src.agents.analyst_team import NewsAnalyst
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
finnhub_api_key = os.getenv('FINNHUB_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model="llama3-8b-8192", temperature=0.1)
analyst = NewsAnalyst("AAPL", llm)

class GraphState(BaseModel):
    news: Optional[dict] = None

def wrap_news_output(state: GraphState) -> GraphState:
    output = analyst.structured_analyze()
    return GraphState(news=output.model_dump())

graph = StateGraph(GraphState)
graph.add_node("news_analysis", wrap_news_output)
graph.set_entry_point("news_analysis")
graph.add_edge("news_analysis", END)
app = graph.compile()

result = app.invoke(GraphState())
print("📊 News Analysis:", result['news'])
print("🧾 Tokens Used:", analyst.get_last_token_count())
