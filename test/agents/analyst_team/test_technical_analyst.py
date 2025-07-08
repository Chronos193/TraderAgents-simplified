# test/agents/analyst_team/test_technical_analyst.py

from src.agents.analyst_team.technical_analyst import TechnicalAnalystAgent
from src.schemas.analyst_schemas import TechnicalAnalysisOutput
from langchain_groq import ChatGroq  # Replace with your actual import if different
import os
from dotenv import load_dotenv

load_dotenv()  # Make sure GROQ_API_KEY is loaded
def test_technical_analyst():
    # 🔧 Instantiate LLM (Groq LLaMA-3 or compatible)
    llm = ChatGroq(model="llama3-8b-8192")  # Or "mixtral-8x7b-32768" if you prefer

    # 📈 Create the agent
    agent = TechnicalAnalystAgent("AAPL", llm)

    # 🧪 Run the structured analysis
    result: TechnicalAnalysisOutput = agent.structured_analyze()

    # ✅ Print result
    print("📊 Technical Analysis:", result.model_dump())
    print("🧾 Tokens Used:", agent.get_last_token_count())

    # 💾 Optionally write to file
    with open("outputs/technical_analysis_AAPL.txt", "w", encoding="utf-8") as f:
        f.write(f"📌 Ticker: {result.ticker}\n")
        f.write(f"🔹 MACD: {result.macd}\n")
        f.write(f"🔹 Signal: {result.signal}\n")
        f.write(f"🔹 Histogram: {result.histogram}\n")
        f.write(f"🔹 RSI: {result.rsi}\n")
        f.write(f"\n📈 Recommendation:\n{result.recommendation}")

if __name__ == "__main__":
    test_technical_analyst()
