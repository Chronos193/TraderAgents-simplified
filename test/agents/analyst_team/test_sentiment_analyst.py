# test/agents/analyst_team/test_sentiment_analyst.py

from src.agents.analyst_team.sentiment_analyst import SentimentAnalyst
from src.schemas.analyst_schemas import SentimentAnalysisOutput
from langchain_groq import ChatGroq  # ✅ Use official Groq integration from LangChain
import os
from dotenv import load_dotenv

load_dotenv()  # Make sure GROQ_API_KEY is loaded


def test_sentiment_analyst():
    # ✅ Load your Groq API key (ensure it's set in .env or your environment)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("🛑 GROQ_API_KEY is not set in environment variables.")

    # ✅ Initialize Groq LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama3-8b-8192",  # You can change this if needed
        temperature=0.3
    )

    # ✅ Run the analyst
    analyst = SentimentAnalyst("AAPL", llm)
    result: SentimentAnalysisOutput = analyst.structured_analyze()

    # ✅ Print results
    print("📊 Sentiment Analysis:", result.model_dump())
    print("🧾 Tokens Used:", analyst.get_last_token_count())

    # ✅ Save to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sentiment_analysis_AAPL.txt", "w", encoding="utf-8") as f:
        f.write(f"📌 Ticker: {result.ticker}\n")
        f.write("\n📰 Headlines:\n" + "\n".join([f"- {h}" for h in result.headlines]))
        f.write("\n\n📊 Sentiment Summary:\n" + result.summary_text)
        f.write("\n\n🔍 Detailed Sentiment Analysis:\n" + result.sentiment_text)


if __name__ == "__main__":
    test_sentiment_analyst()
