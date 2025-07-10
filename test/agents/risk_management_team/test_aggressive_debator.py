from src.agents.risk_management_team import AggressiveDebatorAgent
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

agent = AggressiveDebatorAgent()

response = agent.run({
    "trader_decision": "Buy 100 shares of NVDA due to projected dominance in AI GPU market.",
    "market_research_report": (
        "NVIDIA holds 80% of the global AI chip market. "
        "Market analysts expect 30% YoY growth in demand for AI accelerators over the next 5 years."
    ),
    "sentiment_report": (
        "Social sentiment is overwhelmingly positive on Reddit, Twitter, and Seeking Alpha. "
        "Many retail traders expect NVDA to outperform earnings expectations."
    ),
    "news_report": (
        "NVIDIA announced a multi-billion dollar partnership with Microsoft and Meta "
        "to supply next-gen GPUs for their AI data centers."
    ),
    "fundamentals_report": (
        "NVIDIA reported record revenue growth in Q2, with 45% YoY increase. "
        "Profit margins remain high, and cash reserves are strong."
    ),
    "history": "",
    "current_safe_response": (
        "This trade may be overexposed to AI hype. Valuation is already high, "
        "and any earnings miss could result in a sharp correction."
    ),
    "current_neutral_response": (
        "NVDA is fundamentally strong, but entering now carries short-term volatility risk. "
        "Waiting for a pullback might be safer."
    )
})

print("💥 Aggressive Debator says:\n")
print(response)

# 🔍 Token and Cost Usage
print(f"\n📊 Tokens Used: {agent.get_total_tokens_used()}")
print(f"💸 Estimated Cost: ${agent.get_total_cost():.6f}")
