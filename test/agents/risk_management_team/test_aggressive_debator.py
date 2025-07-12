from src.agents.risk_management_team import AggressiveDebatorAgent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY=os.getenv('GROQ_API_KEY')

# Step 1: Create a mock LLM response (to simulate LLM behavior in tests)
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# Step 2: Instantiate the agent
agent = AggressiveDebatorAgent()

# Step 3: Sample input to match placeholders in the human prompt
sample_input = {
    "reason_for_trade": "Strong AI chip demand, NVDA breaking out technically",
    "ticker": "NVDA",
    "action": "BUY",
    "quantity": 100,
    "price": 115,
    "estimated_cost": 11500,
    "portfolio_value": 60000,
    "cash_balance": 20000,
    "holdings": "AAPL, MSFT, TSLA",
    "sector_exposure": "Tech: 60%, Healthcare: 20%, Energy: 20%",
    "new_position_size": 0.17,
    "new_cash_balance": 8500,
    "simulated_drawdown": 0.08,
    "volatility": 0.27,
    "avg_volume": "25M",
    "correlation_with_portfolio": 0.6,
    "upcoming_events": "Q2 Earnings Call",
    "sentiment": "Positive institutional flow",
    "key_risks": "['Export restrictions', 'China tensions']",
    "risk_opportunities": "['Data center growth', 'AI chip leadership']",
    "volatility_indicators": "{'rsi': 71.2, 'macd': 2.3}",
    "financial_flags": "{'debt_levels': 0.25, 'cash_reserves': 12000000000.0}",
    "negative_news_themes": "['US-China chip regulation tension']",
    "overall_risk_assessment": "Moderate risk, with strong upside potential",
    "current_safe_response": "This is too risky. Suggest holding cash and waiting post-earnings.",
    "current_neutral_response": "Position size should be trimmed to 50 shares. Wait for clarity.",
    "history": "Last aggressive stance yielded a strong win on AMD. Risk tolerance increased slightly."
}

# Step 4: Run the agent and print response
response = agent.run(sample_input)
print("📣 Aggressive Debator Output:\n", response.strip())
