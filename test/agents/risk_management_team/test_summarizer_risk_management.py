from langchain_groq import ChatGroq
from src.agents.risk_management_team.summarizer_risk_management import RiskSummarizerAgent
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

# Initialize the LLM
llm = ChatGroq(model="llama3-8b-8192")  # or any other ChatModel supported

# Instantiate the summarizer agent
agent = RiskSummarizerAgent(llm=llm)

# Sample test input (minimal and clean)
sample_input = {
    "ticker": "TSLA",

    "fundamentals": (
        "Tesla reported 12% YoY revenue growth and stable 9.6% operating margins. "
        "Free cash flow dropped 15% due to capex in gigafactories. "
        "Debt is low, cash reserves are $12B, inventory days rose slightly."
    ),

    "technical": (
        '{"macd": 1.34, "signal": 1.10, "histogram": 0.24, '
        '"rsi": 74.8, "recommendation": "Sell - Overbought on RSI, MACD flattening"}'
    ),

    "news": (
        '"Tesla Recalls 300K Vehicles Due to Autopilot Issues"\n'
        '"New Gigafactory in Mexico Expected to Boost Output"\n'
        '"EU Regulators Scrutinize Tesla\'s Data Privacy Practices"\n\n'
        "Themes: Regulatory risk on autonomy/data, expansion despite headwinds"
    ),

    "sentiment": (
        '"Tesla criticized over autopilot failures, fans remain loyal."\n'
        '"Investor outlook mixed; institutions cautious, retail optimistic."\n\n'
        "Summary: Overall sentiment slightly negative due to recalls and regulation."
    ),
}

# Run the agent and print the structured output
output = agent.run(sample_input)

pprint(output.model_dump())
