# test_trader_agent.py
from src.agents.trader_agents import TraderAgent
from src.schemas.researcher_schemas import DebateResult  # replace with actual mock if needed

debate = DebateResult(
    turns=[],  # use mock or load from a file
    summary="Debate on NVDA shows mixed but positive tone",
    winner="Bullish",
    total_tokens=4200
)

agent = TraderAgent()
agent.run(debate)
