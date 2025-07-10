from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.agents.risk_management_team import BaseRiskDebator

class ConservativeDebatorAgent(BaseRiskDebator):
    def define_prompt(self) -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate.from_template(
            """
You are the Safe Risk Analyst, a bold and principled debater who champions low-risk, capital-preserving investment strategies.
You take a firm stance on protecting capital, avoiding unnecessary volatility, and respecting risk-management fundamentals.
Counter risky and overconfident positions with rational, disciplined logic.
"""
        )

        human_prompt = HumanMessagePromptTemplate.from_template(
            """
Trade Rationale: {reason_for_trade}

Ticker: {ticker} | Action: {action} | Quantity: {quantity} @ ${price}
Estimated Cost: ${estimated_cost}
Portfolio Value: ${portfolio_value} | Cash Balance: ${cash_balance}

Holdings: {holdings}
Sector Exposure: {sector_exposure}
Simulated New Position Size: {new_position_size}
New Cash Balance: {new_cash_balance}
Simulated Drawdown: {simulated_drawdown}

Market Volatility: {volatility}
Avg Volume: {avg_volume}
Correlation with Portfolio: {correlation_with_portfolio}
Upcoming Events: {upcoming_events}
Sentiment: {sentiment}

Previous Debate:
- Aggressive Analyst: {current_risky_response}
- Neutral Analyst: {current_neutral_response}
- History: {history}

Your task is to strongly critique risky strategies and argue in favor of safety, prudence, and preserving long-term capital.
Respond confidently and logically, not emotionally.
"""
        )

        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
