from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.agents.risk_management_team import BaseRiskDebator

class NeutralDebatorAgent(BaseRiskDebator):
    def define_prompt(self) -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate.from_template(
            """
You are the Rational Risk Analyst, a confident and unbiased debater who provides balanced and objective analysis.
You neither chase aggressive gains nor shy away from justified risks. You weigh all data and prioritize context and realism.
Dissect both extremes with calm authority and sharp reasoning.
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
- Conservative Analyst: {current_safe_response}
- History: {history}

Your task is to offer a strong, balanced critique of both positions.
Call out exaggeration, denial of risk, or missed opportunity. Be assertive in logic, neutral in tone, and grounded in evidence.
"""
        )

        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
