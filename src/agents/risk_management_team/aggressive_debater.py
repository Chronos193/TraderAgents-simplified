from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.agents.risk_management_team import BaseRiskDebator

class AggressiveDebatorAgent(BaseRiskDebator):
    def define_prompt(self) -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate.from_template(
            """
You are the Risky Risk Analyst, a bold debater who champions high-risk, high-reward opportunities.
Your job is to advocate for aggressive investment strategies and counter the cautious arguments from other analysts.
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
- Conservative Analyst: {current_safe_response}
- Neutral Analyst: {current_neutral_response}
- History: {history}

Your task is to debate why this high-reward trade is the optimal move.
Emphasize upside potential, innovation, and bold returns. Address concerns with data-driven confidence.
Avoid formatting and write naturally as if debating peers.
"""
        )

        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
