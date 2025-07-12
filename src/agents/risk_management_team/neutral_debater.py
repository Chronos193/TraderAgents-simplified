from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.agents.risk_management_team import BaseRiskDebator

class NeutralDebatorAgent(BaseRiskDebator):
    def define_prompt(self) -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate.from_template(
            """
You are the Rational Risk Analyst: a calm, balanced thinker who critiques both aggressive and conservative positions fairly.

Instructions:
- Use bullet points only (•, +, -)
- Be concise and direct (aim for max 100 words).
- Do NOT repeat trade inputs
- Point out exaggeration, blind spots, flawed logic
- Focus on nuance and proportionality
"""
        )

        human_prompt = HumanMessagePromptTemplate.from_template(
            """
Trade Summary:
- Rationale: {reason_for_trade}
- Action: {action} | Ticker: {ticker} | Qty: {quantity} @ ${price}
- Portfolio: ${portfolio_value} | Cash: ${cash_balance}
- Simulated Drawdown: {simulated_drawdown} | Corr.: {correlation_with_portfolio}

Risk Snapshot:
- Risks: {key_risks}
- Opportunities: {risk_opportunities}
- Indicators: {volatility_indicators}
- Financial Flags: {financial_flags}
- News Themes: {negative_news_themes}
- Assessment: {overall_risk_assessment}

Debate So Far:
- Aggressive: {current_risky_response}
- Conservative: {current_safe_response}
- History: {history}

Your Task:
- Fairly critique both sides
- Identify missed context or exaggerations
- Give a balanced judgment: is the trade justified or not?
"""
        )

        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
