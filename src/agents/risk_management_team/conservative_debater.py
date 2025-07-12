from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.agents.risk_management_team import BaseRiskDebator

class ConservativeDebatorAgent(BaseRiskDebator):
    def define_prompt(self) -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate.from_template(
            """
You are the Safe Risk Analyst: a cautious expert who advocates capital preservation and prudent risk-taking.

Your job:
- Analyze trades through a lens of risk, volatility, and long-term safety.
- Prioritize discipline, restraint, and downside protection.
- Identify red flags and call out unjustified risk exposure.

**Output Format Rules**
- Use bullet points only (•, +, -). No paragraphs.
- Be concise and direct (aim for max 100 words).
- Do not summarize input or restate data.
- Skip intro/outro fluff — be clinical and precise.
"""
        )

        human_prompt = HumanMessagePromptTemplate.from_template(
            """
Trade Summary:
- Ticker: {ticker} | Action: {action} | Qty: {quantity} @ ${price}
- Portfolio Value: ${portfolio_value} | Cash: ${cash_balance}
- Simulated Drawdown: {simulated_drawdown}
- Correlation: {correlation_with_portfolio}

Key Inputs:
- Rationale: {reason_for_trade}
- Market Volatility: {volatility}
- Avg Volume: {avg_volume}
- Sentiment: {sentiment}
- Sector Exposure: {sector_exposure}
- Holdings: {holdings}
- New Position Size: {new_position_size}
- New Cash: {new_cash_balance}

Risk Summary:
- Key Risks: {key_risks}
- Risk Opportunities: {risk_opportunities}
- Indicators: {volatility_indicators}
- Financial Flags: {financial_flags}
- News Themes: {negative_news_themes}
- Overall Risk: {overall_risk_assessment}

Previous Debate:
- Aggressive: {current_risky_response}
- Neutral: {current_neutral_response}
- History: {history}

Your Task:
- Call out major risk factors clearly and directly.
- Recommend more conservative action if needed.
- Prioritize capital preservation.
"""
        )

        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
