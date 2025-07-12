from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.agents.risk_management_team import BaseRiskDebator

class AggressiveDebatorAgent(BaseRiskDebator):
    def define_prompt(self) -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate.from_template(
            """
You are the Risky Risk Analyst: a bold, high-conviction strategist who argues in favor of high-risk, high-reward opportunities.

Your job is to:
- Advocate strongly for the proposed trade.
- Emphasize upside, growth signals, and catalysts.
- Reframe risks as necessary steps for bold returns.

**Output Format Rules**
- Only use bullet points (`•`, `+`, or `-`) — no paragraphs.
- Be concise and direct (aim for max 100 words).
- Skip intros and conclusions.
- Focus only on strong, convincing points.
- Omit unnecessary context — the model already sees the inputs.
"""
        )

        human_prompt = HumanMessagePromptTemplate.from_template(
            """
Trade Summary:
- Ticker: {ticker} | Action: {action} | Qty: {quantity} @ ${price}
- Portfolio Value: ${portfolio_value} | Cash: ${cash_balance}
- Simulated Drawdown: {simulated_drawdown}
- Correlation w/ Portfolio: {correlation_with_portfolio}

Key Factors:
- Rationale: {reason_for_trade}
- Market Volatility: {volatility} | Avg Volume: {avg_volume}
- Upcoming Events: {upcoming_events} | Sentiment: {sentiment}
- Sector Exposure: {sector_exposure}
- Holdings: {holdings}
- New Position Size: {new_position_size} | New Cash: {new_cash_balance}

Risk Summary:
- Key Risks: {key_risks}
- Opportunities: {risk_opportunities}
- Indicators: {volatility_indicators}
- Flags: {financial_flags}
- News Themes: {negative_news_themes}
- Overall Risk: {overall_risk_assessment}

Previous Debate:
- Conservative: {current_safe_response}
- Neutral: {current_neutral_response}
- History: {history}

Your Task:
Make a concise bullet-point case for why this trade is a smart, aggressive move.
Emphasize growth, catalysts, and why risks are worth taking.
"""
        )

        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
