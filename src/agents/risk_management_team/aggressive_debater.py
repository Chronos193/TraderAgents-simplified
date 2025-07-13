from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.callbacks.manager import get_openai_callback
from src.agents.risk_management_team import BaseRiskDebator


class AggressiveDebatorAgent(BaseRiskDebator):
    def define_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
You are the Risky Risk Analyst: a bold, high-conviction strategist who argues in favor of high-risk, high-reward opportunities.

Your job:
- Advocate strongly for the proposed trade.
- Emphasize upside, growth signals, and catalysts.
- Reframe risks as necessary steps for bold returns.

**Output Rules**
- Use only bullet points (•, +, or -).
- No intros or conclusions.
- Be punchy and clear (max ~100 words).
- Omit filler — go straight to compelling logic.
"""
            ),
            HumanMessagePromptTemplate.from_template(
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
Make a bold but concise bullet-point case for why this is a smart, aggressive move.
Emphasize growth, catalysts, and why the risk is worth it.
"""
            )
        ])

    def run_analysis(self, state: dict) -> str:
        prompt = self.define_prompt()
        chain = prompt | self.llm | StrOutputParser()

        try:
            with get_openai_callback() as cb:
                result = chain.invoke(state)
                self._last_token_count = cb.total_tokens
        except Exception as e:
            print(f"[ERROR] AggressiveDebatorAgent failed: {e}")
            return "⚠️ Aggressive response generation failed."

        return result.strip()

    def __call__(self, state: dict) -> dict:
        if not isinstance(state, dict):
            print("[WARN] AggressiveDebatorAgent received invalid state.")
            return state

        response = self.run_analysis(state)
        return {**state, "aggressive_debate_response": response}

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(self.__call__)
