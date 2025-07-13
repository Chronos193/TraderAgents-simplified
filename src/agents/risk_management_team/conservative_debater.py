from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.callbacks.manager import get_openai_callback
from src.agents.risk_management_team import BaseRiskDebator


class ConservativeDebatorAgent(BaseRiskDebator):
    def define_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
You are the Safe Risk Analyst: a cautious expert focused on capital preservation and disciplined investing.

Your goals:
- Analyze trades for downside risk, red flags, and volatility.
- Recommend safer alternatives if necessary.
- Avoid unjustified exposure.

**Output Rules**
- Use only bullet points (•, +, or -).
- Avoid intros, summaries, and conclusions.
- Limit to ~100 words.
- No data restating — focus on direct risk insights.
"""
            ),
            HumanMessagePromptTemplate.from_template(
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
Call out major risk concerns in bullet points.
Be direct. Recommend safer alternatives where appropriate.
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
            print(f"[ERROR] ConservativeDebatorAgent failed: {e}")
            return "⚠️ Conservative response generation failed."

        return result.strip()

    def __call__(self, state: dict) -> dict:
        if not isinstance(state, dict):
            print("[WARN] ConservativeDebatorAgent received invalid state.")
            return state

        response = self.run_analysis(state)
        return {**state, "conservative_debate_response": response}

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(self.__call__)
