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
Ticker: {ticker}
Quantity: {quantity} @ ${price}
Portfolio: ${portfolio_value} | Cash: ${cash_balance}
New Position Size: {new_position_size} | New Cash: {new_cash_balance}
Simulated Drawdown: {simulated_drawdown}

Market:
- Volatility: {volatility} | Volume: {avg_volume} | Sentiment: {sentiment}
- Sector Exposure: {sector_exposure}
- Holdings: {holdings}

Risk Summary:
- Key Risks: {key_risks}
- Risk Opportunities: {risk_opportunities}
- Volatility Indicators: {volatility_indicators}
- Financial Flags: {financial_flags}
- News Themes: {negative_news_themes}
- Overall Risk: {overall_risk_assessment}

Debate Context:
- Aggressive: {current_risky_response}
- Neutral: {current_neutral_response}
- History: {history}

Your Task:
Call out major risk concerns in bullet points.
Be direct. Recommend safer alternatives if necessary.
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

        # Append to history (used by NeutralDebator)
        history = state.get("history", [])
        history.append("Conservative: " + response)

        return {
            **state,
            "conservative_debate_response": response,
            "current_safe_response": response,    # ✅ Used by Aggressive & Neutral
            "history": history                    # ✅ Required downstream
        }


    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(self.__call__)
