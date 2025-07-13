from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.callbacks.manager import get_openai_callback
from src.agents.risk_management_team import BaseRiskDebator


class NeutralDebatorAgent(BaseRiskDebator):
    def define_prompt(self) -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate.from_template(
            """
You are the Rational Risk Analyst: a calm, unbiased observer evaluating both aggressive and conservative positions.

Your goals:
- Highlight flawed reasoning, exaggerations, or missing context.
- Acknowledge strong arguments on both sides.
- Give a final balanced judgment on the trade.

**Output Rules**
- Use only bullet points (•, +, or -).
- No paragraphs, intros, or summaries.
- Stay under ~100 words.
- Do not restate trade inputs or repeat debate content verbatim.
"""
        )

        human_prompt = HumanMessagePromptTemplate.from_template(
            """
Trade Summary:
- Ticker: {ticker} | Action: {action} | Qty: {quantity} @ ${price}
- Rationale: {reason_for_trade}
- Portfolio: ${portfolio_value} | Cash: ${cash_balance}
- Drawdown: {simulated_drawdown} | Correlation: {correlation_with_portfolio}

Risk Snapshot:
- Key Risks: {key_risks}
- Risk Opportunities: {risk_opportunities}
- Indicators: {volatility_indicators}
- Flags: {financial_flags}
- News Themes: {negative_news_themes}
- Assessment: {overall_risk_assessment}

Debate So Far:
- Aggressive: {current_risky_response}
- Conservative: {current_safe_response}
- History: {history}

Your Task:
- Critique both positions clearly and fairly.
- Identify weak arguments, good logic, and overlooked risks.
- State whether the trade is justified or not.
"""
        )

        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    def run_analysis(self, state: dict) -> str:
        prompt = self.define_prompt()
        chain = prompt | self.llm | StrOutputParser()

        try:
            with get_openai_callback() as cb:
                result = chain.invoke(state)
                self._last_token_count = cb.total_tokens
        except Exception as e:
            print(f"[ERROR] NeutralDebatorAgent failed: {e}")
            return "⚠️ Neutral response generation failed."

        return result.strip()

    def __call__(self, state: dict) -> dict:
        if not isinstance(state, dict):
            print("[WARN] NeutralDebatorAgent received invalid state.")
            return state

        response = self.run_analysis(state)
        return {**state, "neutral_debate_response": response}

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(self.__call__)
