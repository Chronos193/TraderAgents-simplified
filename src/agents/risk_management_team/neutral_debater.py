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
🧾 Trade Context:
- Ticker: {ticker} | Quantity: {quantity} @ ${price}
- Portfolio Value: ${portfolio_value} | Cash: ${cash_balance}
- New Position Size: {new_position_size} | New Cash: {new_cash_balance}
- Simulated Drawdown: {simulated_drawdown}

📉 Risk Snapshot:
- Volatility: {volatility} | Sentiment: {sentiment}
- Risk Indicators: {volatility_indicators}
- Financial Flags: {financial_flags}
- Key Risks: {key_risks}
- Opportunities: {risk_opportunities}
- News Themes: {negative_news_themes}
- Overall Assessment: {overall_risk_assessment}

🧠 Prior Arguments:
- Aggressive: {current_risky_response}
- Conservative: {current_safe_response}
- History: {history}

🎯 Your Task:
- Fairly critique both positions.
- Identify weak logic or exaggerations.
- Suggest if the trade is reasonable or if caution is warranted.
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

        # Append to history (used by DebateCoordinatorAgent)
        history = state.get("history", [])
        history.append("Neutral: " + response)

        return {
            **state,
            "neutral_debate_response": response,
            "current_neutral_response": response,   # ✅ Used by Aggressive & Conservative
            "history": history                      # ✅ Used by DebateCoordinator
        }


    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(self.__call__)
