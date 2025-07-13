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
Ticker: {ticker}
Trade Size: {quantity} shares @ ${price}
New Position Size: {new_position_size} | New Cash: {new_cash_balance}
Portfolio: ${portfolio_value} | Cash: ${cash_balance}

Market:
- Volatility: {volatility} | Volume: {avg_volume} | Sentiment: {sentiment}

Holdings: {holdings}
Sector Exposure: {sector_exposure}

Simulated Drawdown: {simulated_drawdown}

Risk Summary:
- Key Risks: {key_risks}
- Opportunities: {risk_opportunities}
- Volatility Indicators: {volatility_indicators}
- Financial Flags: {financial_flags}
- News Risks: {negative_news_themes}
- Risk Level: {overall_risk_assessment}

Debate Context:
- Conservative: {current_safe_response}
- Neutral: {current_neutral_response}
- History: {history}

Make a bold case for why this is a smart, aggressive move.
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

        # Append to history (optional but useful for prompt context)
        history = state.get("history", [])
        history.append("Aggressive: " + response)

        return {
            **state,
            "aggressive_debate_response": response,
            "current_risky_response": response,   # ✅ Required by Conservative & Neutral
            "history": history                    # ✅ Required by all debators
        }
    
    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(self.__call__)
