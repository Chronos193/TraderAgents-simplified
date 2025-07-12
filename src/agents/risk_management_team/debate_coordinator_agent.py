import json
import time
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from src.schemas.risk_manager_schemas.risk_manager_schema import DebateCoordinatorOutput
from src.agents.risk_management_team import (
    AggressiveDebatorAgent,
    ConservativeDebatorAgent,
    NeutralDebatorAgent
)

class DebateCoordinatorAgent:
    def __init__(self, n_rounds: int = 2):
        self.n_rounds = n_rounds
        self.aggressive = AggressiveDebatorAgent()
        self.conservative = ConservativeDebatorAgent()
        self.neutral = NeutralDebatorAgent()
        self.summarizer_llm = ChatGroq(model="llama3-8b-8192")
        # ✅ Lazy import to avoid circular import
        from .summarizer_risk_management import RiskSummarizerAgent
        self.risk_summarizer = RiskSummarizerAgent(llm=self.summarizer_llm)  # ✅ Add summarizer instance

    def run_debate(self, inputs: Dict) -> DebateCoordinatorOutput:
        # ✅ Inject structured risk summary into debate inputs
        risk_summary = self.risk_summarizer.run(inputs)
        inputs.update(risk_summary.model_dump())  # flatten into input dict

        history: List[str] = []
        aggressive_response = ""
        conservative_response = ""
        neutral_response = ""

        for round_num in range(1, self.n_rounds + 1):
            if (round_num) % 2 == 0 and (round_num) < self.n_rounds:
                print("🕒 Waiting 60 seconds to respect TPM limits...")
                time.sleep(60)
            print(f"🔁 Starting round {round_num}...")

            conservative_response = self.conservative.run({
                **inputs,
                "current_risky_response": aggressive_response,
                "current_neutral_response": neutral_response,
                "history": "\n".join(history),
            })
            history.append(f"Round {round_num} - Conservative:\n{conservative_response}")

            aggressive_response = self.aggressive.run({
                **inputs,
                "current_safe_response": conservative_response,
                "current_neutral_response": neutral_response,
                "history": "\n".join(history),
            })
            history.append(f"Round {round_num} - Aggressive:\n{aggressive_response}")

            neutral_response = self.neutral.run({
                **inputs,
                "current_risky_response": aggressive_response,
                "current_safe_response": conservative_response,
                "history": "\n".join(history),
            })
            history.append(f"Round {round_num} - Neutral:\n{neutral_response}")

        # ✅ Use Pydantic parser
        parser = PydanticOutputParser(pydantic_object=DebateCoordinatorOutput)
        format_instructions = parser.get_format_instructions()

        print("=====================================Conservative_response=============================================")
        print(conservative_response)
        print("=====================================Neutral_response=============================================")
        print(neutral_response)
        print("=====================================Aggressive_response=============================================")
        print(aggressive_response)
        print("🧠 Debate concluded. Generating summary and final decision...\n")

        summary_prompt = f"""
You are a final decision maker summarizing a multi-agent investment debate involving three analysts: Conservative, Aggressive, and Neutral.

Trader's Proposal:
{inputs.get('trader_decision') or inputs.get('reason_for_trade')}

Debate Transcript:
{'\n'.join(history)}

Your job is to synthesize the debate and make a final decision on the trade. Choose between:
- "accept" → The trade is sound and should be executed as proposed.
- "revise" → The trade has merit but should be adjusted (e.g., timing, size, portfolio fit).
- "reject" → The trade is fundamentally flawed or too risky, and should not be executed in any form.

When deciding between **revise** and **reject**, consider:
- Is the core thesis promising but poorly implemented? → Choose **revise**.
- Are the market conditions or portfolio risks too severe even with adjustments? → Choose **reject**.

Provide:
1. The structured analyst responses.
2. Your final decision (accept / revise / reject).
3. Your rationale.
4. A specific recommendation to the trader.
5. Your confidence level.
6. Any notes the team should consider for future trades.

Please return your answer in **valid JSON format** with this structure:
{format_instructions}
"""

        summary_response = self.summarizer_llm.invoke(summary_prompt)

        try:
            parsed_output: DebateCoordinatorOutput = parser.parse(summary_response.content)
            parsed_output.rounds_transcript = history
            return parsed_output

        except Exception as e:
            raise ValueError(
                f"[ERROR] Could not parse structured summary:\n\n{summary_response.content}\n\n{e}"
            )

    def get_total_tokens_used(self) -> int:
        return (
            self.aggressive.get_total_tokens_used() +
            self.conservative.get_total_tokens_used() +
            self.neutral.get_total_tokens_used()
        )
