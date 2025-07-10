import json
import re
from typing import Dict, List, Literal, Optional, TypedDict
from src.schemas.risk_manager_schemas.risk_manager_schema import DebateCoordinatorOutput, FinalDecision, AnalystResponse
from langchain_groq import ChatGroq
from src.agents.risk_management_team import (
    AggressiveDebatorAgent,
    ConservativeDebatorAgent,
    NeutralDebatorAgent,
)
import time
from langchain_core.output_parsers import PydanticOutputParser
# ---------------------
# Coordinator Class
# ---------------------
class DebateCoordinatorAgent:
    def __init__(self, n_rounds: int = 2):
        self.n_rounds = n_rounds
        self.aggressive = AggressiveDebatorAgent()
        self.conservative = ConservativeDebatorAgent()
        self.neutral = NeutralDebatorAgent()
        self.summarizer_llm = ChatGroq(model="llama3-8b-8192")

    def run_debate(self, inputs: Dict) -> DebateCoordinatorOutput:
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
You are a final decision maker summarizing a multi-agent investment debate.

Trader's Proposal:
{inputs.get('trader_decision') or inputs.get('reason_for_trade')}

Debate Transcript:
{'\n'.join(history)}

Please return your answer in the following format:
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
