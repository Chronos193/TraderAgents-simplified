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
            print(f"🔁 Starting round {round_num}...")

            # Conservative speaks
            conservative_response = self.conservative.run({
                **inputs,
                "current_risky_response": aggressive_response,
                "current_neutral_response": neutral_response,
                "history": "\n".join(history),
            })
            history.append(f"Round {round_num} - Conservative:\n{conservative_response}")

            # Aggressive speaks
            aggressive_response = self.aggressive.run({
                **inputs,
                "current_safe_response": conservative_response,
                "current_neutral_response": neutral_response,
                "history": "\n".join(history),
            })
            history.append(f"Round {round_num} - Aggressive:\n{aggressive_response}")

            # Neutral speaks
            neutral_response = self.neutral.run({
                **inputs,
                "current_risky_response": aggressive_response,
                "current_safe_response": conservative_response,
                "history": "\n".join(history),
            })
            history.append(f"Round {round_num} - Neutral:\n{neutral_response}")

        # ------------------------
        # Generate Summary + Decision
        # ------------------------
        print("🧠 Debate concluded. Generating summary and final decision...\n")

        summary_prompt = f"""
You are a final decision maker summarizing a multi-agent investment debate.

Trader's Proposal:
{inputs.get('trader_decision') or inputs.get('reason_for_trade')}

Debate Transcript:
{'\n'.join(history)}

Please return your answer in the following JSON format:
{{
  "analyst_responses": [
    {{ "role": "aggressive", "final_argument": "..." }},
    {{ "role": "conservative", "final_argument": "..." }},
    {{ "role": "neutral", "final_argument": "..." }}
  ],
  "rounds_transcript": [...],
  "final_decision": {{
    "decision": "accept" | "reject" | "revise",
    "reason": "...",
    "recommendation": "...",
    "confidence": "high" | "medium" | "low",
    "notes": "..."
  }}
}}
"""
        summary_response = self.summarizer_llm.invoke(summary_prompt)

        try:
            # 🛠 Extract only the JSON block if wrapped in backticks
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", summary_response.content)
            json_str = match.group(1) if match else summary_response.content.strip()

            structured_output: DebateCoordinatorOutput = json.loads(json_str)
            structured_output["rounds_transcript"] = history
            return structured_output

        except Exception as e:
            raise ValueError(
                f"[ERROR] Could not parse summary JSON:\n\n{summary_response.content}\n\n{e}"
            )

    def get_total_tokens_used(self) -> int:
        return (
            self.aggressive.get_total_tokens_used() +
            self.conservative.get_total_tokens_used() +
            self.neutral.get_total_tokens_used()
        )
