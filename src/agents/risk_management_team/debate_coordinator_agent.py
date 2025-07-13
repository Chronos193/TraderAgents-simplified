import json
import time
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.callbacks.manager import get_openai_callback

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
        self._last_token_count = 0
        self.last_summary_raw_output: Optional[str] = None
        self._error_raised = False

        # Avoid circular import
        from .summarizer_risk_management import RiskSummarizerAgent
        self.risk_summarizer = RiskSummarizerAgent(llm=self.summarizer_llm)

    def run_debate(self, inputs: Dict) -> Optional[DebateCoordinatorOutput]:
        inputs = inputs.copy()
        risk_summary = self.risk_summarizer.run(inputs)
        inputs.update(risk_summary.model_dump())

        history: List[str] = []
        aggressive_response = ""
        conservative_response = ""
        neutral_response = ""

        for round_num in range(1, self.n_rounds + 1):
            if round_num % 2 == 0 and round_num < self.n_rounds:
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

        print("===================================== Conservative_response =============================================")
        print(conservative_response)
        print("===================================== Neutral_response ==================================================")
        print(neutral_response)
        print("===================================== Aggressive_response ===============================================")
        print(aggressive_response)
        print("🧠 Debate concluded. Generating summary and final decision...\n")

        return self._summarize_debate(inputs, history)

    def _summarize_debate(
        self,
        inputs: Dict,
        history: List[str]
    ) -> Optional[DebateCoordinatorOutput]:
        parser = PydanticOutputParser(pydantic_object=DebateCoordinatorOutput)
        format_instructions = parser.get_format_instructions()

        summary_prompt = f"""
You are a final decision maker summarizing a multi-agent investment debate involving three analysts: Conservative, Aggressive, and Neutral.

Trader's Proposal:
{inputs.get('trader_decision') or inputs.get('reason_for_trade')}

Debate Transcript:
{''.join(history)}

Your job is to synthesize the debate and make a final decision on the trade. Choose between:
- "accept"
- "revise"
- "reject"

Please respond with only valid JSON. Do not include markdown formatting or any extra explanation.
{format_instructions}
"""

        with get_openai_callback() as cb:
            response = self.summarizer_llm.invoke(summary_prompt)
            self._last_token_count = cb.total_tokens
            self.last_summary_raw_output = getattr(response, "content", None)

        if not self.last_summary_raw_output:
            print("❌ LLM response was empty.")
            self._error_raised = True
            return None

        try:
            parsed = parser.parse(self.last_summary_raw_output)
            parsed.rounds_transcript = history
            return parsed
        except Exception:
            print("⚠️ Structured parse failed. Trying fallback JSON repair...")
            try:
                repaired = self._extract_json_from_string(self.last_summary_raw_output)
                print("🧪 Attempting JSON.loads on:\n", repaired[:3000])
                parsed_dict = json.loads(repaired)
                parsed_dict['rounds_transcript'] = history
                parsed = DebateCoordinatorOutput(**parsed_dict)
                return parsed
            except Exception as e2:
                print("❌ Final JSON parsing failed.")
                print("🪵 Raw LLM Output:\n", self.last_summary_raw_output)
                print("🐛 Error:\n", str(e2))
                self._error_raised = True
                return None

    def _extract_json_from_string(self, text: str) -> str:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No valid JSON object found.")
        return text[start:end + 1]

    def get_total_tokens_used(self) -> int:
        return (
            self.aggressive.get_total_tokens_used() +
            self.conservative.get_total_tokens_used() +
            self.neutral.get_total_tokens_used() +
            self._last_token_count
        )

    def __call__(self, state: Dict) -> Dict:
        result = self.run_debate(state)
        if result is None:
            self._error_raised = True
            raise RuntimeError("💥 DebateCoordinatorAgent returned None — parsing failed.")
        return {**state, "debate_coordinator_output": result}

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(self.__call__)

    def error_occurred(self) -> bool:
        return self._error_raised
