from typing import Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.callbacks.manager import get_openai_callback
from src.schemas.researcher_schemas import DebateResult, DebateTurn
import re


class DebateCoordinator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def _generate_response(self, summary: str, latest_argument: str, speaker: str) -> DebateTurn:
        system = (
            f"You are a {speaker.lower()} financial debater.\n\n"
            f"Debate so far:\n{summary}\n\n"
            "Your task:\n"
            "- Respond to the opponent's latest arguments point-by-point.\n"
            "- Address each numbered point directly.\n"
            "- Keep your tone confident and assertive, not aggressive.\n"
            "- Do not repeat your own arguments unless needed to counter.\n"
            "- Limit to about 250 words."
        )

        chain = (
            ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system),
                HumanMessagePromptTemplate.from_template("{input}")
            ]) | self.llm | StrOutputParser()
        )

        with get_openai_callback() as cb:
            response = chain.invoke({"input": latest_argument})
            tokens = cb.total_tokens

        return DebateTurn(speaker=speaker, message=response.strip(), tokens_used=tokens)

    def _update_summary(self, old_summary: str, new_turn: DebateTurn) -> str:
        summarizer_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
                You are a neutral financial debate summarizer.
                Update the rolling summary with the new turn.
                Keep it factual, short, and highlight only new points.
                Limit to about 100 words.
            """),
            HumanMessagePromptTemplate.from_template(
                "Previous Summary:\n{summary}\n\nNew Turn:\n{turn}"
            )
        ])
        chain = summarizer_prompt | self.llm | StrOutputParser()

        with get_openai_callback():
            return chain.invoke({
                "summary": old_summary,
                "turn": f"{new_turn.speaker}: {new_turn.message}"
            }).strip()

    def conduct_debate(self, ticker: str, bullish_thesis: str, bearish_thesis: str, rounds: int = 2) -> DebateResult:
        turns = []
        prompt_bull = bullish_thesis
        prompt_bear = bearish_thesis
        summary_so_far = ""

        for _ in range(rounds):
            bull_turn = self._generate_response(summary_so_far, prompt_bear, "Bullish")
            turns.append(bull_turn)
            summary_so_far = self._update_summary(summary_so_far, bull_turn)

            bear_turn = self._generate_response(summary_so_far, bull_turn.message, "Bearish")
            turns.append(bear_turn)
            summary_so_far = self._update_summary(summary_so_far, bear_turn)

            prompt_bull = bull_turn.message
            prompt_bear = bear_turn.message

        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
                You are a neutral financial analyst summarizing a stock debate.
                Objectively compare both sides.

                For each investor (Bullish and Bearish), rate:
                - Logic
                - Use of evidence (facts, data, examples)
                - Persuasiveness

                Decide the winner: Bullish, Bearish, or Tie.
                While deciding the winner:
                - Carefully analyze the scores of Logic, Evidence, and Persuasiveness.
                - If one side has higher scores in at least 2 out of 3 categories, declare them the winner.
                - Avoid declaring a tie unless the scores and arguments are nearly equal in all dimensions.
                At the end, **state the winner exactly as:**
                Winner: Bullish  --OR--  Winner: Bearish  --OR--  Winner: Tie
            """),
            HumanMessagePromptTemplate.from_template(
                "Summarize the debate about {ticker}.\n\nDebate Transcript:\n{transcript}"
            )
        ])

        transcript = "\n\n".join(f"{turn.speaker}: {turn.message}" for turn in turns)
        inputs = {"ticker": ticker, "transcript": transcript}

        with get_openai_callback() as cb:
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            result = summary_chain.invoke(inputs)
            summary_tokens = cb.total_tokens

        winner_match = re.search(r"\**\bwinner\b\**\s*[:\-]?\s*(bullish|bearish|tie)", result, re.IGNORECASE)
        winner = winner_match.group(1).capitalize() if winner_match else "Tie"


        return DebateResult(
            turns=turns,
            summary=result.strip(),
            winner=winner,
            total_tokens=sum(t.tokens_used for t in turns) + summary_tokens
        )

    def __call__(self, state: Dict) -> Dict:
        ticker = state.get("ticker")
        bullish = state.get("bullish_thesis")
        bearish = state.get("bearish_thesis")

        if not ticker or not bullish or not bearish:
            print("[WARN] DebateCoordinator missing input.")
            return {**state, "research_debate_result": None}

        debate_result = self.conduct_debate(
            ticker,
            bullish.thesis,
            bearish.thesis
        )
        return {**state, "research_debate_result": debate_result}

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda state: self.__call__(state))
