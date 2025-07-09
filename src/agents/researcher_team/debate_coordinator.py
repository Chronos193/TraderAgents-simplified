from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from src.schemas.researcher_schemas import DebateResult, DebateTurn
import re


class DebateCoordinator:
    """
    DebateCoordinator runs a multi-round debate between a bullish and bearish debater
    on a given stock ticker. It maintains a rolling summary to preserve context
    while keeping token usage manageable.
    """

    def __init__(self, ticker: str, llm: BaseChatModel):
        """
        :param ticker: The stock ticker symbol.
        :param llm: The language model to use (must inherit BaseChatModel).
        """
        self.ticker = ticker
        self.llm = llm

    def _generate_response(self, summary: str, latest_argument: str, speaker: str) -> DebateTurn:
        """
        Generate a single debater's response given the rolling summary
        and the opponent's latest argument.

        :param summary: Rolling summary of debate so far.
        :param latest_argument: Opponent's last turn.
        :param speaker: "Bullish" or "Bearish".
        :return: DebateTurn
        """
        system = f"""
        You are a {speaker.lower()} financial debater.

        Debate so far:
        {summary}

        Your task:
        - Respond to the opponent's latest arguments point-by-point.
        - Address each numbered point directly.
        - Keep your tone confident and assertive, not aggressive.
        - Do not repeat your own arguments unless needed to counter.
        - Limit to about 250 words.
        """

        chain = (
            ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system),
                HumanMessagePromptTemplate.from_template("{input}")
            ])
            | self.llm
            | StrOutputParser()
        )

        with get_openai_callback() as cb:
            response = chain.invoke({"input": latest_argument})
            tokens = cb.total_tokens

        return DebateTurn(speaker=speaker, message=response.strip(), tokens_used=tokens)

    def _update_summary(self, old_summary: str, new_turn: DebateTurn) -> str:
        """
        Update the rolling summary with a new turn.

        :param old_summary: Current summary.
        :param new_turn: New DebateTurn to incorporate.
        :return: Updated summary.
        """
        summarizer_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
                You are a neutral financial debate summarizer.
                Update the rolling summary with the new turn.
                Keep it factual, short, and highlight only new points.
                Limit to about 100 words.
            """),
            HumanMessagePromptTemplate.from_template(
                """
                Previous Summary:
                {summary}

                New Turn:
                {turn}
                """
            )
        ])

        chain = summarizer_prompt | self.llm | StrOutputParser()
        with get_openai_callback() as cb:
            updated_summary = chain.invoke({
                "summary": old_summary,
                "turn": f"{new_turn.speaker}: {new_turn.message}"
            })

        return updated_summary.strip()

    def conduct_debate(self, bullish_thesis: str, bearish_thesis: str, rounds: int = 3) -> DebateResult:
        """
        Run the full multi-round debate and return results.

        :param bullish_thesis: Initial bullish opening statement.
        :param bearish_thesis: Initial bearish opening statement.
        :param rounds: Number of back-and-forth rounds.
        :return: DebateResult
        """
        turns = []
        prompt_bull = bullish_thesis
        prompt_bear = bearish_thesis

        summary_so_far = ""

        for r in range(rounds):
            # Bullish turn
            bull_turn = self._generate_response(summary_so_far, prompt_bear, "Bullish")
            turns.append(bull_turn)
            summary_so_far = self._update_summary(summary_so_far, bull_turn)

            # Bearish turn
            bear_turn = self._generate_response(summary_so_far, bull_turn.message, "Bearish")
            turns.append(bear_turn)
            summary_so_far = self._update_summary(summary_so_far, bear_turn)

            # Next round's prompts
            prompt_bull = bull_turn.message
            prompt_bear = bear_turn.message

        # Final summary + winner
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
                You are a neutral financial analyst summarizing a stock debate.
                Objectively compare both sides.

                For each investor (Bullish and Bearish), rate:
                - Logic
                - Use of evidence (facts, data, examples)
                - Persuasiveness

                Decide the winner: Bullish, Bearish, or Tie.

                At the end, state the winner exactly as:
                "Winner: Bullish"
            """),
            HumanMessagePromptTemplate.from_template(
                """
                Summarize the debate about {ticker}.

                Debate Transcript:
                {transcript}
                """
            )
        ])

        transcript = "\n\n".join([f"{turn.speaker}: {turn.message}" for turn in turns])
        summary_inputs = {"ticker": self.ticker, "transcript": transcript}

        with get_openai_callback() as cb:
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            result = summary_chain.invoke(summary_inputs)
            summary_tokens = cb.total_tokens

        winner_match = re.search(r"winner\s*[:\-]?\s*(bullish|bearish|tie)", result, re.IGNORECASE)
        winner = winner_match.group(1).capitalize() if winner_match else "Tie"

        return DebateResult(
            turns=turns,
            summary=result.strip(),
            winner=winner,
            total_tokens=sum(t.tokens_used for t in turns) + summary_tokens
        )
