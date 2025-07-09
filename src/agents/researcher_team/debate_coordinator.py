from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from src.schemas.researcher_schemas import DebateResult, DebateTurn
import re
class DebateCoordinator:
    def __init__(self, ticker: str, llm: BaseChatModel):
        self.ticker = ticker
        self.llm = llm

    def _generate_response(self, prompt: str, speaker: str) -> DebateTurn:
        system = f"""
        You are a {speaker.lower()} financial debater.
        Respond to the opponent's arguments point-by-point. Address each numbered point made by the opponent directly.
        Maintain a confident, assertive tone without being overly aggressive.
        Limit your reply to approximately 250 words.
        Do not repeat your own thesis unless it directly counters the opponent's claim.
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
            response = chain.invoke({"input": prompt})
            tokens = cb.total_tokens

        return DebateTurn(speaker=speaker, message=response.strip(), tokens_used=tokens)
        
    def _summarize_turn(self, text: str, speaker: str) -> str:
    summarizer_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(f"""
            You are a debate summarizer for financial debates.
            Summarize the key arguments made by the {speaker.lower()} debater.
            Keep only the essential numbered points. Be concise.
            Limit to ~100 words.
        """),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    chain = summarizer_prompt | self.llm | StrOutputParser()
    with get_openai_callback() as cb:
        summary = chain.invoke({"input": text})
    return summary.strip()


    def conduct_debate(self, bullish_thesis: str, bearish_thesis: str, rounds: int = 3) -> DebateResult:
        turns = []
        prompt_bear = bearish_thesis
        prompt_bull = bullish_thesis

        for r in range(rounds):
            bull_turn = self._generate_response(prompt_bear, "Bullish")
            turns.append(bull_turn)

            bear_turn = self._generate_response(bull_turn.message, "Bearish")
            turns.append(bear_turn)

            prompt_bear = self._summarize_turn(bear_turn.message, "Bearish")
            prompt_bull = self._summarize_turn(bull_turn.message, "Bullish")


        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
You are a neutral financial analyst summarizing a stock debate.
Your goal is to objectively compare the two sides.
For each investor (Bullish and Bearish), rate them in the following:
- Logic
- Use of evidence (specific data, facts, examples)
- Persuasiveness

After scoring both sides, decide a winner (Bullish, Bearish, or Tie) based on the overall performance.

At the end of your summary, clearly state the winner in this exact format:
"Winner: Bullish", "Winner: Bearish", or "Winner: Tie"
"""),
            HumanMessagePromptTemplate.from_template(
                """
                Summarize the debate between a Bullish and Bearish investor on {ticker}.
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
        if winner_match:
            winner = winner_match.group(1).capitalize()
        else:
            winner = "Tie"  # fallback if no match


        return DebateResult(
            turns=turns,
            summary=result.strip(),
            winner=winner,
            total_tokens=sum(t.tokens_used for t in turns) + summary_tokens
        )
