from src.schemas.researcher_schemas import BearishThesisOutput
from typing import Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from src.agents.researcher_team import Researcher
from langchain_core.runnables import RunnableLambda
import re


class BearishResearcher(Researcher):
    def __init__(self, llm: BaseChatModel):
        super().__init__(llm=llm)

    def generate_thesis(self, ticker: str, summary_bundle: Dict[str, str]) -> BearishThesisOutput:
        system_msg = (
            "You are a bearish financial analyst in a formal debate.\n"
            "Respond in this strict format:\n"
            "- Line 1: A bolded one-line bearish thesis.\n"
            "- Lines 2–4: Three *numbered* bullet points (each under 20 words).\n"
            "- Line 5: Confidence: <score from 0.0 to 1.0>\n\n"
            "Write sharply, with no fluff. Stay under 100 words total. Do not invent facts."
        )

        human_template = (
            "Debate the stock {ticker} from a bearish viewpoint.\n\n"
            "Here are the analyst summaries:\n"
            "📊 Technical: {technical}\n"
            "🧠 Sentiment: {sentiment}\n"
            "📰 News: {news}\n"
            "📈 Fundamentals: {fundamental}"
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_msg),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        inputs = {
            "ticker": ticker,
            "technical": summary_bundle.get("technical", "N/A"),
            "sentiment": summary_bundle.get("sentiment", "N/A"),
            "news": summary_bundle.get("news", "N/A"),
            "fundamental": summary_bundle.get("fundamental", "N/A")
        }

        chain = prompt | self.llm | StrOutputParser()

        with get_openai_callback() as cb:
            output = chain.invoke(inputs)
            self._last_token_count = cb.total_tokens

        self.memory.chat_memory.add_user_message(str(inputs))
        self.memory.chat_memory.add_ai_message(output)

        lines = [line.strip() for line in output.strip().split("\n") if line.strip()]
        thesis = re.sub(r"\*\*(.*?)\*\*", r"\1", lines[0]) if lines else "No thesis."

        supporting_points = []
        for line in lines[1:4]:
            point = re.sub(r"^\d+[\.\)]\s*", "", line)
            supporting_points.append(point.strip())

        confidence = 0.5
        for line in lines:
            if "confidence" in line.lower():
                match = re.search(r"(\d+(\.\d+)?)", line)
                if match:
                    confidence = float(match.group(1))
                    break

        return BearishThesisOutput(
            ticker=ticker,
            thesis=thesis,
            supporting_points=supporting_points,
            confidence=round(confidence, 2)
        )

    def __call__(self, state: dict) -> dict:
        ticker = state.get("ticker")
        if not ticker:
            print("[ERROR] Ticker missing from state.")
            return {**state, "bearish_thesis": None}
        try:
            summaries = {
                "technical": getattr(state.get("technical_analysis"), "recommendation", "N/A"),
                "sentiment": getattr(state.get("sentiment_analysis"), "sentiment_text", "N/A"),
                "news": getattr(state.get("news_analysis"), "summary_text", "N/A"),
                "fundamental": getattr(state.get("fundamentals_analysis"), "analysis_text", "N/A"),
            }

            output = self.generate_thesis(ticker, summaries)
            return {**state, "bearish_thesis": output}
        except Exception as e:
            print(f"[ERROR] BearishResearcher failed: {e}")
            return {**state, "bearish_thesis": None}

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda state: self.__call__(state))
