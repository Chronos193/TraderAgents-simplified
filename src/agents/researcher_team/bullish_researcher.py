from abc import ABC, abstractmethod
from typing import List, Dict
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain.memory import ConversationBufferMemory
import re
from src.agents.researcher_team import Researcher
from src.schemas.researcher_schemas import BullishThesisOutput

class BullishResearcher(Researcher):
    def __init__(self, ticker: str, llm: BaseChatModel):
        super().__init__(ticker, llm)

    def generate_thesis(self, summary_bundle: Dict[str, str]) -> BullishThesisOutput:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a bullish financial researcher who analyzes upside potential in stocks."
            ),
            HumanMessagePromptTemplate.from_template(
                """
                Construct a bullish investment thesis for {ticker} based on the following summaries:
                📊 Technical: {technical}
                🧠 Sentiment: {sentiment}
                📰 News: {news}
                📈 Fundamentals: {fundamental}

                Provide:
                1. A short 1-line bullish thesis
                2. 2–3 supporting points (bulleted)
                3. A confidence score (0.0–1.0)
                """
            )
        ])

        inputs = {
            "ticker": self.ticker,
            "technical": summary_bundle.get("technical", "N/A"),
            "sentiment": summary_bundle.get("sentiment", "N/A"),
            "news": summary_bundle.get("news", "N/A"),
            "fundamental": summary_bundle.get("fundamental", "N/A")
        }

        chain = prompt | self.llm | StrOutputParser()
        with get_openai_callback() as cb:
            response = chain.invoke(inputs)
            self._last_token_count = cb.total_tokens

        self.memory.chat_memory.add_user_message(str(inputs))
        self.memory.chat_memory.add_ai_message(response)

        lines = response.strip().split("\n")
        lines = [line.strip() for line in lines if line.strip()]
        thesis = lines[0]
        supporting_points = [line.strip("-* ") for line in lines[1:] if not "confidence" in line.lower()]
        confidence_line = next((line for line in lines if "confidence" in line.lower()), "")
        match = re.search(r"([01](\.\d+)?)", confidence_line)
        confidence = float(match.group(1)) if match else 0.5

        return BullishThesisOutput(
            ticker=self.ticker,
            thesis=thesis,
            supporting_points=supporting_points,
            confidence=round(confidence, 2)
        )

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda _: self.generate_thesis())
