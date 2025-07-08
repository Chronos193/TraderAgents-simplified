
from src.schemas.researcher_schemas import BearishThesisOutput
from typing import Dict
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from src.agents.researcher_team import Researcher  # your abstract base class
from langchain_core.runnables import RunnableLambda

class BearishResearcher(Researcher):
    def __init__(self, ticker: str, llm: BaseChatModel):
        super().__init__(ticker, llm)

    def generate_thesis(self, summary_bundle: Dict[str, str]) -> BearishThesisOutput:
        system_msg = "You are a bearish financial researcher focused on identifying risks in a company's outlook."

        human_template = (
            "You are analyzing the stock {ticker} from a bearish viewpoint.\n\n"
            "Here are summaries from various analyst perspectives:\n"
            "📊 Technical: {technical}\n"
            "🧠 Sentiment: {sentiment}\n"
            "📰 News: {news}\n"
            "📈 Fundamentals: {fundamental}\n\n"
            "Based on these, provide:\n"
            "1. A one-line bearish thesis.\n"
            "2. Three key supporting_points (bulleted).\n"
            "3. A confidence score between 0.0 and 1.0.\n"
            "Be critical but realistic. Do not invent facts."
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_msg),
            HumanMessagePromptTemplate.from_template(human_template)
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
            output = chain.invoke(inputs)
            self._last_token_count = cb.total_tokens

        self.memory.chat_memory.add_user_message(str(inputs))
        self.memory.chat_memory.add_ai_message(output)

        lines = output.strip().split("\n")
        lines = [line for line in lines if line.strip()]

        thesis = lines[0].strip()
        supporting_points = [line.strip("-\u2022 ") for line in lines[1:4]]
        # Robust confidence extraction
        import re
        confidence = 0.5
        for line in lines:
            if "confidence" in line.lower():
                match = re.search(r"(\d+(\.\d+)?)", line)
                if match:
                    try:
                        confidence = float(match.group(1))
                    except:
                        pass

        return BearishThesisOutput(
            ticker=self.ticker,
            thesis=thesis,
            supporting_points=supporting_points,  # ✅ correct key
            confidence=round(confidence, 2)
        )

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda _: self.generate_thesis())