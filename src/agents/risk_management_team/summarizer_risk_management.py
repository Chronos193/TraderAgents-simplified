from typing import Dict
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel

from src.schemas.risk_manager_schemas.summarizer_schema import RiskSummaryOutput


class RiskSummarizerAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=RiskSummaryOutput)
        self.prompt = self._build_prompt()

        # LLM Chain: Prompt → LLM → Structured Output
        self.chain: Runnable = self.prompt | self.llm | self.parser

    def _build_prompt(self) -> ChatPromptTemplate:
        system_prompt = SystemMessagePromptTemplate.from_template(
            """You are a precise and efficient risk summary agent for an investment firm.

Your task is to extract only risk-relevant information from multiple analyst reports and return a structured JSON summary.

📌 Guidelines:
- Output must be a **valid JSON object** following the schema exactly.
- Respond with **numeric values only** for `volatility_indicators` and `financial_flags`.
- Do NOT use strings like "moderate", "high", or "N/A".
- If a numeric value is missing or unknown, simply **omit the key**.
- Do NOT include `null` anywhere in the output. Instead leave it empty.
- Use floats like 12.3 instead of strings like "12.3%".
- Do NOT explain or comment — just output the JSON.

Ensure the structure exactly matches:

{format_instructions}
"""
        )

        human_prompt = HumanMessagePromptTemplate.from_template(
            """Fundamentals Analysis:
{fundamentals}

Technical Analysis:
{technical}

News Analysis:
{news}

Sentiment Analysis:
{sentiment}

Generate a structured risk summary for the ticker: {ticker}."""
        )

        return ChatPromptTemplate.from_messages(
            [system_prompt, human_prompt]
        ).partial(format_instructions=self.parser.get_format_instructions())

    def run(self, inputs: Dict[str, str]) -> RiskSummaryOutput:
        return self.chain.invoke(inputs)
