from typing import Dict
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.json import parse_json_markdown
from langchain_community.callbacks.manager import get_openai_callback
from src.schemas.risk_manager_schemas.summarizer_schema import RiskSummaryOutput


class RiskSummarizerAgent:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=RiskSummaryOutput)
        self.prompt = self._build_prompt()
        self.chain: Runnable = self.prompt | self.llm

    def _build_prompt(self) -> ChatPromptTemplate:
        format_instructions = self.parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

        system_prompt = SystemMessagePromptTemplate.from_template(
            f"""You are a concise and accurate risk summarizer for an investment firm.

Your task is to extract **only risk-relevant data** from analyst reports and return a clean, valid JSON summary.

📌 Rules:
- Output must match the provided schema *exactly* — no explanations.
- Use **numeric values only** for `volatility_indicators` and `financial_flags`, as key-value pairs like `"PE Ratio": 27.3`.
- Omit any key entirely if its value is unknown — no nulls or placeholders.
- No vague terms like "low", "moderate", or "high".
- Use raw numbers like `7.5` — not strings or percentages.
- ❌ Do not wrap in Markdown (no ```json) or add any headings like "Here is the summary".
- ✅ Return valid JSON **only**.

Schema:
{format_instructions}
"""
        )

        human_prompt = HumanMessagePromptTemplate.from_template(
            """Fundamentals Report:
{fundamentals_analysis}

Technical Report:
{technical_analysis}

News Report:
{news_analysis}

Sentiment Report:
{sentiment_analysis}

Summarize risk data for ticker: {ticker}."""
        )

        return ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    def _sanitize_output(self, json_dict: dict) -> dict:
        """Remove non-numeric values from volatility_indicators and financial_flags."""
        def filter_numeric(d: dict) -> dict:
            return {k: v for k, v in d.items() if isinstance(v, (int, float))}

        if "volatility_indicators" in json_dict and isinstance(json_dict["volatility_indicators"], dict):
            json_dict["volatility_indicators"] = filter_numeric(json_dict["volatility_indicators"])

        if "financial_flags" in json_dict and isinstance(json_dict["financial_flags"], dict):
            json_dict["financial_flags"] = filter_numeric(json_dict["financial_flags"])

        return json_dict

    def run(self, inputs: Dict[str, str]) -> RiskSummaryOutput:
        ai_msg = self.chain.invoke(inputs)
        
        # Extract text content from AIMessage object
        raw_text = getattr(ai_msg, "content", None)
        if raw_text is None:
            raise ValueError("LLM returned empty or malformed AIMessage content.")

        try:
            parsed = parse_json_markdown(raw_text)
            cleaned = self._sanitize_output(parsed)
            return RiskSummaryOutput.model_validate(cleaned)
        except Exception as e:
            print(f"[ERROR] Failed to parse LLM output:\n{raw_text}\n\nException:\n{e}")
            raise


    def __call__(self, state: Dict) -> Dict:
        """
        Makes this agent compatible with LangGraph.

        Expects:
            state: dict with keys:
                - fundamentals_analysis
                - technical_analysis
                - news_analysis
                - sentiment_analysis
                - ticker

        Returns:
            dict with key:
                - "risk_summary": RiskSummaryOutput
        """
        inputs = {
            "fundamentals_analysis": state["fundamentals_analysis"],
            "technical_analysis": state["technical_analysis"],
            "news_analysis": state["news_analysis"],
            "sentiment_analysis": state["sentiment_analysis"],
            "ticker": state["ticker"]
        }
        result: RiskSummaryOutput = self.run(inputs)
        return {**state, "risk_summary": result}

