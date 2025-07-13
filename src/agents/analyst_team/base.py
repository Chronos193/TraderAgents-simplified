from abc import ABC, abstractmethod
from typing import Any, Dict


class Analyst(ABC):
    def __init__(self, llm: Any):
        """
        Abstract base class for analyst agents (fundamental, technical, sentiment, news, etc.).

        Args:
            llm (Any): The language model instance used by the analyst.
        """
        self.llm = llm
        self._last_token_count = 0

    @abstractmethod
    def analyze(self, ticker: str, inputs: dict) -> str:
        """
        Run the full analysis for a given ticker using the provided inputs.

        Args:
            ticker (str): The stock ticker to analyze.
            inputs (dict): Contextual input data relevant to the analysis.

        Returns:
            str: Analyst's raw output or report.
        """
        pass

    @abstractmethod
    def summary(self, ticker: str, inputs: dict) -> str:
        """
        Provide a concise, structured summary for a given ticker suitable for downstream agents.

        Args:
            ticker (str): The stock ticker to analyze.
            inputs (dict): Same input context as `analyze`.

        Returns:
            str: Shortened and focused output.
        """
        pass

    @abstractmethod
    def __call__(self, ticker: str, inputs: Dict) -> Dict:
        """
        Run both analysis and summary, return both in a structured dict.

        Args:
            ticker (str): The stock ticker.
            inputs (Dict): Contextual data.

        Returns:
            Dict: Dictionary with both raw and summarized analysis.
        """
        pass

    @abstractmethod
    def get_last_token_count(self) -> int:
        """
        Return the number of tokens consumed in the last LLM call.

        Returns:
            int: Number of tokens.
        """
        pass
