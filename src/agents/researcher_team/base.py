# src/agents/research_team/base_researcher.py

from abc import ABC, abstractmethod
from typing import Dict
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferWindowMemory


class Researcher(ABC):
    def __init__(self, llm: BaseChatModel):
        """
        Abstract base class for researcher agents responsible for generating investment theses.

        Args:
            llm (BaseChatModel): The language model instance.
        """
        self.llm = llm
        self._last_token_count = 0
        self.memory = ConversationBufferWindowMemory(
            k=1, memory_key="chat_history", return_messages=True
        )

    @abstractmethod
    def generate_thesis(self, ticker: str, **kwargs) -> str:
        """
        Generate a fundamental investment thesis.

        Args:
            ticker (str): The stock ticker being analyzed.

        Returns:
            str: The investment thesis.
        """
        pass

    @abstractmethod
    def __call__(self, ticker: str, **kwargs) -> Dict:
        """
        Run the research pipeline and return structured output.

        Args:
            ticker (str): The stock ticker.
            kwargs: Any additional input data.

        Returns:
            Dict: Structured researcher output.
        """
        pass

    def get_last_token_count(self) -> int:
        """
        Return the number of tokens used in the last LLM interaction.

        Returns:
            int: Token count.
        """
        return self._last_token_count
