# src/agents/research_team/base_researcher.py

from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferWindowMemory
class Researcher(ABC):
    def __init__(self, ticker: str, llm: BaseChatModel):
        self.ticker = ticker
        self.llm = llm
        self._last_token_count = 0
        self.memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True)
    @abstractmethod
    def generate_thesis(self):
        pass

    def get_last_token_count(self):
        return self._last_token_count
