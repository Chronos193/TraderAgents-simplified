from abc import ABC, abstractmethod

class Analyst(ABC):
    def __init__(self, ticker, llm):
        self.ticker = ticker
        self.llm = llm
        self._last_token_count = 0

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def get_last_token_count(self):
        """
        Return the number of tokens consumed in the last LLM call.
        """
        pass