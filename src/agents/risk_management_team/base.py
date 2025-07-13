from abc import ABC, abstractmethod
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain_community.callbacks.manager import get_openai_callback


class BaseRiskDebator(ABC):
    def __init__(self, llm: BaseChatModel = None):
        self.llm = llm or ChatGroq(model="llama3-8b-8192")
        self.memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, k=1)
        self.prompt_template: ChatPromptTemplate = self.define_prompt()
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.ticker = None  # ← ✅ store the ticker passed via __call__

    @abstractmethod
    def define_prompt(self) -> ChatPromptTemplate:
        pass

    def run(self, inputs: Dict) -> str:
        # ✅ Make a safe copy
        inputs = inputs.copy()

        # ✅ Auto-fill missing input variables with "N/A"
        required_keys = self.prompt_template.input_variables
        missing_keys = [k for k in required_keys if k not in inputs]
        for key in missing_keys:
            inputs[key] = "N/A"
            print(f"[WARN] Missing input '{key}' — defaulting to 'N/A'")

        prompt = self.prompt_template.format_messages(**inputs)

        with get_openai_callback() as cb:
            response = self.llm.invoke(prompt)
            self.total_tokens_used += cb.total_tokens
            self.total_cost += cb.total_cost

        self.memory.chat_memory.add_user_message(prompt[-1].content)
        self.memory.chat_memory.add_ai_message(response.content)

        return response.content.strip()

    def __call__(self, ticker: str, state: Dict) -> Dict:
        self.ticker = ticker  # ← ✅ inject ticker into the agent instance
        state = {**state, "ticker": self.ticker}
        result = self.run(state)
        return {**state, f"{self.__class__.__name__.lower()}_response": result}

    def get_total_tokens_used(self) -> int:
        return self.total_tokens_used

    def get_total_cost(self) -> float:
        return self.total_cost
