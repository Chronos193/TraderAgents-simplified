from typing import Dict
from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain_community.callbacks.manager import get_openai_callback
from langchain.memory import CombinedMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory
class BaseRiskDebator(ABC):
    def __init__(self):
        self.llm = ChatGroq(model="llama3-8b-8192")
        self.memory =self.memory = ConversationBufferWindowMemory(memory_key="history",return_messages=True,k=1)
        self.prompt_template = self.define_prompt()
        self.total_tokens_used = 0
        self.total_cost = 0.0

    @abstractmethod
    def define_prompt(self) -> ChatPromptTemplate:
        pass

    def run(self, inputs: Dict) -> str:
        prompt = self.prompt_template.format_messages(**inputs)

        with get_openai_callback() as cb:
            response = self.llm.invoke(prompt)
            self.total_tokens_used += cb.total_tokens
            self.total_cost += cb.total_cost

        self.memory.chat_memory.add_user_message(prompt[-1].content)
        self.memory.chat_memory.add_ai_message(response.content)
        return response.content

    def get_total_tokens_used(self) -> int:
        return self.total_tokens_used

    def get_total_cost(self) -> float:
        return self.total_cost
