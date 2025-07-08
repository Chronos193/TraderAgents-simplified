from abc import ABC, abstractmethod
from src.schemas.analyst_schemas import SentimentAnalysisOutput
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseChatModel
from langchain_community.callbacks.manager import get_openai_callback
import yfinance as yf
import os
import requests
from datetime import datetime, timedelta, timezone

class SentimentAnalyst(ABC):
    def __init__(self, ticker, llm: BaseChatModel, finnhub_api_key=None):
        self.ticker = ticker
        self.llm = llm
        self.stock = yf.Ticker(ticker)
        self.finnhub_api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self._last_token_count = 0

    def get_recent_headlines(self, max_items=7, max_len=160):
        headlines = []
        if self.finnhub_api_key:
            today = datetime.now(timezone.utc).date()
            from_date = (today - timedelta(days=30)).isoformat()
            to_date = today.isoformat()
            url = f"https://finnhub.io/api/v1/company-news?symbol={self.ticker}&from={from_date}&to={to_date}&token={self.finnhub_api_key}"
            try:
                resp = requests.get(url, timeout=5)
                data = resp.json()
                data_sorted = sorted(data, key=lambda x: x.get("datetime", 0), reverse=True)
                for article in data_sorted:
                    headline = article.get("headline", "")[:max_len]
                    summary = article.get("summary", "")[:max_len]
                    combined = f"{headline} -- {summary}" if summary else headline
                    combined_simple = combined.lower().strip()
                    if combined_simple and all(combined_simple != h.lower().strip() for h in headlines):
                        headlines.append(combined)
                    if len(headlines) >= max_items:
                        break
            except Exception as e:
                print(f"Finnhub API error: {e}")
                pass

        if not headlines:
            news = getattr(self.stock, "news", [])
            for item in news:
                headline = item.get("title", "")[:max_len]
                snippet = item.get("summary", "")[:max_len]
                combined = f"{headline} -- {snippet}" if snippet else headline
                combined_simple = combined.lower().strip()
                if combined_simple and all(combined_simple != h.lower().strip() for h in headlines):
                    headlines.append(combined)
                if len(headlines) >= max_items:
                    break
        return headlines

    def structured_analyze(self) -> SentimentAnalysisOutput:
        headlines = self.get_recent_headlines()
        if not headlines:
            return SentimentAnalysisOutput(
                ticker=self.ticker,
                headlines=[],
                summary_text="No recent news headlines found.",
                sentiment_text="Sentiment analysis not available due to lack of news."
            )

        prompt_summary = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a market sentiment analyst."),
            HumanMessagePromptTemplate.from_template(
                "Summarize the overall sentiment for {ticker} based on these recent headlines in a single sentence:\n{headlines}"
            )
        ])

        prompt_sentiment = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a market sentiment analyst."),
            HumanMessagePromptTemplate.from_template(
                "Analyze the sentiment for {ticker} in under 3 sentences:\n{headlines}\nMention if it is positive, negative, or neutral."
            )
        ])

        inputs = {
            "ticker": self.ticker,
            "headlines": "\n".join([f"- {h}" for h in headlines])
        }

        with get_openai_callback() as cb:
            summary_chain = prompt_summary | self.llm | StrOutputParser()
            summary_text = summary_chain.invoke(inputs)

            sentiment_chain = prompt_sentiment | self.llm | StrOutputParser()
            sentiment_text = sentiment_chain.invoke(inputs)

            self._last_token_count = cb.total_tokens

        return SentimentAnalysisOutput(
            ticker=self.ticker,
            headlines=headlines,
            summary_text=summary_text,
            sentiment_text=sentiment_text
        )

    def analyze(self):
        return self.structured_analyze().sentiment_text

    def summary(self):
        return self.structured_analyze().summary_text

    def get_last_token_count(self):
        return self._last_token_count

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda _: self.structured_analyze())
