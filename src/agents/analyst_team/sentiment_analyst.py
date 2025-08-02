from abc import ABC
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
    def __init__(self, llm: BaseChatModel, finnhub_api_key=None):
        self.llm = llm
        self.finnhub_api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self._last_token_count = 0

    def get_recent_headlines(self, ticker: str, max_items=7, max_len=160):
        headlines = []

        # Finnhub
        if self.finnhub_api_key:
            today = datetime.now(timezone.utc).date()
            from_date = (today - timedelta(days=30)).isoformat()
            to_date = today.isoformat()
            url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={self.finnhub_api_key}"
            try:
                resp = requests.get(url, timeout=30)
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
                print(f"[ERROR] Finnhub API error: {e}")
        else:
            print("[WARN] No FINNHUB_API_KEY provided.")

        # Fallback: yfinance
        if not headlines:
            try:
                stock = yf.Ticker(ticker)
                news = getattr(stock, "news", [])
                for item in news:
                    headline = item.get("title", "")[:max_len]
                    snippet = item.get("summary", "")[:max_len]
                    combined = f"{headline} -- {snippet}" if snippet else headline
                    combined_simple = combined.lower().strip()
                    if combined_simple and all(combined_simple != h.lower().strip() for h in headlines):
                        headlines.append(combined)
                    if len(headlines) >= max_items:
                        break
            except Exception as e:
                print(f"[ERROR] yfinance fallback failed: {e}")

        return headlines

    def structured_analyze(self, ticker: str) -> SentimentAnalysisOutput:
        headlines = self.get_recent_headlines(ticker)
        if not headlines:
            return SentimentAnalysisOutput(
                ticker=ticker,
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
            "ticker": ticker,
            "headlines": "\n".join([f"- {h}" for h in headlines])
        }

        with get_openai_callback() as cb:
            summary_chain = prompt_summary | self.llm | StrOutputParser()
            summary_text = summary_chain.invoke(inputs)

            sentiment_chain = prompt_sentiment | self.llm | StrOutputParser()
            sentiment_text = sentiment_chain.invoke(inputs)

            self._last_token_count = cb.total_tokens

        return SentimentAnalysisOutput(
            ticker=ticker,
            headlines=headlines,
            summary_text=summary_text,
            sentiment_text=sentiment_text
        )

    def analyze(self, ticker: str):
        return self.structured_analyze(ticker).sentiment_text

    def summary(self, ticker: str):
        return self.structured_analyze(ticker).summary_text

    def get_last_token_count(self):
        return self._last_token_count

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda state: self.__call__(state))

    def __call__(self, state: dict) -> dict:
        ticker = state.get("ticker")
        if not ticker:
            print("[ERROR] Missing ticker in state.")
            return {**state, "sentiment_analysis": None}
    
        try:
            output = self.structured_analyze(ticker)
            return {**state, "sentiment_analysis": output}
        except Exception as e:
            print(f"[ERROR] SentimentAnalyst failed for {ticker}: {e}")
            return {**state, "sentiment_analysis": None}
