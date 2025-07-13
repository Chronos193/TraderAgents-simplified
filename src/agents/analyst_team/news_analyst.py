from src.schemas.analyst_schemas import NewsAnalysisOutput
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseChatModel
from langchain_community.callbacks.manager import get_openai_callback
import yfinance as yf
import os
import requests
from datetime import datetime, timedelta, timezone


class NewsAnalyst:
    def __init__(self, llm: BaseChatModel, finnhub_api_key=None):
        self.llm = llm
        self.finnhub_api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self._last_token_count = 0

    def get_recent_headlines(self, ticker: str, max_items=5, max_len=120):
        headlines = []

        # Try Finnhub
        if self.finnhub_api_key:
            try:
                today = datetime.now(timezone.utc).date()
                from_date = (today - timedelta(days=14)).isoformat()
                to_date = today.isoformat()
                url = (
                    f"https://finnhub.io/api/v1/company-news"
                    f"?symbol={ticker}&from={from_date}&to={to_date}&token={self.finnhub_api_key}"
                )
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200:
                    print(f"[WARN] Finnhub API error: {resp.status_code}")
                    raise ValueError("Finnhub API call failed.")

                data = resp.json()
                if not isinstance(data, list) or not data:
                    print("[WARN] Finnhub returned no usable news.")
                    raise ValueError("No news returned.")

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
                print(f"[ERROR] Finnhub fetch failed: {e}")
        else:
            print("[WARN] No Finnhub API key provided.")

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

    def structured_analyze(self, ticker: str) -> NewsAnalysisOutput:
        headlines = self.get_recent_headlines(ticker)
        if not headlines:
            return NewsAnalysisOutput(
                ticker=ticker,
                headlines=[],
                summary_text="No recent news headlines found.",
                themes_text="No themes extracted due to lack of data."
            )

        inputs = {
            "ticker": ticker,
            "headlines": "\n".join([f"- {h}" for h in headlines])
        }

        # Prompts
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a financial research analyst."),
            HumanMessagePromptTemplate.from_template(
                "Summarize the news flow for {ticker} in 2-3 sentences:\n{headlines}"
            )
        ])
        theme_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a financial news analyst."),
            HumanMessagePromptTemplate.from_template(
                "Based on the headlines for {ticker}, list 2-3 major themes or events and briefly describe them:\n{headlines}"
            )
        ])

        with get_openai_callback() as cb:
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            summary_text = summary_chain.invoke(inputs)

            themes_chain = theme_prompt | self.llm | StrOutputParser()
            themes_text = themes_chain.invoke(inputs)

            self._last_token_count = cb.total_tokens

        return NewsAnalysisOutput(
            ticker=ticker,
            headlines=headlines,
            summary_text=summary_text,
            themes_text=themes_text
        )

    def get_last_token_count(self):
        return self._last_token_count

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda state: self.__call__(state))

    def __call__(self, state: dict) -> dict:
        ticker = state.get("ticker")
        if not ticker:
            print("[ERROR] Missing ticker in state.")
            return {**state, "news_analysis": None}
    
        try:
            output = self.structured_analyze(ticker)
            return {**state, "news_analysis": output}
        except Exception as e:
            print(f"[ERROR] NewsAnalyst failed for {ticker}: {e}")
            return {**state, "news_analysis": None}
