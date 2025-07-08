from src.schemas.analyst_schemas import NewsAnalysisOutput
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseChatModel
from langchain_community.callbacks.manager import get_openai_callback
import yfinance as yf
import os 

class NewsAnalyst:
    def __init__(self, ticker, llm: BaseChatModel, finnhub_api_key=None):
        self.ticker = ticker
        self.llm = llm
        self.stock = yf.Ticker(ticker)
        self.finnhub_api_key = finnhub_api_key or os.getenv("FINHUB_API_KEY")
        self._last_token_count = 0

    def get_recent_headlines(self, max_items=5, max_len=120):
        """Fetch recent unique news headlines for the ticker, truncated for token efficiency."""
        headlines = []

        if self.finnhub_api_key:
            from datetime import datetime, timedelta, timezone
            import requests

            today = datetime.now(timezone.utc).date()
            from_date = (today - timedelta(days=14)).isoformat()
            to_date = today.isoformat()
            url = (
                f"https://finnhub.io/api/v1/company-news"
                f"?symbol={self.ticker}&from={from_date}&to={to_date}&token={self.finnhub_api_key}"
            )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200:
                    print(f"[WARN] Finnhub API error: Status code {resp.status_code}")
                    raise ValueError("Finnhub API call failed.")

                data = resp.json()
                if not isinstance(data, list) or not data:
                    print("[WARN] Finnhub returned no usable news.")
                    raise ValueError("No news returned.")

                # Sort by most recent
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
                print(f"[ERROR] Failed to fetch from Finnhub: {e}")
        else:
            print("[WARN] No Finnhub API key provided.")

        # Fallback: yfinance
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


    def structured_analyze(self) -> NewsAnalysisOutput:
        headlines = self.get_recent_headlines()
        if not headlines:
            return NewsAnalysisOutput(
                ticker=self.ticker,
                headlines=[],
                summary_text="No recent news headlines found.",
                themes_text="No themes extracted due to lack of data."
            )

        # 🧠 Prompt 1: Summary
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a financial research analyst."),
            HumanMessagePromptTemplate.from_template(
                "Summarize the news flow for {ticker} in 2-3 sentences:\n{headlines}"
            )
        ])

        # 🧠 Prompt 2: Themes
        theme_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a financial news analyst."),
            HumanMessagePromptTemplate.from_template(
                "Based on the headlines for {ticker}, list 2-3 major themes or events and briefly describe them:\n{headlines}"
            )
        ])

        inputs = {
            "ticker": self.ticker,
            "headlines": "\n".join([f"- {h}" for h in headlines])
        }

        with get_openai_callback() as cb:
            summary_chain = summary_prompt | self.llm | StrOutputParser()
            summary_text = summary_chain.invoke(inputs)

            themes_chain = theme_prompt | self.llm | StrOutputParser()
            themes_text = themes_chain.invoke(inputs)

            self._last_token_count = cb.total_tokens

        return NewsAnalysisOutput(
            ticker=self.ticker,
            headlines=headlines,
            summary_text=summary_text,
            themes_text=themes_text
        )

    def get_last_token_count(self):
        return self._last_token_count

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda _: self.structured_analyze())
