from src.schemas.analyst_schemas import TechnicalAnalysisOutput
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_community.callbacks.manager import get_openai_callback
import yfinance as yf
import pandas as pd
import numpy as np


class TechnicalAnalystAgent:
    def __init__(self, llm: BaseChatModel, period='6mo', interval='1d'):
        self.llm = llm
        self.period = period
        self.interval = interval
        self._last_token_count = 0

    def _download_data(self, ticker: str):
        try:
            df = yf.download(ticker, period=self.period, interval=self.interval)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"[ERROR] Failed to download data for {ticker}: {e}")
            return pd.DataFrame()

    def _calculate_macd(self, close_prices, fast=12, slow=26, signal=9):
        ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
        ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_rsi(self, close_prices, period=14):
        delta = close_prices.diff().squeeze()
        gain = pd.Series(np.where(delta > 0, delta, 0), index=close_prices.index)
        loss = pd.Series(np.where(delta < 0, -delta, 0), index=close_prices.index)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def structured_analyze(self, ticker: str) -> TechnicalAnalysisOutput:
        df = self._download_data(ticker)
        if df.empty or 'Close' not in df:
            raise ValueError(f"[ERROR] No data for ticker {ticker}. Cannot compute technical indicators.")

        macd_line, signal_line, histogram = self._calculate_macd(df['Close'])
        df['MACD'] = macd_line
        df['Signal'] = signal_line
        df['Hist'] = histogram

        rsi_series = self._calculate_rsi(df['Close'])
        df['RSI'] = rsi_series

        macd = round(df['MACD'].iloc[-1], 3)
        signal = round(df['Signal'].iloc[-1], 3)
        hist = round(df['Hist'].iloc[-1], 3)
        rsi = round(df['RSI'].iloc[-1], 2)

        prompt = (
            f"{ticker} indicators:\n"
            f"- MACD: {macd} | Signal: {signal} | Hist: {hist}\n"
            f"- RSI: {rsi}\n\n"
            "Give a short recommendation: Buy, Sell, or Hold? Justify briefly based on trends."
        )

        with get_openai_callback() as cb:
            result = self.llm.invoke(prompt)
            self._last_token_count = cb.total_tokens

        recommendation = result.content.strip() if hasattr(result, 'content') else result

        return TechnicalAnalysisOutput(
            ticker=ticker,
            macd=macd,
            signal=signal,
            histogram=hist,
            rsi=rsi,
            recommendation=recommendation
        )

    def get_last_token_count(self):
        return self._last_token_count

    def analyze(self, ticker: str):
        return self.structured_analyze(ticker)

    def summary(self, ticker: str):
        return self.structured_analyze(ticker).recommendation

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda state: self.__call__(state))

    def __call__(self, state: dict) -> dict:
        ticker = state.get("ticker")
        if not ticker:
            print("[ERROR] Missing ticker in state.")
            return {**state, "technical_analysis": None}
    
        try:
            output = self.structured_analyze(ticker)
            return {**state, "technical_analysis": output}
        except Exception as e:
            print(f"[ERROR] TechnicalAnalystAgent failed for {ticker}: {e}")
            return {**state, "technical_analysis": None}

