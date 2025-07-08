from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
import os
import requests
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
import numpy as np
from src.agents.analyst_team import Analyst
from src.schemas.analyst_schemas import TechnicalAnalysisOutput
# Technical Analyst
class TechnicalAnalystAgent(Analyst):
    def __init__(self, ticker, llm, period='6mo', interval='1d'):
        super().__init__(ticker, llm)
        self.period = period
        self.interval = interval
        self.data = self._download_data()
        self.indicators = {}

    def _download_data(self):
        df = yf.download(self.ticker, period=self.period, interval=self.interval)
        df.dropna(inplace=True)
        return df

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

    def structured_analyze(self) -> TechnicalAnalysisOutput:
        df = self.data.copy()

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
            f"{self.ticker} indicators:\n"
            f"- MACD: {macd} | Signal: {signal} | Hist: {hist}\n"
            f"- RSI: {rsi}\n\n"
            "Give a short recommendation: Buy, Sell, or Hold? Justify briefly based on trends."
        )

        with get_openai_callback() as cb:
            result = self.llm.invoke(prompt)
            self._last_token_count = cb.total_tokens

        recommendation = result.content.strip() if hasattr(result, 'content') else result

        return TechnicalAnalysisOutput(
            ticker=self.ticker,
            macd=macd,
            signal=signal,
            histogram=hist,
            rsi=rsi,
            recommendation=recommendation
        )

    def get_last_token_count(self):
        return self._last_token_count

    def analyze(self):
        return self.structured_analyze()

    def summary(self):
        return self.structured_analyze().recommendation

    def as_runnable_node(self) -> RunnableLambda:
        return RunnableLambda(lambda _: self.structured_analyze())