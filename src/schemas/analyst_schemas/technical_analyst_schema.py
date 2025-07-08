from pydantic import BaseModel
from typing import List

# Schema
class TechnicalAnalysisOutput(BaseModel):
    ticker: str
    macd: float
    signal: float
    histogram: float
    rsi: float
    recommendation: str
