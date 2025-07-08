from pydantic import BaseModel
from typing import List

class SentimentAnalysisOutput(BaseModel):
    ticker: str
    headlines: List[str]
    summary_text: str
    sentiment_text: str
