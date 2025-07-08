from pydantic import BaseModel
from typing import List

# Schema for Bullish Thesis Output
class BullishThesisOutput(BaseModel):
    ticker: str
    thesis: str
    supporting_points: List[str]
    confidence: float  # 0.0 to 1.0