from typing import List
from pydantic import BaseModel

class BearishThesisOutput(BaseModel):
    ticker: str
    thesis: str
    supporting_points: List[str]
    confidence: float
