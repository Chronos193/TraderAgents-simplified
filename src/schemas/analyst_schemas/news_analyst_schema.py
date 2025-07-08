# src/schemas/analyst_schemas.py

from pydantic import BaseModel
from typing import List

class NewsAnalysisOutput(BaseModel):
    ticker: str
    headlines: List[str]
    summary_text: str
    themes_text: str
