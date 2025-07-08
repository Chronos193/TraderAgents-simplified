# src/agents/analyst_team/schemas.py
from pydantic import BaseModel
from typing import Dict

class FundamentalsAnalysisOutput(BaseModel):
    ticker: str
    financials_summary: Dict[str, float]
    balance_summary: Dict[str, float]
    cashflow_summary: Dict[str, float]
    analysis_text: str  # This is the LLM’s final textual output
