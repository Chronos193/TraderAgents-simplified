from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class RiskSummaryOutput(BaseModel):
    ticker: str = Field(..., description="The stock ticker being evaluated")

    key_risks: List[str] = Field(..., description="Main risk factors identified")

    risk_opportunities: List[str] = Field(..., description="Opportunities with potential reward despite risks")

    volatility_indicators: Optional[Dict[str, Optional[float]]] = None

    financial_flags: Optional[Dict[str, float]] = None

    negative_news_themes: Optional[List[str]] = None

    overall_risk_assessment: str = Field(..., description="Concise summary of overall risk")
