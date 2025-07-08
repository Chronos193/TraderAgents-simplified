from typing import List, Literal
from pydantic import BaseModel

class DebateTurn(BaseModel):
    speaker: Literal["Bullish", "Bearish"]
    message: str
    tokens_used: int

class DebateResult(BaseModel):
    turns: List[DebateTurn]
    summary: str
    winner: Literal["Bullish", "Bearish", "Tie"]
    total_tokens: int
