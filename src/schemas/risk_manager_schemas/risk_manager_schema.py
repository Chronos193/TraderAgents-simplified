# src/schemas/risk_manager_schemas/risk_manager_schema.py
from pydantic import BaseModel, field_validator
from typing import List, Literal


class AnalystResponse(BaseModel):
    role: Literal["aggressive", "conservative", "neutral"]
    final_argument: str

    @field_validator("role", mode="before")
    @classmethod
    def lowercase_role(cls, v):
        return v.lower()


class FinalDecision(BaseModel):
    decision: Literal["accept", "reject", "revise"]
    reason: str
    recommendation: str
    confidence: Literal["high", "medium", "low"]
    notes: str


class DebateCoordinatorOutput(BaseModel):
    analyst_responses: List[AnalystResponse]
    rounds_transcript: List[str]
    final_decision: FinalDecision

