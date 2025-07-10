from typing import List, Literal, Optional, TypedDict

# ---------------------
# Output Schema
# ---------------------
class AnalystResponse(TypedDict):
    role: Literal["aggressive", "conservative", "neutral"]
    final_argument: str

class FinalDecision(TypedDict):
    decision: Literal["accept", "reject", "revise"]
    reason: str
    recommendation: Optional[str]
    confidence: Literal["high", "medium", "low"]
    notes: Optional[str]

class DebateCoordinatorOutput(TypedDict):
    analyst_responses: List[AnalystResponse]
    rounds_transcript: List[str]
    final_decision: FinalDecision