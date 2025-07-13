from typing import Optional, TypedDict
from src.schemas.analyst_schemas import (
    FundamentalsAnalysisOutput,
    TechnicalAnalysisOutput,
    NewsAnalysisOutput,
    SentimentAnalysisOutput,
)
from src.schemas.researcher_schemas import (
    BearishThesisOutput,
    BullishThesisOutput,
    DebateResult,
)
from src.schemas.risk_manager_schemas.risk_manager_schema import (
    AnalystResponse,
    DebateCoordinatorOutput,
)
from src.schemas.risk_manager_schemas.summarizer_schema import RiskSummaryOutput
from src.schemas.trader_agent_schemas import TradeProposal


class BaseTradingState(TypedDict):
    """Fields that are required for the pipeline to run at all."""
    ticker: str


class TradingState(BaseTradingState, total=False):
    """Optional intermediate outputs from each agent in the pipeline."""
    
    # Analyst team outputs
    fundamentals_analysis: Optional[FundamentalsAnalysisOutput]
    technical_analysis: Optional[TechnicalAnalysisOutput]
    news_analysis: Optional[NewsAnalysisOutput]
    sentiment_analysis: Optional[SentimentAnalysisOutput]

    # Researcher team outputs
    bullish_thesis: Optional[BullishThesisOutput]
    bearish_thesis: Optional[BearishThesisOutput]
    research_debate_result: Optional[DebateResult]

    # Trader decision
    trade_proposal: Optional[TradeProposal]

    # Risk team outputs
    risk_summary: Optional[RiskSummaryOutput]
    aggressive_response: Optional[AnalystResponse]
    conservative_response: Optional[AnalystResponse]
    neutral_response: Optional[AnalystResponse]
    risk_debate_result: Optional[DebateCoordinatorOutput]
