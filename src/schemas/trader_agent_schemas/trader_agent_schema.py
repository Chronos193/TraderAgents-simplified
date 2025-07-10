from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional


class TradeProposal(BaseModel):
    # --- Trade details ---
    trade_id: str
    ticker: str                                     # e.g., "NVDA"
    action: Literal["buy", "sell", "hold"]
    quantity: float
    price: float
    estimated_cost: float                           # total cost = quantity * price

    # --- Portfolio snapshot ---
    portfolio_value: float                          # total portfolio value in $
    cash_balance: float                             # available cash
    holdings: Dict[str, float]                      # e.g., {"NVDA": 10000.0}
    sector_exposure: Dict[str, float]               # e.g., {"tech": 0.7, "energy": 0.2}
    daily_returns: List[float]                      # past N-day returns for portfolio
    unrealized_pnl: Optional[float] = None

    # --- Trade impact simulation ---
    new_position_size: Optional[float] = None       # post-trade position size in portfolio %
    new_cash_balance: Optional[float] = None
    simulated_drawdown: Optional[float] = None

    # --- Market context (optional) ---
    volatility: Optional[float] = None              # annualized or daily
    avg_volume: Optional[float] = None
    correlation_with_portfolio: Optional[float] = None
    upcoming_events: Optional[List[str]] = None
    sentiment: Optional[Literal["positive", "neutral", "negative"]] = None

    # --- Meta ---
    reason_for_trade: Optional[str] = None          # from TraderAgent
    timestamp: Optional[str] = None
