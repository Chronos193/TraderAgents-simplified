from typing import List, Dict, Literal, TypedDict, Optional


class TradeProposal(TypedDict):
    # --- Trade details ---
    trade_id: str
    ticker: str                    # e.g. "NVDA"
    action: Literal["buy", "sell", "hold"]
    quantity: float                # number of shares
    price: float                   # expected price per share
    estimated_cost: float          # total cost = quantity * price

    # --- Portfolio snapshot ---
    portfolio_value: float         # total portfolio value in $
    cash_balance: float            # available cash
    holdings: Dict[str, float]     # {ticker: position_size}, in dollars
    sector_exposure: Dict[str, float]  # e.g., {"tech": 0.7, "energy": 0.2}
    daily_returns: List[float]     # past N-day returns for portfolio
    unrealized_pnl: float          # optional

    # --- Trade impact simulation ---
    new_position_size: Optional[float]  # post-trade position size in portfolio %
    new_cash_balance: Optional[float]
    simulated_drawdown: Optional[float]

    # --- Market context (optional) ---
    volatility: Optional[float]        # annualized or daily
    avg_volume: Optional[float]
    correlation_with_portfolio: Optional[float]
    upcoming_events: Optional[List[str]]
    sentiment: Optional[Literal["positive", "neutral", "negative"]]

    # --- Meta ---
    reason_for_trade: Optional[str]    # from TraderAgent
    timestamp: Optional[str]
