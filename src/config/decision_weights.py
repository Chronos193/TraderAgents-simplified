# src/config/decision_weights.py
"""
Decision weighting configuration to reduce conservative bias in TraderAgents system.
"""

# Factor weights for decision making (higher = more important)
DECISION_WEIGHTS = {
    # Recent data carries more weight than historical
    "recent_earnings": 2.0,
    "recent_news": 1.8,
    "technical_momentum": 1.5,
    
    # Forward-looking factors
    "growth_catalysts": 2.2,
    "earnings_revisions": 1.9,
    "analyst_upgrades": 1.6,
    
    # Risk factors (still important but not overwhelming)
    "valuation_concerns": 1.0,
    "market_volatility": 0.8,
    "sector_rotation": 0.7,
    
    # Generic concerns get lower weight
    "general_market_risk": 0.5,
    "macro_uncertainty": 0.6,
}

# Tie-breaking preferences
TIE_BREAKING_RULES = {
    "favor_action_over_inaction": True,
    "recent_data_wins_ties": True,
    "technical_momentum_threshold": 0.6,  # RSI < 40 or > 60 breaks ties
    "earnings_surprise_threshold": 0.05,  # 5% earnings beat breaks ties toward bullish
}

# Conservative bias reduction settings
BIAS_REDUCTION = {
    "aggressive_weight": 1.2,     # Give aggressive agent slightly more weight
    "conservative_weight": 0.8,   # Reduce conservative agent weight
    "neutral_weight": 1.0,        # Keep neutral at baseline
    
    # Decision thresholds
    "accept_threshold": 0.6,      # Lower threshold for accepting trades
    "reject_threshold": 0.8,      # Higher threshold for rejecting trades
}

# Context-specific overrides
CONTEXT_OVERRIDES = {
    "oversold_technicals": {
        "condition": "RSI < 35",
        "action_bias": "bullish",
        "weight_multiplier": 1.5
    },
    "strong_earnings_beat": {
        "condition": "earnings_surprise > 10%",
        "action_bias": "bullish", 
        "weight_multiplier": 1.8
    },
    "recent_insider_buying": {
        "condition": "insider_net_buying > 0",
        "action_bias": "bullish",
        "weight_multiplier": 1.3
    }
}
