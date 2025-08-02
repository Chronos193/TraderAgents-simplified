#!/usr/bin/env python3
"""
Test script to verify improved decision logic reduces conservative bias.
Run this to test the AAPL case that previously resulted in overly conservative HOLD.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from pipeline_tests.full_pipeline import pipeline, llm

# Load environment
load_dotenv()

def test_improved_decision_logic():
    """Test the improved system on AAPL to see if it's less conservative."""
    
    print("🧪 Testing Improved Decision Logic - AAPL Analysis")
    print("=" * 60)
    
    # Test state - fixed data types to match schema expectations
    test_state = {
        "ticker": "NVDA",
        "portfolio_value": 50000,
        "cash_balance": 15000,
        "holdings": {"MSFT": 50, "GOOGL": 25},
        "sector_exposure": {"Technology": 0.7, "Healthcare": 0.3},
        "daily_returns": [-0.02, 0.01, -0.005, 0.015, 0.008],
        "correlation_with_portfolio": 0.65,  # Changed from string to float
        "volatility": 0.50,  # Changed from "Medium" to numeric (50% annualized)
        "avg_volume": 45000000,  # Changed from "45M" to numeric
        "upcoming_events": ["Earnings in 2 weeks"],  # Changed from string to list
        "sentiment": "positive"  # Changed from "Positive" to lowercase
    }
    
    try:
        print("🚀 Running improved pipeline...")
        result = pipeline.invoke(test_state)
        
        # Extract results
        trade_proposal = result.get("trade_proposal")
        debate_output = result.get("debate_coordinator_output")
        
        print("\n📊 RESULTS:")
        print("-" * 40)
        
        if trade_proposal:
            print(f"🎯 Trade Decision: {trade_proposal.action.upper()}")
            print(f"📈 Ticker: {trade_proposal.ticker}")
            print(f"💰 Quantity: {trade_proposal.quantity}")
            print(f"💵 Price: ${trade_proposal.price}")
            print(f"📝 Reason: {trade_proposal.reason_for_trade}")
        else:
            print("❌ No trade proposal generated")
            
        if debate_output:
            print(f"\n🧠 Risk Management Decision: {debate_output.final_decision.decision.upper()}")
            print(f"🎭 Confidence: {debate_output.final_decision.confidence}")
            print(f"💡 Recommendation: {debate_output.final_decision.recommendation}")
            print(f"📋 Reason: {debate_output.final_decision.reason}")
        else:
            print("\n❌ No risk management output")
            
        # Check if system is still overly conservative
        if trade_proposal and trade_proposal.action == "hold":
            print("\n⚠️  STILL CONSERVATIVE: System recommended HOLD")
            print("   Consider further tuning decision weights")
        elif trade_proposal and trade_proposal.action in ["buy", "sell"]:
            print(f"\n✅ IMPROVED: System recommended {trade_proposal.action.upper()}")
            print("   Decision logic improvements working!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("TraderAgents Decision Logic Improvement Test")
    print("=" * 60)
    
    success = test_improved_decision_logic()
    
    if success:
        print("\n🎉 Test completed! Check results above.")
    else:
        print("\n❌ Test failed. Check configuration and try again.")
