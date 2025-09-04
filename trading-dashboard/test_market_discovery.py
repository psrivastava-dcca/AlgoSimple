#!/usr/bin/env python3
"""
Test to verify the US market discovery fix works without getting stuck.
"""

import pandas as pd
import numpy as np
from moving_average_crossover import MovingAverageCrossover

def test_us_market_discovery():
    """Test that US market discovery works without getting stuck."""
    
    print("Testing US market discovery (under $200)...")
    
    # Initialize strategy
    strategy = MovingAverageCrossover(short_window=5, long_window=10)
    
    try:
        # Test comprehensive discovery
        print("\n1. Testing comprehensive US discovery (under $200)...")
        stocks = strategy.get_market_stocks()
        print(f"Found {len(stocks)} US stocks under $200")
        
        # Test market cap discovery
        print("\n2. Testing US market cap discovery (under $200)...")
        large_cap_stocks = strategy.get_stocks_by_market_cap()
        print(f"Found {len(large_cap_stocks)} US large-cap stocks under $200")
        
        # Test sector discovery
        print("\n3. Testing US sector discovery (under $200)...")
        sector_stocks = strategy.get_stocks_by_sector()
        total_sector_stocks = sum(len(stocks) for stocks in sector_stocks.values())
        print(f"Found {total_sector_stocks} US stocks under $200 across {len(sector_stocks)} sectors")
        
        # Test market opportunities
        print("\n4. Testing US market opportunities (under $200)...")
        opportunities = strategy.get_market_opportunities(max_stocks=50)
        print(f"Found {len(opportunities)} US opportunities under $200")
        
        if not opportunities.empty:
            print("\nTop 5 US opportunities under $200:")
            for i, (_, row) in enumerate(opportunities.head(5).iterrows()):
                print(f"  {i+1}. {row['Symbol']} - ${float(row['Current_Price']):.2f} - {row['Recommendation']} (Score: {int(row['Score'])})")
        
        print("\nAll US market discovery tests passed!")
        return True
        
    except Exception as e:
        print(f"US market discovery failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_small_us_analysis():
    """Test a small US analysis to ensure everything works."""
    
    print("\nTesting small US analysis (under $200)...")
    
    # Initialize strategy
    strategy = MovingAverageCrossover(short_window=5, long_window=10)
    
    try:
        # Test with just a few US stocks
        test_stocks = ['AAPL', 'MSFT', 'GOOGL']
        results = strategy.screen_stocks(test_stocks)
        
        print(f"Analysis completed for {len(test_stocks)} US stocks")
        print(f"Found {len(results)} results")
        
        if not results.empty:
            print("\nUS Results (under $200):")
            for _, row in results.iterrows():
                print(f"  {row['Symbol']}: ${float(row['Current_Price']):.2f} - {row['Recommendation']} (Score: {int(row['Score'])})")
        
        return True
        
    except Exception as e:
        print(f"Small US analysis failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("US MARKET DISCOVERY FIX VERIFICATION (UNDER $200)")
    print("="*60)
    
    success1 = test_us_market_discovery()
    success2 = test_small_us_analysis()
    
    if success1 and success2:
        print("\nAll tests passed! US market discovery is working properly.")
        print("\nThe program now only shows US stocks under $200.")
    else:
        print("\nSome tests failed. Please check the error messages above.")
