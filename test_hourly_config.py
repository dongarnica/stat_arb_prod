#!/usr/bin/env python3
"""
Test script to verify configuration loading and hourly analysis with fresh process.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analytics.stats.configuration_manager import ConfigurationManager
from analytics.stats.hourly_statistics_manager import HourlyStatisticsManager
from analytics.stats.cointegration_manager import CointegrationManager
from datetime import date

def main():
    print("=== Testing Fresh Configuration Load ===")
    
    # Create fresh configuration
    config = ConfigurationManager()
    
    print("\nConfiguration values:")
    print(f"MA_LONG_WINDOW: {config.get_int('MA_LONG_WINDOW', 50)}")
    print(f"MA_SHORT_WINDOW: {config.get_int('MA_SHORT_WINDOW', 20)}")
    print(f"ZSCORE_WINDOW: {config.get_int('ZSCORE_WINDOW', 252)}")
    print(f"HOURLY_LOOKBACK_HOURS: {config.get_int('HOURLY_LOOKBACK_HOURS', 24)}")
    
    # Test hourly manager
    print("\n=== Testing Hourly Analysis ===")
    hm = HourlyStatisticsManager(config)
    print(f"HourlyManager ma_long_window: {hm.ma_long_window}")
    print(f"HourlyManager lookback_hours: {hm.lookback_hours}")
    
    # Test with cointegrated pairs
    cm = CointegrationManager(config)
    pairs = cm.get_cointegrated_pairs(date.today())
    print(f"\nFound {len(pairs)} cointegrated pairs")
    
    if pairs:
        # Test first pair
        first_pair = pairs[0]
        print(f"Testing pair: {first_pair['symbol1']} - {first_pair['symbol2']}")
        
        symbols = [first_pair['symbol1'], first_pair['symbol2']]
        price_data = hm._fetch_recent_prices(symbols)
        print(f"Price data shape: {price_data.shape}")
        print(f"Required minimum periods: {hm.ma_long_window}")
        
        if len(price_data) >= hm.ma_long_window:
            result = hm._analyze_pair_hourly(first_pair, price_data)
            print(f"Analysis result: {result is not None}")
            if result:
                print(f"Has signal: {result.get('has_signal', False)}")
                print(f"Z-score: {result.get('current_z_score', 'N/A')}")
        else:
            print(f"Insufficient data: need {hm.ma_long_window}, have {len(price_data)}")

if __name__ == "__main__":
    main()
