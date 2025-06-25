#!/usr/bin/env python3
"""
Simple test to check if ConfigurationManager reads .env correctly.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.configuration_manager import ConfigurationManager


def main():
    """Test configuration reading."""
    
    print("=== Configuration Manager Test ===")
    
    # Initialize configuration
    config = ConfigurationManager()
    
    print(f"Current working directory: {os.getcwd()}")
    
    # Test specific values
    test_vars = [
        'COINTEGRATION_SIGNIFICANCE_LEVEL',
        'COINTEGRATION_MIN_OBSERVATIONS', 
        'COINTEGRATION_LOOKBACK_DAYS',
        'ASSET_SYMBOLS',
        'DB_HOST'
    ]
    
    for var in test_vars:
        raw_value = config.get_str(var, 'NOT_FOUND')
        print(f"  {var} = '{raw_value}'")
    
    print("\n=== Parsed Values ===")
    print(f"  COINTEGRATION_MIN_OBSERVATIONS (int): {config.get_int('COINTEGRATION_MIN_OBSERVATIONS', -1)}")
    print(f"  COINTEGRATION_LOOKBACK_DAYS (int): {config.get_int('COINTEGRATION_LOOKBACK_DAYS', -1)}")
    print(f"  COINTEGRATION_SIGNIFICANCE_LEVEL (float): {config.get_float('COINTEGRATION_SIGNIFICANCE_LEVEL', -1.0)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
