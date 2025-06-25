#!/usr/bin/env python3
"""
Test the new advanced metrics functionality.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.metrics import compute_hourly_statistics


def test_metrics():
    """Test the new metrics calculation."""
    print("Testing Advanced Metrics Calculation")
    print("=" * 50)
    
    # Create sample spread data
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=200, freq='H')
    
    # Generate mean-reverting spread series
    spread_values = []
    current_value = 0.0
    for i in range(200):
        # AR(1) process with mean reversion
        current_value = 0.95 * current_value + np.random.normal(0, 0.1)
        spread_values.append(current_value)
    
    spread = pd.Series(spread_values, index=dates)
    
    # Calculate Z-scores
    window = 120
    spread_window = spread.tail(window)
    spread_mean = spread_window.mean()
    spread_std = spread_window.std()
    zscore = (spread_window - spread_mean) / spread_std
    
    print(f"Sample spread series: {len(spread)} observations")
    print(f"Z-score window: {len(zscore)} observations")
    print(f"Spread mean: {spread_mean:.4f}")
    print(f"Spread std: {spread_std:.4f}")
    print("")
    
    try:
        # Test the metrics calculation
        metrics = compute_hourly_statistics(spread_window, zscore)
        
        print("Advanced Metrics Results:")
        print("-" * 30)
        print(f"Half-life: {metrics['half_life']:.2f}")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Z-scores > +2: {metrics['zscore_over_2']}")
        print(f"Z-scores < -2: {metrics['zscore_under_minus_2']}")
        print("")
        
        # Verify Z-score counts manually
        manual_over_2 = (zscore > 2).sum()
        manual_under_minus_2 = (zscore < -2).sum()
        
        print("Manual verification:")
        print(f"Z-scores > +2 (manual): {manual_over_2}")
        print(f"Z-scores < -2 (manual): {manual_under_minus_2}")
        
        if (metrics['zscore_over_2'] == manual_over_2 and 
            metrics['zscore_under_minus_2'] == manual_under_minus_2):
            print("✅ Z-score counts match!")
        else:
            print("❌ Z-score counts don't match!")
        
        print("")
        print("✅ Metrics calculation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")


def test_edge_cases():
    """Test edge cases for metrics calculation."""
    print("\nTesting Edge Cases")
    print("=" * 30)
    
    # Test with minimal data
    try:
        spread_small = pd.Series([1.0, 1.1], index=pd.date_range('2025-01-01', periods=2, freq='H'))
        zscore_small = pd.Series([0.0, 1.0], index=pd.date_range('2025-01-01', periods=2, freq='H'))
        
        metrics = compute_hourly_statistics(spread_small, zscore_small)
        print("✅ Minimal data test passed")
        print(f"  Half-life: {metrics['half_life']}")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        
    except Exception as e:
        print(f"Minimal data test: {e}")
    
    # Test with constant spread (zero volatility)
    try:
        spread_const = pd.Series([1.0] * 50, index=pd.date_range('2025-01-01', periods=50, freq='H'))
        zscore_const = pd.Series([0.0] * 50, index=pd.date_range('2025-01-01', periods=50, freq='H'))
        
        metrics = compute_hourly_statistics(spread_const, zscore_const)
        print("✅ Constant spread test passed")
        print(f"  Half-life: {metrics['half_life']}")
        print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        
    except Exception as e:
        print(f"Constant spread test: {e}")


if __name__ == "__main__":
    test_metrics()
    test_edge_cases()
