#!/usr/bin/env python3
"""
Example script demonstrating how to use the zscore_breach_counter module
to analyze Z-Score breaches for statistical arbitrage pairs.
"""

import logging
from zscore_breach_counter import run_zscore_breach_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    """Run the Z-Score breach analysis example."""
    logger.info("Starting Z-Score breach analysis...")
    
    # Run the analysis with default parameters:
    # - 365 days of historical data
    # - 252-day rolling window for Z-Score calculation
    results = run_zscore_breach_analysis(lookback_days=365, window=252)
    
    # Display results summary
    if results:
        logger.info(f"Analysis completed for {len(results)} pairs")
        
        # Find pairs with the most breaches
        most_breaches_above = None
        most_breaches_below = None
        max_above = 0
        max_below = 0
        
        for pair, counts in results.items():
            above_count = counts.get('zscore_breach_count_above_2.0', 0)
            below_count = counts.get('zscore_breach_count_below_-2.0', 0)
            
            if above_count > max_above:
                max_above = above_count
                most_breaches_above = pair
                
            if below_count > max_below:
                max_below = below_count
                most_breaches_below = pair
        
        # Display top pairs
        if most_breaches_above:
            logger.info(
                f"Pair with most upper breaches: {most_breaches_above} "
                f"with {max_above} breaches above 2.0"
            )
        
        if most_breaches_below:
            logger.info(
                f"Pair with most lower breaches: {most_breaches_below} "
                f"with {max_below} breaches below -2.0"
            )
    else:
        logger.warning("No results were returned from the analysis")


if __name__ == "__main__":
    main()
