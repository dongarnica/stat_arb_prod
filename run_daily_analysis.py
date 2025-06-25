#!/usr/bin/env python3
"""
Daily Cointegration Analysis Runner.

Runs comprehensive cointegration tests on all available symbol pairs
and stores the results for use in hourly analysis.
"""

import sys
import os
import logging
from datetime import datetime
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.configuration_manager import ConfigurationManager
from analytics.stats.cointegration_manager import CointegrationManager


def setup_logging():
    """Setup logging for daily analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('daily_cointegration_analysis.log')
        ]
    )


def main():
    """Main function for daily cointegration analysis."""
    parser = argparse.ArgumentParser(
        description='Run daily cointegration analysis'
    )
    parser.add_argument(
        '--symbols',
        nargs='*',
        help='Specific symbols to analyze (optional)'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration file (optional)'
    )
    parser.add_argument(
        '--create-tables',
        action='store_true',
        help='Create database tables if they don\'t exist'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting daily cointegration analysis")
        
        # Initialize configuration
        config = ConfigurationManager(config_path=args.config)
        
        # Initialize cointegration manager
        cointegration_manager = CointegrationManager(config)
        
        # Create tables if requested
        if args.create_tables:
            logger.info("Creating database tables...")
            cointegration_manager.create_tables()
        
        # Run analysis
        start_time = datetime.now()
        
        results = cointegration_manager.run_daily_cointegration_analysis(
            symbols=args.symbols
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print results summary
        print("\n" + "="*60)
        print("DAILY COINTEGRATION ANALYSIS RESULTS")
        print("="*60)
        print(f"Analysis Date: {results['analysis_date']}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Symbols Analyzed: {results['symbols_analyzed']}")
        print(f"Pairs Tested: {results['pairs_tested']}")
        print(f"Cointegrated Pairs Found: {results['cointegrated_pairs']}")
        print(f"Cointegration Rate: {results['cointegration_rate']:.2%}")
        print(f"Test Method: {results['test_method']}")
        print(f"Significance Level: {results['significance_level']}")
        print("="*60)
        
        logger.info(
            f"Daily analysis completed successfully in {duration:.2f} seconds"
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Daily analysis failed: {e}")
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
