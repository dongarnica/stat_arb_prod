#!/usr/bin/env python3
"""
Hourly Statistics Analysis Runner.

Runs fast statistical calculations on pre-identified cointegrated pairs
from the daily analysis.
"""

import sys
import os
import logging
from datetime import datetime, date
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.configuration_manager import ConfigurationManager
from analytics.stats.hourly_statistics_manager import HourlyStatisticsManager


def setup_logging():
    """Setup logging for hourly analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('hourly_statistics_analysis.log')
        ]
    )


def main():
    """Main function for hourly statistics analysis."""
    parser = argparse.ArgumentParser(
        description='Run hourly statistics analysis on cointegrated pairs'
    )
    parser.add_argument(
        '--date',
        help='Date to get cointegrated pairs for (YYYY-MM-DD format)'
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
    parser.add_argument(
        '--show-signals',
        action='store_true',
        help='Show current trading signals'
    )
    parser.add_argument(
        '--min-zscore',
        type=float,
        default=2.0,
        help='Minimum Z-score for signal filtering'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting hourly statistics analysis")
        
        # Parse date if provided
        analysis_date = None
        if args.date:
            try:
                analysis_date = datetime.strptime(args.date, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError(
                    "Invalid date format. Use YYYY-MM-DD format."
                )
        
        # Initialize configuration
        config = ConfigurationManager(config_path=args.config)
        
        # Initialize hourly statistics manager
        hourly_manager = HourlyStatisticsManager(config)
        
        # Create tables if requested
        if args.create_tables:
            logger.info("Creating database tables...")
            hourly_manager.create_tables()
        
        # Run analysis
        start_time = datetime.now()
        
        results = hourly_manager.run_hourly_analysis(analysis_date)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print results summary
        print("\n" + "="*60)
        print("HOURLY STATISTICS ANALYSIS RESULTS")
        print("="*60)
        print(f"Analysis Timestamp: {results['analysis_timestamp']}")
        print(f"Analysis Date: {results.get('analysis_date', 'Today')}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Pairs Analyzed: {results['pairs_analyzed']}")
        print(f"Signals Generated: {results['signals_generated']}")
        print(f"Total Cointegrated Pairs: {results['total_cointegrated_pairs']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print("="*60)
        
        # Show current signals if requested
        if args.show_signals:
            print("\nCURRENT TRADING SIGNALS")
            print("-" * 40)
            
            signals = hourly_manager.get_current_signals(args.min_zscore)
            
            if signals:
                for signal in signals[:10]:  # Show top 10 signals
                    print(
                        f"{signal['symbol1']}-{signal['symbol2']}: "
                        f"Z-Score={signal['current_z_score']:.2f}, "
                        f"Signal={signal['signal_type']}, "
                        f"Strength={signal['signal_strength']:.2f}"
                    )
                if len(signals) > 10:
                    print(f"... and {len(signals) - 10} more signals")
            else:
                print("No current signals found")
            
            print("-" * 40)
        
        logger.info(
            f"Hourly analysis completed successfully in {duration:.2f} seconds"
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Hourly analysis failed: {e}")
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
