#!/usr/bin/env python3
"""
Statistical Arbitrage Main Application.

Main entry point for the statistical arbitrage system that can run:
- The full scheduler (daily + hourly)
- Individual daily analysis
- Individual hourly analysis
- System status checks
"""

import sys
import os
import logging
import argparse
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.configuration_manager import ConfigurationManager
from analytics.stats.scheduler import StatisticalArbitrageScheduler


def setup_logging(log_level='INFO'):
    """Setup comprehensive logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('statistical_arbitrage_main.log')
        ]
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Statistical Arbitrage Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scheduler              # Run the full scheduler
  %(prog)s daily                 # Run daily analysis once
  %(prog)s hourly                # Run hourly analysis once
  %(prog)s status                # Show system status
  %(prog)s daily --symbols SPY IVV  # Analyze specific symbols
        """
    )
    
    parser.add_argument(
        'command',
        choices=['scheduler', 'daily', 'hourly', 'status'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='*',
        help='Specific symbols to analyze (for daily command)'
    )
    
    parser.add_argument(
        '--date',
        help='Date for analysis (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--create-tables',
        action='store_true',
        help='Create database tables if they don\'t exist'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting statistical arbitrage system: {args.command}")
        
        # Initialize configuration
        config = ConfigurationManager(config_path=args.config)
        
        # Initialize scheduler
        scheduler = StatisticalArbitrageScheduler(config)
        
        # Create tables if requested
        if args.create_tables:
            logger.info("Creating database tables...")
            scheduler._initialize_tables()
        
        # Execute command
        if args.command == 'scheduler':
            print("Starting Statistical Arbitrage Scheduler...")
            print("Press Ctrl+C to stop")
            scheduler.setup_schedules()
            scheduler.run_scheduler()
            
        elif args.command == 'daily':
            print("Running daily cointegration analysis...")
            
            # Parse date if provided
            analysis_date = None
            if args.date:
                try:
                    analysis_date = datetime.strptime(args.date, '%Y-%m-%d').date()
                except ValueError:
                    raise ValueError("Invalid date format. Use YYYY-MM-DD.")
            
            results = scheduler.run_daily_analysis_now(args.symbols)
            
            print("\nDaily Analysis Results:")
            print("-" * 40)
            for key, value in results.items():
                print(f"{key}: {value}")
            
        elif args.command == 'hourly':
            print("Running hourly statistics analysis...")
            
            # Parse date if provided
            analysis_date = None
            if args.date:
                try:
                    analysis_date = datetime.strptime(args.date, '%Y-%m-%d').date()
                except ValueError:
                    raise ValueError("Invalid date format. Use YYYY-MM-DD.")
            
            results = scheduler.run_hourly_analysis_now(analysis_date)
            
            print("\nHourly Analysis Results:")
            print("-" * 40)
            for key, value in results.items():
                print(f"{key}: {value}")
            
            # Show recent signals
            signals = scheduler.hourly_manager.get_current_signals()
            if signals:
                print(f"\nCurrent Trading Signals ({len(signals)}):")
                print("-" * 40)
                for signal in signals[:5]:  # Show top 5
                    print(
                        f"{signal['symbol1']}-{signal['symbol2']}: "
                        f"Z={signal['current_z_score']:.2f}, "
                        f"{signal['signal_type']}"
                    )
                if len(signals) > 5:
                    print(f"... and {len(signals) - 5} more")
            
        elif args.command == 'status':
            print("System Status:")
            print("=" * 50)
            
            status = scheduler.get_system_status()
            
            # Format and display status
            print(f"System Time: {status.get('system_time', 'Unknown')}")
            print(f"Scheduler Running: {status.get('scheduler_running', False)}")
            print(f"Cointegrated Pairs Today: {status.get('cointegrated_pairs_today', 0)}")
            print(f"Current Signals: {status.get('current_signals', 0)}")
            print(f"Recent Events (24h): {status.get('recent_events', 0)}")
            
            config_info = status.get('configuration', {})
            if config_info:
                print("\nConfiguration:")
                print(f"  Daily Analysis Time: {config_info.get('daily_time', 'Unknown')}")
                print(f"  Hourly Analysis: {config_info.get('hourly_enabled', False)}")
                print(f"  Trading Hours: {config_info.get('trading_hours', [])}")
            
            next_daily = status.get('next_daily_analysis')
            next_hourly = status.get('next_hourly_analysis')
            
            if next_daily:
                print(f"Next Daily Analysis: {next_daily}")
            if next_hourly:
                print(f"Next Hourly Analysis: {next_hourly}")
        
        logger.info(f"Command '{args.command}' completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 0
        
    except Exception as e:
        logger.error(f"Command '{args.command}' failed: {e}")
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
