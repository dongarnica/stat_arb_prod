#!/usr/bin/env python3
"""
Setup script for Statistical Arbitrage System.

Initializes the database and creates necessary tables.
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.configuration_manager import ConfigurationManager
from analytics.stats.cointegration_manager import CointegrationManager
from analytics.stats.hourly_statistics_manager import HourlyStatisticsManager


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    """Setup the statistical arbitrage system."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        print("Setting up Statistical Arbitrage System...")
        print("=" * 50)
        
        # Initialize configuration
        logger.info("Loading configuration...")
        config = ConfigurationManager()
        
        # Test database connection
        logger.info("Testing database connection...")
        
        # Initialize managers
        cointegration_manager = CointegrationManager(config)
        hourly_manager = HourlyStatisticsManager(config)
        
        # Create tables
        logger.info("Creating database tables...")
        
        print("Creating cointegration tables...")
        cointegration_manager.create_tables()
        
        print("Creating hourly statistics tables...")
        hourly_manager.create_tables()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Edit .env file with your database credentials")
        print("2. Run daily analysis: python main.py daily")
        print("3. Run hourly analysis: python main.py hourly")
        print("4. Start scheduler: python main.py scheduler")
        print("5. Check status: python main.py status")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check database connection settings in .env")
        print("2. Ensure database server is running")
        print("3. Verify database exists and user has permissions")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
