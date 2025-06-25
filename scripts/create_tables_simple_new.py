#!/usr/bin/env python3
"""
DEPRECATED: This script used the old statistics_service system.
Use the modern analytics.stats system instead.

For database operations, use:
- analytics.stats.DatabaseManager
- analytics.stats.ConfigurationManager
"""

import sys
from pathlib import Path

# Add analytics to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'analytics'))

from analytics.stats import DatabaseManager, ConfigurationManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Use the modern analytics.stats system for database operations."""
    logger.info('This script is deprecated.')
    logger.info('Use analytics.stats.DatabaseManager for database operations.')
    logger.info('See analytics/example_usage.py for modern usage examples.')
    
    try:
        config = ConfigurationManager()
        db_manager = DatabaseManager(config)
        logger.info('✅ Modern analytics.stats system is working correctly!')
    except Exception as e:
        logger.error(f'❌ Error with modern system: {e}')


if __name__ == '__main__':
    main()
