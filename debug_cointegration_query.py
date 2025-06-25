#!/usr/bin/env python3
"""
Debug the cointegration manager query to see why no symbols are found.
"""

import os
import sys
from typing import Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import date, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.configuration_manager import ConfigurationManager


def main():
    """Debug the cointegration query."""
    
    # Initialize configuration
    config = ConfigurationManager()
    db_config = config.get_database_config()
    
    # Get the same configuration that CointegrationManager uses  
    min_observations = config.get_int('COINTEGRATION_MIN_OBSERVATIONS', 252)
    lookback_days = config.get_int('COINTEGRATION_LOOKBACK_DAYS', 365)
    
    print("Configuration values from .env:")
    print(f"  Raw COINTEGRATION_MIN_OBSERVATIONS: {config.get_str('COINTEGRATION_MIN_OBSERVATIONS', 'NOT_SET')}")
    print(f"  Raw COINTEGRATION_LOOKBACK_DAYS: {config.get_str('COINTEGRATION_LOOKBACK_DAYS', 'NOT_SET')}")
    print(f"  Parsed COINTEGRATION_MIN_OBSERVATIONS: {min_observations}")
    print(f"  Parsed COINTEGRATION_LOOKBACK_DAYS: {lookback_days}")
    
    # Calculate the minimum date like CointegrationManager does
    min_date = date.today() - timedelta(days=lookback_days)
    print(f"  Calculated min_date: {min_date}")
    print(f"  Today: {date.today()}")
    
    try:
        # Connect to database
        connection = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password'],
            sslmode=db_config.get('sslmode', 'prefer')
        )
        
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            print(f"\n=== Executing CointegrationManager Query ===")
            
            # Execute the exact same query that CointegrationManager uses
            query = """
                SELECT DISTINCT symbol
                FROM historical_bars_1_day
                WHERE bar_date >= %s
                GROUP BY symbol
                HAVING COUNT(*) >= %s
                ORDER BY symbol
            """
            
            print(f"Query: {query}")
            print(f"Parameters: min_date={min_date}, min_observations={min_observations}")
            
            cursor.execute(query, (min_date, min_observations))
            results = cursor.fetchall()
            
            print(f"Results: {len(results)} symbols found")
            
            if results:
                symbols = [row['symbol'] for row in results]
                print(f"Symbols: {symbols}")
            else:
                print("No symbols found. Let's debug...")
                
                # Check what symbols have data in the date range
                cursor.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as record_count,
                        MIN(bar_date) as first_date,
                        MAX(bar_date) as last_date
                    FROM historical_bars_1_day
                    WHERE bar_date >= %s
                    GROUP BY symbol
                    ORDER BY record_count DESC
                """, (min_date,))
                
                debug_results = cursor.fetchall()
                print(f"\nAll symbols with data since {min_date}:")
                for row in debug_results:
                    meets_criteria = row['record_count'] >= min_observations
                    status = "✓" if meets_criteria else "✗"
                    print(f"  {status} {row['symbol']}: {row['record_count']} records "
                          f"({row['first_date']} to {row['last_date']})")
                    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        if 'connection' in locals():
            connection.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
