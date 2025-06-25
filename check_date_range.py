#!/usr/bin/env python3
"""
Check the date range and data availability in historical_bars_1_day table.
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
    """Check date range and data availability."""
    
    # Initialize configuration
    config = ConfigurationManager()
    db_config = config.get_database_config()
    
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
            print("=== Date Range Analysis ===")
            
            # Get overall date range
            cursor.execute("""
                SELECT 
                    MIN(bar_date) as min_date, 
                    MAX(bar_date) as max_date,
                    COUNT(DISTINCT bar_date) as unique_dates,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(*) as total_records
                FROM historical_bars_1_day
            """)
            overall_stats = cursor.fetchone()
            
            print(f"Overall Statistics:")
            print(f"  Date range: {overall_stats['min_date']} to {overall_stats['max_date']}")
            print(f"  Unique dates: {overall_stats['unique_dates']}")
            print(f"  Unique symbols: {overall_stats['unique_symbols']}")
            print(f"  Total records: {overall_stats['total_records']}")
            
            # Check recent data availability (last 365 days from today)
            today = date.today()
            one_year_ago = today - timedelta(days=365)
            
            print(f"\n=== Recent Data Analysis (since {one_year_ago}) ===")
            
            cursor.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as record_count,
                    MIN(bar_date) as first_date,
                    MAX(bar_date) as last_date
                FROM historical_bars_1_day 
                WHERE bar_date >= %s
                GROUP BY symbol
                HAVING COUNT(*) >= 252  -- At least 252 trading days
                ORDER BY record_count DESC
            """, (one_year_ago,))
            
            recent_symbols = cursor.fetchall()
            
            print(f"Symbols with >= 252 records in the last year: {len(recent_symbols)}")
            
            if recent_symbols:
                print("\nTop 10 symbols by record count:")
                for symbol_info in recent_symbols[:10]:
                    print(f"  {symbol_info['symbol']}: {symbol_info['record_count']} records "
                          f"({symbol_info['first_date']} to {symbol_info['last_date']})")
            
            # Check data for the specific symbols from .env
            asset_symbols = config.get_str('ASSET_SYMBOLS', '').split(',')
            asset_symbols = [s.strip() for s in asset_symbols if s.strip()]
            
            if asset_symbols:
                print(f"\n=== Analysis for configured symbols: {asset_symbols} ===")
                
                symbol_placeholders = ','.join(['%s'] * len(asset_symbols))
                cursor.execute(f"""
                    SELECT 
                        symbol,
                        COUNT(*) as record_count,
                        MIN(bar_date) as first_date,
                        MAX(bar_date) as last_date
                    FROM historical_bars_1_day 
                    WHERE symbol IN ({symbol_placeholders})
                    AND bar_date >= %s
                    GROUP BY symbol
                    ORDER BY symbol
                """, asset_symbols + [one_year_ago])
                
                configured_symbols = cursor.fetchall()
                
                for symbol_info in configured_symbols:
                    print(f"  {symbol_info['symbol']}: {symbol_info['record_count']} records "
                          f"({symbol_info['first_date']} to {symbol_info['last_date']})")
                
                missing_symbols = set(asset_symbols) - {s['symbol'] for s in configured_symbols}
                if missing_symbols:
                    print(f"  Missing symbols: {missing_symbols}")
                    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        if 'connection' in locals():
            connection.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
