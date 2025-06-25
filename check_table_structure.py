#!/usr/bin/env python3
"""Check database table structure."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.configuration_manager import ConfigurationManager
from analytics.stats.database_manager import DatabaseManager

def main():
    config = ConfigurationManager()
    db_manager = DatabaseManager(config)
    conn = db_manager.get_connection()

    try:
        with conn.cursor() as cursor:
            # Check structure of historical_bars_1_day table
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'historical_bars_1_day'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            print('historical_bars_1_day table structure:')
            for col in columns:
                print(f'  - {col[0]} ({col[1]})')
                
            # Check sample data
            cursor.execute("""
                SELECT DISTINCT symbol 
                FROM historical_bars_1_day 
                ORDER BY symbol 
                LIMIT 10;
            """)
            symbols = cursor.fetchall()
            print(f'\nSample symbols (first 10):')
            for symbol in symbols:
                print(f'  - {symbol[0]}')
                
            # Check date range
            cursor.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as total_rows
                FROM historical_bars_1_day;
            """)
            date_info = cursor.fetchone()
            print(f'\nData range: {date_info[0]} to {date_info[1]} ({date_info[2]} total rows)')
                        
    finally:
        db_manager.return_connection(conn)

if __name__ == "__main__":
    main()
