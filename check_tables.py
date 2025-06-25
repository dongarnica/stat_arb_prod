#!/usr/bin/env python3
"""Check database tables."""

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
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            print('Available tables:')
            for table in tables:
                print(f'  - {table[0]}')
                
            # Check if price_data table exists and sample its structure
            if any('price_data' in table for table in tables):
                print('\nPrice data table exists')
            else:
                print('\nPrice data table does NOT exist')
                print('Looking for similar tables...')
                for table in tables:
                    if 'price' in table[0].lower():
                        print(f'  Found: {table[0]}')
                    elif 'data' in table[0].lower():
                        print(f'  Found: {table[0]}')
                        
    finally:
        db_manager.return_connection(conn)

if __name__ == "__main__":
    main()
