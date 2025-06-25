#!/usr/bin/env python3
"""
Check hourly_statistics table structure.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.database_manager import DatabaseManager
from analytics.stats.configuration_manager import ConfigurationManager

def check_table_structure():
    """Check the structure of hourly_statistics table."""
    try:
        config = ConfigurationManager()
        db = DatabaseManager(config)
        connection = db.get_connection()
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'hourly_statistics' 
                ORDER BY ordinal_position
            """)
            results = cursor.fetchall()
            
            print("Current hourly_statistics table structure:")
            print("-" * 60)
            print(f"{'Column Name':<25} {'Data Type':<20} {'Nullable'}")
            print("-" * 60)
            
            for row in results:
                print(f"{row[0]:<25} {row[1]:<20} {row[2]}")
        
        db.return_connection(connection)
        
    except Exception as e:
        print(f"Error checking table structure: {e}")

if __name__ == "__main__":
    check_table_structure()
