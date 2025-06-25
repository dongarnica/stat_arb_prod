#!/usr/bin/env python3
"""
Update hourly_statistics table with new metrics columns.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.hourly_statistics_manager import HourlyStatisticsManager
from analytics.stats.configuration_manager import ConfigurationManager

def update_table():
    """Update the hourly_statistics table with new columns."""
    try:
        config = ConfigurationManager()
        manager = HourlyStatisticsManager(config)
        
        print("Updating hourly_statistics table with new metrics columns...")
        manager.create_tables()
        print("✅ Table updated successfully!")
        
        # Verify the new columns exist
        print("\nVerifying new columns were added...")
        connection = manager.db_manager.get_connection()
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'hourly_statistics' 
                AND column_name IN ('half_life', 'sharpe_ratio', 'zscore_over_2', 'zscore_under_minus_2')
                ORDER BY column_name
            """)
            results = cursor.fetchall()
            
            if results:
                print("New columns found:")
                print("-" * 50)
                for row in results:
                    print(f"  {row[0]:<20} {row[1]:<20} {row[2]}")
                print("✅ All new columns added successfully!")
            else:
                print("❌ New columns not found!")
        
        manager.db_manager.return_connection(connection)
        
    except Exception as e:
        print(f"❌ Error updating table: {e}")

if __name__ == "__main__":
    update_table()
