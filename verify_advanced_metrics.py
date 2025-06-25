#!/usr/bin/env python3
"""
Verify that advanced metrics are being stored in the database.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics.stats.database_manager import DatabaseManager
from analytics.stats.configuration_manager import ConfigurationManager

def verify_advanced_metrics():
    """Verify that advanced metrics are being stored."""
    try:
        config = ConfigurationManager()
        db = DatabaseManager(config)
        connection = db.get_connection()
        
        print("Checking the most recent hourly statistics with advanced metrics...")
        print("=" * 80)
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    symbol1,
                    symbol2,
                    current_z_score,
                    half_life,
                    sharpe_ratio,
                    zscore_over_2,
                    zscore_under_minus_2,
                    analysis_timestamp
                FROM hourly_statistics 
                WHERE analysis_timestamp >= CURRENT_DATE
                AND half_life IS NOT NULL
                ORDER BY analysis_timestamp DESC
                LIMIT 10
            """)
            results = cursor.fetchall()
            
            if results:
                print(f"Found {len(results)} recent records with advanced metrics:")
                print("-" * 80)
                headers = ["Symbol1", "Symbol2", "Z-Score", "Half-Life", "Sharpe", "Z>2", "Z<-2", "Timestamp"]
                print(f"{headers[0]:<8} {headers[1]:<8} {headers[2]:<8} {headers[3]:<10} {headers[4]:<8} {headers[5]:<4} {headers[6]:<5} {headers[7]}")
                print("-" * 80)
                
                for row in results:
                    symbol1, symbol2, z_score, half_life, sharpe, z_over_2, z_under_minus_2, timestamp = row
                    half_life_str = f"{half_life:.2f}" if half_life and half_life != float('inf') else "inf"
                    print(f"{symbol1:<8} {symbol2:<8} {z_score:<8.2f} {half_life_str:<10} {sharpe:<8.4f} {z_over_2:<4} {z_under_minus_2:<5} {timestamp}")
                
                print("\n✅ Advanced metrics are being calculated and stored successfully!")
                
                # Summary statistics
                valid_half_life = [row[3] for row in results if row[3] and row[3] != float('inf')]
                if valid_half_life:
                    avg_half_life = sum(valid_half_life) / len(valid_half_life)
                    print(f"\nSummary Statistics:")
                    print(f"  Average half-life: {avg_half_life:.2f}")
                    print(f"  Records with finite half-life: {len(valid_half_life)}/{len(results)}")
                
                total_z_breaches = sum(row[5] + row[6] for row in results)
                print(f"  Total Z-score breaches (|Z| > 2): {total_z_breaches}")
                
            else:
                print("❌ No recent records with advanced metrics found!")
                
                # Check if any records exist at all
                cursor.execute("""
                    SELECT COUNT(*) FROM hourly_statistics 
                    WHERE analysis_timestamp >= CURRENT_DATE
                """)
                total_count = cursor.fetchone()[0]
                print(f"Total records from today: {total_count}")
                
                if total_count > 0:
                    print("Records exist but advanced metrics are NULL - this may indicate a calculation error.")
        
        db.return_connection(connection)
        
    except Exception as e:
        print(f"❌ Error verifying metrics: {e}")

if __name__ == "__main__":
    verify_advanced_metrics()
