#!/usr/bin/env python3
"""
Database Deduplication Manager

This script provides a safe, automated way to deduplicate the historical_bars_baseline
table while preserving data integrity and providing rollback capabilities.
"""

import sys
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statistics_service.data_ingestion.postgres_loader import get_engine
# DEPRECATED: This script uses the old statistics_service system
# Use analytics.stats.DatabaseManager instead
from sqlalchemy import text


class DatabaseDeduplicator:
    """Manages the deduplication process with safety checks and rollback capability."""
    
    def __init__(self):
        self.engine = get_engine()
        self.backup_table = "historical_bars_baseline_backup"
        self.dedup_table = "historical_bars_baseline_dedup"
        self.original_table = "historical_bars_baseline"
    
    def analyze_duplicates(self):
        """Analyze the current duplicate situation."""
        print("=== Duplicate Analysis ===")
        
        queries = {
            "Total records": "SELECT COUNT(*) FROM historical_bars_baseline",
            "Unique combinations": """
                SELECT COUNT(DISTINCT (symbol, bar_date, bar_size)) 
                FROM historical_bars_baseline
            """,
            "Sample duplicates": """
                SELECT symbol, bar_date, bar_size, COUNT(*) as duplicates
                FROM historical_bars_baseline
                GROUP BY symbol, bar_date, bar_size
                HAVING COUNT(*) > 1
                ORDER BY duplicates DESC
                LIMIT 5
            """
        }
        
        results = {}
        with self.engine.connect() as conn:
            for name, query in queries.items():
                result = conn.execute(text(query))
                if name in ["Total records", "Unique combinations"]:
                    results[name] = result.fetchone()[0]
                else:
                    results[name] = result.fetchall()
        
        total = results["Total records"]
        unique = results["Unique combinations"]
        duplicates = total - unique
        duplicate_pct = (duplicates / total) * 100
        
        print(f"Total records: {total:,}")
        print(f"Unique combinations: {unique:,}")
        print(f"Duplicate records: {duplicates:,}")
        print(f"Duplicate percentage: {duplicate_pct:.1f}%")
        
        print(f"\nTop duplicate groups:")
        for row in results["Sample duplicates"]:
            print(f"  {row.symbol} {row.bar_date} {row.bar_size}: "
                  f"{row.duplicates} copies")
        
        return {
            'total': total,
            'unique': unique,
            'duplicates': duplicates,
            'duplicate_percentage': duplicate_pct
        }
    
    def create_backup(self):
        """Create a backup of the original table."""
        print(f"\n=== Creating Backup ===")
        
        with self.engine.connect() as conn:
            # Drop backup if exists
            conn.execute(text(f"DROP TABLE IF EXISTS {self.backup_table}"))
            conn.commit()
            
            # Create backup
            start_time = time.time()
            conn.execute(text(f"""
                CREATE TABLE {self.backup_table} AS 
                SELECT * FROM {self.original_table}
            """))
            conn.commit()
            elapsed = time.time() - start_time
            
            # Verify backup
            result = conn.execute(text(f"SELECT COUNT(*) FROM {self.backup_table}"))
            backup_count = result.fetchone()[0]
            
            print(f"Backup created in {elapsed:.1f}s: {backup_count:,} records")
            return backup_count
    
    def create_deduplicated_table(self):
        """Create the deduplicated table."""
        print(f"\n=== Creating Deduplicated Table ===")
        
        with self.engine.connect() as conn:
            # Drop dedup table if exists
            conn.execute(text(f"DROP TABLE IF EXISTS {self.dedup_table}"))
            conn.commit()
            
            # Create deduplicated table using DISTINCT ON
            start_time = time.time()
            conn.execute(text(f"""
                CREATE TABLE {self.dedup_table} AS
                SELECT DISTINCT ON (symbol, bar_date, bar_size)
                    symbol, bar_date, bar_size,
                    open_price, high_price, low_price, close_price, volume
                FROM {self.original_table}
                ORDER BY symbol, bar_date, bar_size, ctid
            """))
            conn.commit()
            elapsed = time.time() - start_time
            
            # Verify deduplicated table
            result = conn.execute(text(f"SELECT COUNT(*) FROM {self.dedup_table}"))
            dedup_count = result.fetchone()[0]
            
            print(f"Deduplication completed in {elapsed:.1f}s: {dedup_count:,} records")
            return dedup_count
    
    def validate_deduplication(self, original_unique_count):
        """Validate that deduplication worked correctly."""
        print(f"\n=== Validation ===")
        
        with self.engine.connect() as conn:
            # Check record count matches expected unique count
            result = conn.execute(text(f"SELECT COUNT(*) FROM {self.dedup_table}"))
            dedup_count = result.fetchone()[0]
            
            # Check for any remaining duplicates
            result = conn.execute(text(f"""
                SELECT COUNT(*) FROM (
                    SELECT symbol, bar_date, bar_size, COUNT(*)
                    FROM {self.dedup_table}
                    GROUP BY symbol, bar_date, bar_size
                    HAVING COUNT(*) > 1
                ) t
            """))
            remaining_dups = result.fetchone()[0]
            
            # Sample data validation
            result = conn.execute(text(f"""
                SELECT COUNT(DISTINCT symbol) as symbols,
                       MIN(bar_date) as min_date,
                       MAX(bar_date) as max_date,
                       COUNT(DISTINCT bar_size) as timeframes
                FROM {self.dedup_table}
            """))
            stats = result.fetchone()
            
            print(f"Deduplicated record count: {dedup_count:,}")
            print(f"Expected unique count: {original_unique_count:,}")
            print(f"Remaining duplicates: {remaining_dups}")
            print(f"Symbols: {stats.symbols}")
            print(f"Date range: {stats.min_date} to {stats.max_date}")
            print(f"Timeframes: {stats.timeframes}")
            
            # Validation checks
            validation_passed = (
                dedup_count == original_unique_count and
                remaining_dups == 0 and
                stats.symbols > 0
            )
            
            if validation_passed:
                print("✅ Validation PASSED - Deduplication successful!")
            else:
                print("❌ Validation FAILED - Check the results")
            
            return validation_passed
    
    def replace_original_table(self):
        """Replace the original table with the deduplicated version."""
        print(f"\n=== Replacing Original Table ===")
        
        with self.engine.connect() as conn:
            transaction = conn.begin()
            try:
                # Drop original table
                conn.execute(text(f"DROP TABLE {self.original_table}"))
                
                # Rename deduplicated table
                conn.execute(text(f"""
                    ALTER TABLE {self.dedup_table} 
                    RENAME TO {self.original_table}
                """))
                
                transaction.commit()
                print("✅ Table replacement successful!")
                return True
                
            except Exception as e:
                transaction.rollback()
                print(f"❌ Table replacement failed: {e}")
                return False
    
    def add_unique_constraint(self):
        """Add unique constraint to prevent future duplicates."""
        print(f"\n=== Adding Unique Constraint ===")
        
        with self.engine.connect() as conn:
            try:
                conn.execute(text(f"""
                    ALTER TABLE {self.original_table}
                    ADD CONSTRAINT unique_bar_key 
                    UNIQUE (symbol, bar_date, bar_size)
                """))
                conn.commit()
                print("✅ Unique constraint added successfully!")
                return True
                
            except Exception as e:
                print(f"❌ Failed to add unique constraint: {e}")
                return False
    
    def create_optimized_indices(self):
        """Create optimized indices after deduplication."""
        print(f"\n=== Creating Optimized Indices ===")
        
        indices = [
            "CREATE INDEX CONCURRENTLY idx_symbol_date ON historical_bars_baseline (symbol, bar_date)",
            "CREATE INDEX CONCURRENTLY idx_date ON historical_bars_baseline (bar_date)",
            "CREATE INDEX CONCURRENTLY idx_symbol ON historical_bars_baseline (symbol)",
            "CREATE INDEX CONCURRENTLY idx_bar_size ON historical_bars_baseline (bar_size)"
        ]
        
        with self.engine.connect() as conn:
            for idx_sql in indices:
                try:
                    start_time = time.time()
                    conn.execute(text(idx_sql))
                    conn.commit()
                    elapsed = time.time() - start_time
                    print(f"✅ Created index in {elapsed:.1f}s")
                except Exception as e:
                    print(f"⚠️  Index creation skipped: {e}")
    
    def vacuum_and_analyze(self):
        """Vacuum and analyze the table to reclaim space and update statistics."""
        print(f"\n=== Vacuum and Analyze ===")
        
        with self.engine.connect() as conn:
            # Note: VACUUM FULL requires autocommit mode
            conn.connection.autocommit = True
            
            try:
                start_time = time.time()
                conn.execute(text(f"VACUUM FULL {self.original_table}"))
                vacuum_time = time.time() - start_time
                
                start_time = time.time()
                conn.execute(text(f"ANALYZE {self.original_table}"))
                analyze_time = time.time() - start_time
                
                print(f"✅ VACUUM FULL completed in {vacuum_time:.1f}s")
                print(f"✅ ANALYZE completed in {analyze_time:.1f}s")
                
            except Exception as e:
                print(f"⚠️  VACUUM/ANALYZE warning: {e}")
            finally:
                conn.connection.autocommit = False
    
    def cleanup_temp_tables(self):
        """Clean up temporary tables."""
        print(f"\n=== Cleanup ===")
        
        with self.engine.connect() as conn:
            # Keep backup table but remove dedup table if it exists
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {self.dedup_table}"))
                conn.commit()
                print(f"✅ Cleanup completed")
                print(f"📁 Backup table '{self.backup_table}' retained for safety")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
    
    def rollback_from_backup(self):
        """Rollback to backup if something goes wrong."""
        print(f"\n=== EMERGENCY ROLLBACK ===")
        
        with self.engine.connect() as conn:
            transaction = conn.begin()
            try:
                # Check if backup exists
                result = conn.execute(text(f"""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = '{self.backup_table}'
                """))
                
                if result.fetchone()[0] == 0:
                    print(f"❌ Backup table '{self.backup_table}' not found!")
                    return False
                
                # Restore from backup
                conn.execute(text(f"DROP TABLE IF EXISTS {self.original_table}"))
                conn.execute(text(f"""
                    ALTER TABLE {self.backup_table} 
                    RENAME TO {self.original_table}
                """))
                
                transaction.commit()
                print(f"✅ Successfully rolled back to backup!")
                return True
                
            except Exception as e:
                transaction.rollback()
                print(f"❌ Rollback failed: {e}")
                return False


def main():
    """Main deduplication process with user interaction."""
    print("Database Deduplication Manager")
    print("=" * 40)
    print("⚠️  WARNING: This will modify your database!")
    print("📁 A backup will be created before any changes.")
    
    deduplicator = DatabaseDeduplicator()
    
    try:
        # Step 1: Analyze current situation
        stats = deduplicator.analyze_duplicates()
        
        if stats['duplicate_percentage'] < 10:
            print(f"\n✅ Low duplicate rate ({stats['duplicate_percentage']:.1f}%)")
            print("Deduplication may not be necessary.")
            return
        
        # Confirm with user
        print(f"\n🔍 Found {stats['duplicates']:,} duplicate records "
              f"({stats['duplicate_percentage']:.1f}%)")
        
        response = input("\nProceed with deduplication? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
        
        # Step 2: Create backup
        backup_count = deduplicator.create_backup()
        
        # Step 3: Create deduplicated table
        dedup_count = deduplicator.create_deduplicated_table()
        
        # Step 4: Validate deduplication
        if not deduplicator.validate_deduplication(stats['unique']):
            print("\n❌ Validation failed! Check the results before proceeding.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled. Backup table preserved.")
                return
        
        # Step 5: Replace original table
        if not deduplicator.replace_original_table():
            print("\n❌ Failed to replace table! Attempting rollback...")
            deduplicator.rollback_from_backup()
            return
        
        # Step 6: Add unique constraint
        deduplicator.add_unique_constraint()
        
        # Step 7: Create optimized indices
        deduplicator.create_optimized_indices()
        
        # Step 8: Vacuum and analyze
        deduplicator.vacuum_and_analyze()
        
        # Step 9: Cleanup
        deduplicator.cleanup_temp_tables()
        
        # Final summary
        space_saved = (stats['total'] - dedup_count) / stats['total'] * 100
        print(f"\n🎉 DEDUPLICATION COMPLETE!")
        print(f"📊 Records: {stats['total']:,} → {dedup_count:,}")
        print(f"💾 Space saved: ~{space_saved:.1f}%")
        print(f"🔒 Unique constraint added to prevent future duplicates")
        print(f"⚡ Optimized indices created for better performance")
        
    except Exception as e:
        print(f"\n❌ Error during deduplication: {e}")
        print("Attempting emergency rollback...")
        deduplicator.rollback_from_backup()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
