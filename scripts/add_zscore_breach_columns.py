#!/usr/bin/env python3
"""
Script to add Z-Score breach count columns to the pair_statistics table.

This script alters the pair_statistics table to add two new columns:
1. zscore_breach_count_above_2: Count of breaches above 2.0
2. zscore_breach_count_below_neg2: Count of breaches below -2.0
"""

import os
import logging
import psycopg2
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_db_config():
    """Load database configuration from environment variables."""
    # Load environment variables
    load_dotenv()
    
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'stat_arb'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
    }


def alter_pair_statistics_table():
    """
    Alter the pair_statistics table to add Z-Score breach count columns.
    
    Returns:
        bool: True if successful, False otherwise
    """
    db_config = load_db_config()
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Check if the columns already exist
        cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'pair_statistics' 
        AND column_name IN ('zscore_breach_count_above_2', 'zscore_breach_count_below_neg2')
        """)
        
        existing_columns = [row[0] for row in cursor.fetchall()]
        
        # Add missing columns
        if 'zscore_breach_count_above_2' not in existing_columns:
            logger.info("Adding zscore_breach_count_above_2 column...")
            cursor.execute("""
            ALTER TABLE pair_statistics
            ADD COLUMN zscore_breach_count_above_2 INTEGER DEFAULT 0
            """)
            logger.info("Column zscore_breach_count_above_2 added successfully")
        else:
            logger.info("Column zscore_breach_count_above_2 already exists")
        
        if 'zscore_breach_count_below_neg2' not in existing_columns:
            logger.info("Adding zscore_breach_count_below_neg2 column...")
            cursor.execute("""
            ALTER TABLE pair_statistics
            ADD COLUMN zscore_breach_count_below_neg2 INTEGER DEFAULT 0
            """)
            logger.info("Column zscore_breach_count_below_neg2 added successfully")
        else:
            logger.info("Column zscore_breach_count_below_neg2 already exists")
        
        # Commit the changes
        conn.commit()
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        logger.info("Database schema update completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error altering table: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting database schema update...")
    
    if alter_pair_statistics_table():
        logger.info("Successfully added Z-Score breach count columns")
    else:
        logger.error("Failed to add Z-Score breach count columns")
