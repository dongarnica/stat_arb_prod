#!/usr/bin/env python3
"""
Simple Backtest Table Creator

Creates backtest tables using the modern analytics.stats system.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analytics"))

from analytics.stats import DatabaseManager, ConfigurationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_tables_simple():
    """Create backtest tables using the modern database manager."""
    
    logger.info("Creating backtest tables...")
    
    # Define table creation SQL
    tables_sql = {
        'backtest_portfolio': """
            CREATE TABLE IF NOT EXISTS backtest_portfolio (
                id SERIAL PRIMARY KEY,
                time TIMESTAMP NOT NULL,
                value NUMERIC(15,6) NOT NULL,
                strategy_name VARCHAR(100),
                run_id VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        
        'backtest_trades': """
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id SERIAL PRIMARY KEY,
                time TIMESTAMP NOT NULL,
                signal NUMERIC(10,6),
                hedge_ratio NUMERIC(15,8),
                delta_a NUMERIC(15,6),
                delta_b NUMERIC(15,6),
                cash_change NUMERIC(15,6),
                new_positions JSONB,
                strategy_name VARCHAR(100),
                run_id VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        
        'backtest_performance': """
            CREATE TABLE IF NOT EXISTS backtest_performance (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                strategy_name VARCHAR(100),
                run_id VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
    }
    
    # Define indices
    indices_sql = [
        "CREATE INDEX IF NOT EXISTS idx_backtest_portfolio_time ON backtest_portfolio (time)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_portfolio_strategy ON backtest_portfolio (strategy_name)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_trades_time ON backtest_trades (time)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_trades_strategy ON backtest_trades (strategy_name)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_performance_metric ON backtest_performance (metric_name)",
        "CREATE INDEX IF NOT EXISTS idx_backtest_performance_strategy ON backtest_performance (strategy_name)"
    ]
    
    engine = get_engine()
    
    success_count = 0
    
    # Create tables
    for table_name, sql in tables_sql.items():
        try:
            with engine.begin() as conn:
                conn.execute(text(sql))
                logger.info(f"✅ Created table: {table_name}")
                success_count += 1
        except Exception as e:
            logger.error(f"❌ Failed to create {table_name}: {e}")
    
    # Create indices
    for idx_sql in indices_sql:
        try:
            with engine.begin() as conn:
                conn.execute(text(idx_sql))
                logger.info(f"✅ Created index")
        except Exception as e:
            logger.warning(f"⚠️  Index creation failed (may exist): {e}")
    
    return success_count == len(tables_sql)


def verify_and_show_tables():
    """Verify tables and show their structure."""
    
    engine = get_engine()
    
    try:
        with engine.connect() as conn:
            # Check tables exist
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'backtest_%'
                ORDER BY table_name
            """))
            
            tables = [row[0] for row in result.fetchall()]
            
            logger.info("\\n=== BACKTEST TABLES CREATED ===")
            for table in tables:
                logger.info(f"✅ {table}")
                
                # Show row count
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = count_result.fetchone()[0]
                logger.info(f"   📊 Rows: {count}")
            
            if len(tables) >= 3:
                logger.info("\\n🎉 SUCCESS: All backtest tables created!")
                return True
            else:
                logger.error(f"\\n❌ Only {len(tables)} tables found")
                return False
                
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False


def main():
    """Main function."""
    
    logger.info("🚀 Creating backtest database tables...")
    
    if create_tables_simple():
        if verify_and_show_tables():
            logger.info("\\n✅ Database setup complete!")
            logger.info("✅ Ready to save backtest results to database")
            return True
    
    logger.error("\\n❌ Database setup failed")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
