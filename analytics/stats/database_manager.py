"""
Database Manager for Statistics System.

Handles PostgreSQL database connections, schema management, and data
operations.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool
import json

from .configuration_manager import ConfigurationManager


class DatabaseManager:
    """
    Manages database operations for statistics storage and retrieval.
    
    Features:
    - Connection pooling for performance
    - Transaction management
    - Schema validation and creation
    - Batch operations for efficiency
    - Error handling and retry logic
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize database manager.
        
        Parameters:
        -----------
        config : ConfigurationManager
            Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.db_config = config.get_database_config()
        
        # Connection pool
        self.connection_pool: Optional[ThreadedConnectionPool] = None
        
        # Initialize connection pool
        self._create_connection_pool()
        
        self.logger.info("DatabaseManager initialized")
    
    def _create_connection_pool(self) -> None:
        """Create database connection pool."""
        try:
            min_connections = self.config.get_int(
                'STATISTICS_DB_MIN_CONNECTIONS', 1)
            max_connections = self.config.get_int(
                'STATISTICS_DB_MAX_CONNECTIONS', 10)
            
            if 'database_url' in self.db_config:
                # Use full database URL
                self.connection_pool = ThreadedConnectionPool(
                    min_connections, max_connections,
                    dsn=self.db_config['database_url']
                )
            else:
                # Use individual parameters
                self.connection_pool = ThreadedConnectionPool(
                    min_connections, max_connections,
                    host=self.db_config['host'],
                    port=self.db_config['port'],
                    database=self.db_config['database'],
                    user=self.db_config['user'],
                    password=self.db_config['password'],
                    sslmode=self.db_config.get('sslmode', 'prefer')
                )
            
            self.logger.info(
                f"Database connection pool created: "
                f"{min_connections}-{max_connections} connections"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool."""
        if not self.connection_pool:
            raise RuntimeError("Connection pool not initialized")
        return self.connection_pool.getconn()
    
    def return_connection(self, connection) -> None:
        """Return connection to pool."""
        if self.connection_pool:
            self.connection_pool.putconn(connection)
    
    def validate_connection(self) -> bool:
        """
        Validate database connectivity.
        
        Returns:
        --------
        bool
            True if connection is successful
        """
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result[0] == 1
            finally:
                self.return_connection(conn)
        except Exception as e:
            self.logger.error(f"Database connection validation failed: {e}")
            return False
    
    def ensure_schema(self) -> None:
        """Ensure statistics tables and schema exist."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Create statistics table if not exists
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS ai_statistics (
                    id SERIAL PRIMARY KEY,
                    calculation_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    symbol VARCHAR(10) NOT NULL,
                    pair_symbol VARCHAR(10),
                    statistic_module VARCHAR(100) NOT NULL,
                    statistic_name VARCHAR(100) NOT NULL,
                    statistic_type VARCHAR(50) NOT NULL,
                    statistic_category VARCHAR(50) NOT NULL,
                    output_type VARCHAR(50) NOT NULL,
                    statistic_value JSONB NOT NULL,
                    metadata JSONB,
                    data_start_date DATE,
                    data_end_date DATE,
                    data_points_used INTEGER,
                    success BOOLEAN NOT NULL DEFAULT TRUE,
                    error_message TEXT,
                    execution_time_ms FLOAT,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """
                cursor.execute(create_table_sql)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_ai_statistics_symbol "
                    "ON ai_statistics(symbol);",
                    
                    "CREATE INDEX IF NOT EXISTS idx_ai_statistics_pair "
                    "ON ai_statistics(symbol, pair_symbol);",
                    
                    "CREATE INDEX IF NOT EXISTS idx_ai_statistics_module "
                    "ON ai_statistics(statistic_module);",
                    
                    "CREATE INDEX IF NOT EXISTS idx_ai_statistics_timestamp "
                    "ON ai_statistics(calculation_timestamp);",
                    
                    "CREATE INDEX IF NOT EXISTS idx_ai_statistics_type "
                    "ON ai_statistics(statistic_type);",
                    
                    "CREATE INDEX IF NOT EXISTS idx_ai_statistics_category "
                    "ON ai_statistics(statistic_category);",
                    
                    "CREATE INDEX IF NOT EXISTS idx_ai_statistics_success "
                    "ON ai_statistics(success);",
                    
                    "CREATE INDEX IF NOT EXISTS idx_ai_statistics_value "
                    "ON ai_statistics USING GIN(statistic_value);"
                ]
                
                for index_sql in indexes:
                    cursor.execute(index_sql)
                
                conn.commit()
                self.logger.info("Database schema validated/created successfully")
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Schema creation failed: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def write_statistics(self, statistics: List[Dict[str, Any]]) -> int:
        """
        Write statistics to database in batch.
        
        Parameters:
        -----------
        statistics : List[Dict[str, Any]]
            List of statistic results to write
            
        Returns:
        --------
        int
            Number of records written
        """
        if not statistics:
            return 0
        
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:                # Prepare data for batch insert
                insert_data = []
                for stat in statistics:
                    insert_data.append((
                        stat.get('calculation_timestamp', datetime.now()),
                        stat.get('symbol'),
                        stat.get('pair_symbol'),
                        stat.get('statistic_module'),
                        stat.get('statistic_name'),
                        stat.get('statistic_type'),
                        stat.get('statistic_category'),
                        stat.get('output_type'),
                        json.dumps(stat.get('statistic_value')),
                        json.dumps(stat.get('metadata', {})),
                        stat.get('data_start_date'),
                        stat.get('data_end_date'),
                        stat.get('data_points_used'),
                        stat.get('success', True),
                        stat.get('error_message'),
                        stat.get('execution_time_ms')
                    ))
                
                # Batch insert
                insert_sql = """
                INSERT INTO ai_statistics (
                    calculation_timestamp, symbol, pair_symbol,
                    statistic_module, statistic_name, statistic_type,
                    statistic_category, output_type, statistic_value,
                    metadata, data_start_date, data_end_date,
                    data_points_used, success, error_message,
                    execution_time_ms
                ) VALUES %s
                """
                
                execute_values(cursor, insert_sql, insert_data)
                conn.commit()
                
                written_count = len(insert_data)
                self.logger.info(f"Written {written_count} statistics to database")
                return written_count
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to write statistics: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def get_statistics_summary(self,
                               start_date: Optional[date] = None,
                               end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Get summary of statistics in database.
        
        Parameters:
        -----------
        start_date : date, optional
            Start date for summary period
        end_date : date, optional
            End date for summary period
            
        Returns:
        --------
        Dict[str, Any]
            Summary statistics
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build WHERE clause for date filtering
                where_clause = "WHERE 1=1"
                params = []
                
                if start_date:
                    where_clause += " AND calculation_timestamp >= %s"
                    params.append(start_date)
                
                if end_date:
                    where_clause += " AND calculation_timestamp <= %s"
                    params.append(end_date)
                
                # Get overall summary
                summary_sql = f"""
                SELECT 
                    COUNT(*) as total_statistics,
                    COUNT(DISTINCT symbol) as unique_symbols,
                    COUNT(DISTINCT statistic_module) as unique_modules,
                    COUNT(DISTINCT statistic_type) as unique_types,
                    COUNT(CASE WHEN success THEN 1 END) as successful_calcs,
                    COUNT(CASE WHEN NOT success THEN 1 END) as failed_calcs,
                    MIN(calculation_timestamp) as earliest_calc,
                    MAX(calculation_timestamp) as latest_calc
                FROM ai_statistics 
                {where_clause}
                """
                
                cursor.execute(summary_sql, params)
                summary = dict(cursor.fetchone())
                
                # Get breakdown by type
                type_sql = f"""
                SELECT 
                    statistic_type,
                    COUNT(*) as count,
                    COUNT(CASE WHEN success THEN 1 END) as successful
                FROM ai_statistics 
                {where_clause}
                GROUP BY statistic_type
                ORDER BY count DESC
                """
                
                cursor.execute(type_sql, params)
                type_breakdown = [dict(row) for row in cursor.fetchall()]
                
                # Get breakdown by module
                module_sql = f"""
                SELECT 
                    statistic_module,
                    COUNT(*) as count,
                    COUNT(CASE WHEN success THEN 1 END) as successful
                FROM ai_statistics 
                {where_clause}
                GROUP BY statistic_module
                ORDER BY count DESC
                LIMIT 20
                """
                
                cursor.execute(module_sql, params)
                module_breakdown = [dict(row) for row in cursor.fetchall()]
                
                return {
                    'summary': summary,
                    'type_breakdown': type_breakdown,
                    'module_breakdown': module_breakdown
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics summary: {e}")
            return {'error': str(e)}
        finally:
            self.return_connection(conn)
    
    def get_statistics_by_symbol(self,
                                 symbol: str,
                                 pair_symbol: Optional[str] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get statistics for a specific symbol or pair.
        
        Parameters:
        -----------
        symbol : str
            Primary symbol
        pair_symbol : str, optional
            Second symbol for pair trading statistics
        limit : int
            Maximum number of records to return
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of statistics records
        """
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                if pair_symbol:
                    # Pair trading query
                    sql = """
                    SELECT * FROM ai_statistics 
                    WHERE symbol = %s AND pair_symbol = %s
                    ORDER BY calculation_timestamp DESC
                    LIMIT %s
                    """
                    cursor.execute(sql, (symbol, pair_symbol, limit))
                else:
                    # Single asset query
                    sql = """
                    SELECT * FROM ai_statistics 
                    WHERE symbol = %s AND pair_symbol IS NULL
                    ORDER BY calculation_timestamp DESC
                    LIMIT %s
                    """
                    cursor.execute(sql, (symbol, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(
                f"Failed to get statistics for {symbol}: {e}")
            return []
        finally:
            self.return_connection(conn)
    
    def cleanup_old_statistics(self, days_to_keep: int = 30) -> int:
        """
        Remove old statistics records.
        
        Parameters:
        -----------
        days_to_keep : int
            Number of days of data to retain
            
        Returns:
        --------
        int
            Number of records deleted
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                DELETE FROM ai_statistics 
                WHERE calculation_timestamp < NOW() - INTERVAL '%s days'
                """
                cursor.execute(sql, (days_to_keep,))
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old statistics")
                return deleted_count
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to cleanup old statistics: {e}")
            return 0
        finally:
            self.return_connection(conn)
    
    def check_health(self) -> Dict[str, Any]:
        """Check database health status."""
        try:
            start_time = time.time()
            is_connected = self.validate_connection()
            response_time = (time.time() - start_time) * 1000
            
            if is_connected:
                # Get connection pool status
                pool_info = {}
                if self.connection_pool:
                    pool_info = {
                        'pool_size': self.connection_pool.maxconn,
                        'available_connections': len(
                            self.connection_pool._pool),
                        'used_connections': (
                            self.connection_pool.maxconn -
                            len(self.connection_pool._pool))
                    }
                
                return {
                    'status': 'healthy',
                    'connected': True,
                    'response_time_ms': response_time,
                    'pool_info': pool_info
                }
            else:
                return {
                    'status': 'unhealthy',
                    'connected': False,
                    'response_time_ms': response_time
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'connected': False,
                'error': str(e)
            }
    
    def close_connections(self) -> None:
        """Close all database connections."""
        try:
            if self.connection_pool:
                self.connection_pool.closeall()
                self.connection_pool = None
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")
    
    def connect(self) -> None:
        """
        Connect to the database.
        This method validates the connection pool is working.
        """
        try:
            # Test the connection
            if self.validate_connection():
                self.logger.info("Database connection validated successfully")
                # Ensure schema exists
                self.ensure_schema()
            else:
                raise RuntimeError("Database connection validation failed")
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self) -> None:
        """Disconnect and close all connections."""
        self.close_connections()
    
    def insert_statistic(self, statistic: Dict[str, Any]) -> None:
        """
        Insert a single statistic into the database.
        
        Parameters:
        -----------
        statistic : Dict[str, Any]
            Statistic data with required fields
        """
        try:
            # Convert single statistic to list and use batch method
            self.write_statistics([statistic])
        except Exception as e:
            self.logger.error(f"Failed to insert statistic: {e}")
            raise
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a query and return results.
        
        Parameters:
        -----------
        query : str
            SQL query to execute
        params : tuple, optional
            Query parameters
            
        Returns:
        --------
        List[Dict[str, Any]]
            Query results as list of dictionaries
        """
        connection = None
        try:
            connection = self.get_connection()
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    results = [dict(row) for row in cursor.fetchall()]
                    return results
                else:
                    return []
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
        finally:
            if connection:
                self.return_connection(connection)
