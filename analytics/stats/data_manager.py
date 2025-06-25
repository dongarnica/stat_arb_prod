"""
Data Manager for Statistics System.

Handles data retrieval, caching, and preprocessing for statistics calculations.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import date, timedelta
import pandas as pd
import threading


class DataManager:
    """
    Manages data access and caching for the statistics system.
    
    Features:
    - Database connectivity for price data
    - Intelligent caching with TTL
    - Data validation and preprocessing
    - Pair data alignment
    - Memory-efficient data handling
    """
    
    def __init__(self, config):
        """
        Initialize data manager.
        
        Parameters:
        -----------
        config : ConfigurationManager
            Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cache configuration
        self.cache_ttl = config.get_int('STATISTICS_CACHE_TTL', 3600)
        self.max_cache_size = config.get_int('STATISTICS_CACHE_SIZE', 1000)
        
        # Data source configuration
        self.db_config = config.get_database_config()
        
        # Cache storage
        self._cache_lock = threading.Lock()
        self._data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Database connection (will be lazy-loaded)
        self._db_connection = None
        
        self.logger.info("DataManager initialized")
    
    def get_asset_data(self, symbol: str,
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Retrieve price data for a single asset.
        
        Parameters:
        -----------
        symbol : str
            Asset symbol
        start_date : date, optional
            Start date for data retrieval
        end_date : date, optional
            End date for data retrieval
            
        Returns:
        --------
        pd.DataFrame
            Price data with OHLCV columns
        """        # Set default date range if not provided
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        # Include current hour in cache key to force daily refresh
        from datetime import datetime
        current_hour = datetime.now().hour
        cache_key = f"{symbol}_{start_date}_{end_date}_h{current_hour}"
        
        # For current date queries, always bypass cache to ensure fresh data
        force_refresh = (end_date == date.today())
        
        # Check cache first (only if not forcing refresh)
        if not force_refresh:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Cache hit for {symbol}")
                return cached_data
        else:
            self.logger.debug(
                f"Forcing fresh data fetch for {symbol} (current date query)"
            )
        
        # Fetch from database
        try:
            data = self._fetch_from_database(symbol, start_date, end_date)
            
            # Validate and preprocess
            data = self._validate_and_preprocess(data, symbol)
            
            # Cache the result
            self._add_to_cache(cache_key, data)
            
            self.logger.debug(
                f"Fetched {len(data)} records for {symbol} "
                f"from {start_date} to {end_date}"
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get data for {symbol}: {e}")
            return pd.DataFrame()
    
    def align_pair_data(self, data1: pd.DataFrame,
                        data2: pd.DataFrame) -> pd.DataFrame:
        """
        Align data for two assets for pair trading analysis.
        
        Parameters:
        -----------
        data1 : pd.DataFrame
            First asset data
        data2 : pd.DataFrame
            Second asset data
            
        Returns:
        --------
        pd.DataFrame
            Aligned data with columns for both assets
        """
        if data1.empty or data2.empty:
            return pd.DataFrame()
        
        try:
            # Ensure both dataframes have datetime index
            if not isinstance(data1.index, pd.DatetimeIndex):
                data1.index = pd.to_datetime(data1.index)
            if not isinstance(data2.index, pd.DatetimeIndex):
                data2.index = pd.to_datetime(data2.index)
            
            # Find common date range
            common_start = max(data1.index.min(), data2.index.min())
            common_end = min(data1.index.max(), data2.index.max())
            
            # Filter to common range
            data1_filtered = data1[common_start:common_end]
            data2_filtered = data2[common_start:common_end]
            
            # Align on dates (inner join)
            aligned = pd.merge(
                data1_filtered, data2_filtered,
                left_index=True, right_index=True,
                how='inner',
                suffixes=('_1', '_2')
            )
            
            # Ensure we have enough data points
            min_required_points = self.config.get_int(
                'STATISTICS_MIN_DATA_POINTS', 30)
            
            if len(aligned) < min_required_points:
                self.logger.warning(
                    f"Insufficient aligned data points: {len(aligned)} "
                    f"(minimum required: {min_required_points})"
                )
                return pd.DataFrame()
            
            self.logger.debug(f"Aligned data: {len(aligned)} common points")
            return aligned
            
        except Exception as e:
            self.logger.error(f"Failed to align pair data: {e}")
            return pd.DataFrame()
    
    def get_pair_data(self, symbol1: str, symbol2: str,
                      start_date: Optional[date] = None,
                      end_date: Optional[date] = None) -> pd.DataFrame:
        """
        Get aligned data for a pair of assets.
        
        Parameters:
        -----------
        symbol1 : str
            First asset symbol
        symbol2 : str
            Second asset symbol
        start_date : Optional[date]
            Start date for data
        end_date : Optional[date]
            End date for data
            
        Returns:
        --------
        pd.DataFrame
            Aligned pair data with columns for both assets
        """
        try:
            # Get data for both symbols
            data1 = self.get_asset_data(symbol1, start_date, end_date)
            data2 = self.get_asset_data(symbol2, start_date, end_date)
            
            if data1.empty or data2.empty:
                self.logger.warning(f"No data available for pair {symbol1}/{symbol2}")
                return pd.DataFrame()
              # Align the data
            aligned_data = self.align_pair_data(data1, data2)
            
            if not aligned_data.empty:
                # Rename columns for pair analysis to match expected format
                # Analytics modules expect columns like 'asset_1_close', 'asset_2_close', etc.
                columns_map = {}
                for col in aligned_data.columns:
                    if col.endswith('_1'):
                        base_col = col[:-2]
                        columns_map[col] = f"asset_1_{base_col}"
                    elif col.endswith('_2'):
                        base_col = col[:-2]
                        columns_map[col] = f"asset_2_{base_col}"
                
                aligned_data = aligned_data.rename(columns=columns_map)
                
                self.logger.info(
                    f"Retrieved pair data for {symbol1}/{symbol2}: "
                    f"{len(aligned_data)} aligned records"
                )
            
            return aligned_data
            
        except Exception as e:
            self.logger.error(f"Failed to get pair data for {symbol1}/{symbol2}: {e}")
            return pd.DataFrame()

    def _fetch_from_database(self, symbol: str, start_date: date,
                             end_date: date) -> pd.DataFrame:
        """Fetch data from the database."""
        # Import here to avoid circular dependencies
        from .database_manager import DatabaseManager
        from sqlalchemy import create_engine
        
        if not self._db_connection:
            db_manager = DatabaseManager(self.config)
            self._db_connection = db_manager.get_connection()
        
        try:
            # Create SQLAlchemy engine for pandas compatibility
            db_config = self.config.get_database_config()
            connection_string = (
                f"postgresql://{db_config['user']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            engine = create_engine(connection_string)
            
            # Use the correct table name and column names from the database schema
            query = """
            SELECT 
                bar_date as date_time,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            FROM historical_bars_1_day 
            WHERE symbol = %(symbol)s 
                AND bar_date >= %(start_date)s 
                AND bar_date <= %(end_date)s
            ORDER BY bar_date
            """
            
            df = pd.read_sql_query(
                query,
                engine,
                params={'symbol': symbol, 'start_date': start_date, 'end_date': end_date},
                parse_dates=['date_time'],
                index_col='date_time'
            )
            
            # Rename columns to standard format
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Database query failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _validate_and_preprocess(self, data: pd.DataFrame,
                                 symbol: str) -> pd.DataFrame:
        """
        Validate and preprocess price data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw price data
        symbol : str
            Asset symbol for logging
            
        Returns:
        --------
        pd.DataFrame
            Validated and preprocessed data
        """
        if data.empty:
            return data
        
        original_length = len(data)
        
        # Remove rows with any NaN values
        data = data.dropna()
        
        # Validate price relationships (high >= low, etc.)
        valid_mask = (
            (data['high'] >= data['low']) &
            (data['high'] >= data['open']) &
            (data['high'] >= data['close']) &
            (data['low'] <= data['open']) &
            (data['low'] <= data['close']) &
            (data['volume'] >= 0)
        )
        
        data = data[valid_mask]
        
        # Remove extreme outliers (prices more than 10x different from previous)
        if len(data) > 1:
            price_ratios = data['close'] / data['close'].shift(1)
            outlier_mask = (price_ratios < 10) & (price_ratios > 0.1)
            outlier_mask.iloc[0] = True  # Keep first row
            data = data[outlier_mask]
        
        # Ensure minimum data points
        min_points = self.config.get_int('STATISTICS_MIN_DATA_POINTS', 30)
        if len(data) < min_points:
            self.logger.warning(
                f"Insufficient data for {symbol}: {len(data)} points "
                f"(minimum: {min_points})"
            )
        
        # Log data quality metrics
        if len(data) < original_length:
            removed = original_length - len(data)
            self.logger.debug(
                f"Data validation for {symbol}: removed {removed} "
                f"invalid records ({removed/original_length*100:.1f}%)"
            )
        
        return data
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if valid."""
        with self._cache_lock:
            if cache_key in self._data_cache:
                cache_entry = self._data_cache[cache_key]
                
                # Check if cache entry is still valid
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['data'].copy()
                else:
                    # Remove expired entry
                    del self._data_cache[cache_key]
        
        return None
    
    def _add_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Add data to cache."""
        with self._cache_lock:
            # Check cache size limit
            if len(self._data_cache) >= self.max_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = min(
                    self._data_cache.keys(),
                    key=lambda k: self._data_cache[k]['timestamp']
                )
                del self._data_cache[oldest_key]
              # Add new entry
            self._data_cache[cache_key] = {
                'data': data.copy(),
                'timestamp': time.time()
            }
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status."""
        with self._cache_lock:
            cache_size = len(self._data_cache)
            total_memory = sum(
                entry['data'].memory_usage(deep=True).sum()
                for entry in self._data_cache.values()
            )
            
            # Calculate hit rate (simplified)
            # In production, you'd track hits/misses more accurately
            cache_utilization = cache_size / self.max_cache_size
            
            return {
                'cache_entries': cache_size,
                'max_cache_size': self.max_cache_size,
                'cache_utilization': cache_utilization,
                'total_memory_mb': total_memory / 1024 / 1024,
                'ttl_seconds': self.cache_ttl
            }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._cache_lock:
            cleared_count = len(self._data_cache)
            self._data_cache.clear()
        
        self.logger.info(f"Cleared {cleared_count} cache entries")
    
    def clear_stale_cache(self, max_age_hours: int = 1) -> None:
        """Clear cache entries older than specified hours."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        with self._cache_lock:
            stale_keys = [
                key for key, entry in self._data_cache.items()
                if entry['timestamp'] < cutoff_time
            ]
            
            for key in stale_keys:
                del self._data_cache[key]
            
            if stale_keys:
                self.logger.info(f"Cleared {len(stale_keys)} stale cache entries")
    
    def clear_today_cache(self) -> None:
        """Clear all cache entries for today's date to force fresh data."""
        today_str = str(date.today())
        
        with self._cache_lock:
            keys_to_remove = [
                key for key in self._data_cache.keys()
                if today_str in key
            ]
            
            for key in keys_to_remove:
                del self._data_cache[key]
            
            if keys_to_remove:
                self.logger.info(
                    f"Cleared {len(keys_to_remove)} cache entries for today"
                )
    
    def warm_cache(self, symbols: List[str],
                   days_back: int = 365) -> Dict[str, bool]:
        """
        Pre-warm cache with data for specified symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of symbols to cache
        days_back : int
            Number of days of historical data to cache
            
        Returns:
        --------
        Dict[str, bool]
            Success status for each symbol
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_asset_data(symbol, start_date, end_date)
                results[symbol] = not data.empty
                
                if data.empty:
                    self.logger.warning(f"No data available for {symbol}")
                else:
                    self.logger.debug(f"Cached {len(data)} records for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Failed to warm cache for {symbol}: {e}")
                results[symbol] = False
        
        successful = sum(results.values())
        self.logger.info(
            f"Cache warming completed: {successful}/{len(symbols)} "
            f"symbols loaded"
        )
        
        return results
    
    def validate_data_access(self) -> bool:
        """
        Validate that data access is working properly.
        
        Returns:
        --------
        bool
            True if data access is functional
        """
        try:
            # Test with a simple query for a common symbol
            test_symbol = self.config.get_str('STATISTICS_TEST_SYMBOL', 'SPY')
            test_date = date.today() - timedelta(days=7)
            
            test_data = self.get_asset_data(
                test_symbol,
                test_date,
                date.today()
            )
            
            if test_data.empty:
                self.logger.warning(
                    f"Data validation warning: No data for test symbol "
                    f"{test_symbol}"
                )
                return False
            
            # Validate data structure
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in test_data.columns for col in required_columns):
                self.logger.error("Data validation failed: Missing required columns")
                return False
            
            self.logger.info("Data access validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Data access validation failed: {e}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols in the database.
        
        Returns:
        --------
        List[str]
            List of available symbols
        """
        try:
            # Import here to avoid circular dependencies
            from .database_manager import DatabaseManager
            
            if not self._db_connection:
                db_manager = DatabaseManager(self.config)
                self._db_connection = db_manager.get_connection()
            
            query = ("SELECT DISTINCT symbol FROM historical_bars_1_day "
                     "ORDER BY symbol")
            
            with self._db_connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                return [row[0] for row in results]
                
        except Exception as e:
            self.logger.error(f"Failed to get available symbols: {e}")
            return []
    
    def get_data_quality_report(self, symbol: str,
                                days_back: int = 30) -> Dict[str, Any]:
        """
        Generate data quality report for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Symbol to analyze
        days_back : int
            Number of days to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Data quality metrics
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            data = self.get_asset_data(symbol, start_date, end_date)
            
            if data.empty:
                return {
                    'symbol': symbol,
                    'status': 'no_data',
                    'period_days': days_back
                }
            
            # Calculate quality metrics
            total_points = len(data)
            missing_points = data.isnull().sum().sum()
            
            # Price consistency checks
            price_gaps = (data['close'] / data['close'].shift(1))
            large_gaps = ((price_gaps > 1.5) | (price_gaps < 0.5)).sum()
            
            # Volume analysis
            zero_volume_days = (data['volume'] == 0).sum()
            
            return {
                'symbol': symbol,
                'status': 'analyzed',
                'period_days': days_back,
                'total_data_points': total_points,
                'missing_values': int(missing_points),
                'data_completeness': 1 - (missing_points / (total_points * 5)),
                'large_price_gaps': int(large_gaps),
                'zero_volume_days': int(zero_volume_days),
                'date_range': {
                    'start': data.index.min().date().isoformat(),
                    'end': data.index.max().date().isoformat()
                }
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e),
                'period_days': days_back
            }
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.clear_cache()
        
        if self._db_connection:
            try:
                self._db_connection.close()
                self._db_connection = None
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")
        
        self.logger.info("DataManager cleanup completed")
