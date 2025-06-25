"""
Cointegration Manager for Statistical Arbitrage.

Handles daily cointegration testing and storage of cointegrated pairs
for efficient hourly analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, date, timedelta
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import json

from .configuration_manager import ConfigurationManager
from .database_manager import DatabaseManager
from .data_manager import DataManager


class CointegrationManager:
    """
    Manages cointegration testing and storage for pair trading analysis.
    
    Features:
    - Daily comprehensive cointegration testing
    - Multiple cointegration tests (Engle-Granger, Johansen)
    - Efficient storage of cointegrated pairs
    - Batch processing for large datasets
    - Configurable significance levels
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize cointegration manager.
        
        Parameters:
        -----------
        config : ConfigurationManager
            Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config)
        self.data_manager = DataManager(config)
        
        # Configuration parameters
        self.significance_level = config.get_float(
            'COINTEGRATION_SIGNIFICANCE_LEVEL', 0.05
        )
        self.min_observations = config.get_int(
            'COINTEGRATION_MIN_OBSERVATIONS', 252
        )
        self.lookback_days = config.get_int('COINTEGRATION_LOOKBACK_DAYS', 365)
        self.test_method = config.get_str(
            'COINTEGRATION_TEST_METHOD', 'engle_granger'
        )
        
        self.logger.info("CointegrationManager initialized")
    
    def run_daily_cointegration_analysis(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive daily cointegration analysis.
        
        Parameters:
        -----------
        symbols : List[str], optional
            List of symbols to analyze. If None, uses all available symbols.
            
        Returns:
        --------
        Dict[str, Any]
            Results summary including number of pairs tested and cointegrated
        """
        self.logger.info("Starting daily cointegration analysis")
        
        try:
            # Get symbols if not provided
            if symbols is None:
                symbols = self._get_available_symbols()
            
            # Validate minimum number of symbols
            if len(symbols) < 2:
                raise ValueError(
                    "At least 2 symbols required for cointegration analysis"
                )
            
            # Get price data
            end_date = date.today()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            price_data = self._fetch_price_data(symbols, start_date, end_date)
            
            # Generate all possible pairs
            pairs_to_test = self._generate_pairs(symbols)
            
            # Run cointegration tests
            cointegration_results = self._test_cointegration_batch(
                pairs_to_test, price_data
            )
            
            # Store results
            stored_pairs = self._store_cointegration_results(cointegration_results)
            
            # Clean up old results
            self._cleanup_old_results()
            
            results_summary = {
                'analysis_date': end_date.isoformat(),
                'symbols_analyzed': len(symbols),
                'pairs_tested': len(pairs_to_test),
                'cointegrated_pairs': len(stored_pairs),
                'cointegration_rate': len(stored_pairs) / len(pairs_to_test) if pairs_to_test else 0,
                'test_method': self.test_method,
                'significance_level': self.significance_level
            }
            
            self.logger.info(
                f"Daily cointegration analysis completed. "
                f"Found {len(stored_pairs)} cointegrated pairs out of {len(pairs_to_test)} tested"
            )
            
            return results_summary
            
        except Exception as e:
            self.logger.error(f"Error in daily cointegration analysis: {e}")
            raise
    
    def get_cointegrated_pairs(self, as_of_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """
        Get list of cointegrated pairs for use in hourly analysis.
        
        Parameters:
        -----------
        as_of_date : date, optional
            Date to get cointegrated pairs for. Defaults to today.
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of cointegrated pairs with metadata
        """
        if as_of_date is None:
            as_of_date = date.today()
        
        try:
            connection = self.db_manager.get_connection()
            
            with connection.cursor() as cursor:
                query = """
                    SELECT 
                        symbol1,
                        symbol2,
                        test_statistic,
                        p_value,
                        critical_value,
                        hedge_ratio,
                        test_method,
                        analysis_date,
                        metadata
                    FROM cointegrated_pairs 
                    WHERE analysis_date = %s 
                    AND is_active = true
                    ORDER BY p_value ASC
                """
                
                cursor.execute(query, (as_of_date,))
                results = cursor.fetchall()
                
                pairs = []
                for row in results:
                    pair_data = {
                        'symbol1': row[0],
                        'symbol2': row[1],
                        'test_statistic': float(row[2]),
                        'p_value': float(row[3]),
                        'critical_value': float(row[4]),
                        'hedge_ratio': float(row[5]),
                        'test_method': row[6],
                        'analysis_date': row[7],
                        'metadata': (
                            row[8] if isinstance(row[8], dict)
                            else json.loads(row[8]) if row[8] else {}
                        )
                    }
                    pairs.append(pair_data)
                
                self.logger.info(f"Retrieved {len(pairs)} cointegrated pairs for {as_of_date}")
                return pairs
                
        except Exception as e:
            self.logger.error(f"Error retrieving cointegrated pairs: {e}")
            raise
        finally:
            self.db_manager.return_connection(connection)
    
    def _get_available_symbols(self) -> List[str]:
        """Get available symbols from the database."""
        try:
            connection = self.db_manager.get_connection()
            
            with connection.cursor() as cursor:
                # Get symbols that have sufficient recent data
                min_date = date.today() - timedelta(days=self.lookback_days)
                
                query = """
                    SELECT DISTINCT symbol
                    FROM historical_bars_1_day
                    WHERE bar_date >= %s
                    GROUP BY symbol
                    HAVING COUNT(*) >= %s
                    ORDER BY symbol
                """
                
                cursor.execute(query, (min_date, self.min_observations))
                results = cursor.fetchall()
                
                symbols = [row[0] for row in results]
                self.logger.info(f"Found {len(symbols)} symbols with sufficient data")
                return symbols
                
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            raise
        finally:
            self.db_manager.return_connection(connection)
    
    def _fetch_price_data(self, symbols: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch price data for symbols."""
        try:
            connection = self.db_manager.get_connection()
            
            # Build query for multiple symbols
            symbol_placeholders = ','.join(['%s'] * len(symbols))
            query = f"""
                SELECT symbol, bar_date, close_price
                FROM historical_bars_1_day
                WHERE symbol IN ({symbol_placeholders})
                AND bar_date BETWEEN %s AND %s
                ORDER BY bar_date ASC
            """
            
            params = symbols + [start_date, end_date]
            
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
            
            # Convert to DataFrame and pivot
            if not results:
                raise ValueError("No price data found for the specified period")
            
            df = pd.DataFrame(
                results, columns=['symbol', 'bar_date', 'close_price']
            )
            df['bar_date'] = pd.to_datetime(df['bar_date'])
            
            # Convert Decimal to float for mathematical operations
            # Handle both Decimal objects and regular numeric types
            df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
            
            # Pivot to get symbols as columns
            price_matrix = df.pivot(
                index='bar_date', columns='symbol', values='close_price'
            )
            
            # Fill missing values and ensure we have enough data
            price_matrix = price_matrix.dropna()
            
            if len(price_matrix) < self.min_observations:
                raise ValueError(
                    f"Insufficient data points: {len(price_matrix)} < {self.min_observations}"
                )
            
            self.logger.info(f"Fetched price data: {price_matrix.shape}")
            return price_matrix
            
        except Exception as e:
            self.logger.error(f"Error fetching price data: {e}")
            raise
        finally:
            self.db_manager.return_connection(connection)
    
    def _generate_pairs(self, symbols: List[str]) -> List[Tuple[str, str]]:
        """Generate all possible pairs from symbols."""
        pairs = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pairs.append((symbols[i], symbols[j]))
        
        self.logger.info(f"Generated {len(pairs)} pairs from {len(symbols)} symbols")
        return pairs
    
    def _test_cointegration_batch(self, pairs: List[Tuple[str, str]], 
                                 price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Test cointegration for a batch of pairs."""
        results = []
        
        self.logger.info(f"Testing cointegration for {len(pairs)} pairs")
        
        for i, (symbol1, symbol2) in enumerate(pairs):
            if (i + 1) % 100 == 0:
                self.logger.info(f"Processed {i + 1}/{len(pairs)} pairs")
            
            try:
                result = self._test_pair_cointegration(symbol1, symbol2, price_data)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Error testing pair {symbol1}-{symbol2}: {e}")
                continue
        
        self.logger.info(f"Cointegration testing completed. Found {len(results)} cointegrated pairs")
        return results
    
    def _test_pair_cointegration(self, symbol1: str, symbol2: str, 
                                price_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Test cointegration for a single pair."""
        try:
            # Get price series for the pair
            if symbol1 not in price_data.columns or symbol2 not in price_data.columns:
                return None
            
            prices1 = price_data[symbol1].dropna()
            prices2 = price_data[symbol2].dropna()
            
            # Align the series
            common_dates = prices1.index.intersection(prices2.index)
            if len(common_dates) < self.min_observations:
                return None
            
            y1 = prices1.loc[common_dates]
            y2 = prices2.loc[common_dates]
            
            # Ensure we have proper float values (not Decimal)
            y1 = pd.to_numeric(y1, errors='coerce').astype(float)
            y2 = pd.to_numeric(y2, errors='coerce').astype(float)
            
            # Convert to log prices for better statistical properties
            log_y1 = np.log(y1)
            log_y2 = np.log(y2)
            
            if self.test_method == 'engle_granger':
                return self._engle_granger_test(symbol1, symbol2, log_y1, log_y2)
            elif self.test_method == 'johansen':
                return self._johansen_test(symbol1, symbol2, log_y1, log_y2)
            else:
                raise ValueError(f"Unknown test method: {self.test_method}")
            
        except Exception as e:
            self.logger.warning(f"Error in pair cointegration test {symbol1}-{symbol2}: {e}")
            return None
    
    def _engle_granger_test(self, symbol1: str, symbol2: str, 
                           y1: pd.Series, y2: pd.Series) -> Optional[Dict[str, Any]]:
        """Perform Engle-Granger cointegration test."""
        try:
            # Test both directions and use the better one
            test_stat1, p_value1, critical_values1 = coint(y1, y2)
            test_stat2, p_value2, critical_values2 = coint(y2, y1)
            
            # Choose the test with lower p-value
            if p_value1 <= p_value2:
                test_stat, p_value, critical_values = test_stat1, p_value1, critical_values1
                dependent, independent = symbol1, symbol2
                y_dep, y_indep = y1, y2
            else:
                test_stat, p_value, critical_values = test_stat2, p_value2, critical_values2
                dependent, independent = symbol2, symbol1
                y_dep, y_indep = y2, y1
            
            # Check if cointegrated at the specified significance level
            critical_value = critical_values[1]  # 5% critical value
            is_cointegrated = p_value < self.significance_level
            
            if is_cointegrated:
                # Calculate hedge ratio using OLS
                ols_model = OLS(y_dep, y_indep).fit()
                hedge_ratio = ols_model.params.iloc[0]
                
                # Calculate additional statistics
                residuals = y_dep - hedge_ratio * y_indep
                
                metadata = {
                    'dependent_symbol': dependent,
                    'independent_symbol': independent,
                    'residual_mean': float(residuals.mean()),
                    'residual_std': float(residuals.std()),
                    'r_squared': float(ols_model.rsquared),
                    'observations': len(y_dep)
                }
                
                return {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'test_statistic': float(test_stat),
                    'p_value': float(p_value),
                    'critical_value': float(critical_value),
                    'hedge_ratio': float(hedge_ratio),
                    'test_method': 'engle_granger',
                    'analysis_date': date.today(),
                    'metadata': metadata
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error in Engle-Granger test: {e}")
            return None
    
    def _johansen_test(self, symbol1: str, symbol2: str, 
                      y1: pd.Series, y2: pd.Series) -> Optional[Dict[str, Any]]:
        """Perform Johansen cointegration test."""
        try:
            # Prepare data matrix - ensure float values
            data_matrix = np.column_stack([
                y1.astype(float).values, 
                y2.astype(float).values
            ])
            
            # Determine lag order (simplified - could be made more sophisticated)
            max_lags = min(12, len(data_matrix) // 20)
            
            # Run Johansen test
            johansen_result = coint_johansen(data_matrix, det_order=0, k_ar_lags=max_lags)
            
            # Check for cointegration using trace statistic
            trace_stat = johansen_result.lr1[0]  # Test for at least 1 cointegrating relationship
            critical_value = johansen_result.cvt[0, 1]  # 5% critical value
            
            is_cointegrated = trace_stat > critical_value
            
            if is_cointegrated:
                # Get cointegrating vector (normalized)
                coint_vector = johansen_result.evec[:, 0]
                hedge_ratio = -coint_vector[1] / coint_vector[0]
                
                # Calculate p-value approximation (not directly available from Johansen)
                # This is a simplified approximation
                excess_stat = trace_stat - critical_value
                p_value = max(0.01, min(0.049, 0.05 * np.exp(-excess_stat / 10)))
                
                metadata = {
                    'trace_statistic': float(trace_stat),
                    'eigenvalue': float(johansen_result.eig[0]),
                    'cointegrating_vector': coint_vector.tolist(),
                    'lag_order': max_lags,
                    'observations': len(data_matrix)
                }
                
                return {
                    'symbol1': symbol1,
                    'symbol2': symbol2,
                    'test_statistic': float(trace_stat),
                    'p_value': float(p_value),
                    'critical_value': float(critical_value),
                    'hedge_ratio': float(hedge_ratio),
                    'test_method': 'johansen',
                    'analysis_date': date.today(),
                    'metadata': metadata
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error in Johansen test: {e}")
            return None
    
    def _store_cointegration_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Store cointegration results in the database."""
        if not results:
            return []
        
        try:
            connection = self.db_manager.get_connection()
            
            # First, deactivate previous results for today
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE cointegrated_pairs SET is_active = false WHERE analysis_date = %s",
                    (date.today(),)
                )
            
            # Insert new results
            insert_query = """
                INSERT INTO cointegrated_pairs (
                    symbol1, symbol2, test_statistic, p_value, critical_value,
                    hedge_ratio, test_method, analysis_date, metadata, is_active,
                    created_at
                ) VALUES %s
                ON CONFLICT (symbol1, symbol2, analysis_date) 
                DO UPDATE SET
                    test_statistic = EXCLUDED.test_statistic,
                    p_value = EXCLUDED.p_value,
                    critical_value = EXCLUDED.critical_value,
                    hedge_ratio = EXCLUDED.hedge_ratio,
                    test_method = EXCLUDED.test_method,
                    metadata = EXCLUDED.metadata,
                    is_active = EXCLUDED.is_active,
                    updated_at = CURRENT_TIMESTAMP
            """
            
            values = []
            for result in results:
                values.append((
                    result['symbol1'],
                    result['symbol2'],
                    result['test_statistic'],
                    result['p_value'],
                    result['critical_value'],
                    result['hedge_ratio'],
                    result['test_method'],
                    result['analysis_date'],
                    json.dumps(result['metadata']),
                    True,  # is_active
                    datetime.now()
                ))
            
            with connection.cursor() as cursor:
                from psycopg2.extras import execute_values
                execute_values(cursor, insert_query, values)
            
            connection.commit()
            
            self.logger.info(f"Stored {len(results)} cointegration results")
            return results
            
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Error storing cointegration results: {e}")
            raise
        finally:
            self.db_manager.return_connection(connection)
    
    def _cleanup_old_results(self) -> None:
        """Clean up old cointegration results."""
        try:
            # Keep results for the last 30 days
            cutoff_date = date.today() - timedelta(days=30)
            
            connection = self.db_manager.get_connection()
            
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM cointegrated_pairs WHERE analysis_date < %s",
                    (cutoff_date,)
                )
                deleted_count = cursor.rowcount
            
            connection.commit()
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old cointegration results")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old results: {e}")
        finally:
            self.db_manager.return_connection(connection)
    
    def create_tables(self) -> None:
        """Create necessary database tables."""
        try:
            connection = self.db_manager.get_connection()
            
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS cointegrated_pairs (
                    id SERIAL PRIMARY KEY,
                    symbol1 VARCHAR(20) NOT NULL,
                    symbol2 VARCHAR(20) NOT NULL,
                    test_statistic DOUBLE PRECISION NOT NULL,
                    p_value DOUBLE PRECISION NOT NULL,
                    critical_value DOUBLE PRECISION NOT NULL,
                    hedge_ratio DOUBLE PRECISION NOT NULL,
                    test_method VARCHAR(50) NOT NULL,
                    analysis_date DATE NOT NULL,
                    metadata JSONB,
                    is_active BOOLEAN DEFAULT true,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol1, symbol2, analysis_date)
                );
                
                CREATE INDEX IF NOT EXISTS idx_cointegrated_pairs_date 
                ON cointegrated_pairs(analysis_date);
                
                CREATE INDEX IF NOT EXISTS idx_cointegrated_pairs_symbols 
                ON cointegrated_pairs(symbol1, symbol2);
                
                CREATE INDEX IF NOT EXISTS idx_cointegrated_pairs_active 
                ON cointegrated_pairs(is_active, analysis_date);
            """
            
            with connection.cursor() as cursor:
                cursor.execute(create_table_sql)
            
            connection.commit()
            self.logger.info("Cointegration tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
        finally:
            self.db_manager.return_connection(connection)
