"""
Hourly Statistics Manager for Statistical Arbitrage.

Performs fast statistical calculations on pre-identified cointegrated pairs
from the daily cointegration analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, date, timedelta
import json

from .configuration_manager import ConfigurationManager
from .database_manager import DatabaseManager
from .data_manager import DataManager
from .cointegration_manager import CointegrationManager


class HourlyStatisticsManager:
    """
    Manages hourly statistical calculations for cointegrated pairs.
    
    Features:
    - Fast calculations on pre-filtered cointegrated pairs
    - Z-score calculations and monitoring
    - Moving averages and technical indicators
    - Real-time spread analysis
    - Efficient batch processing
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize hourly statistics manager.
        
        Parameters:
        -----------
        config : ConfigurationManager
            Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config)
        self.data_manager = DataManager(config)
        self.cointegration_manager = CointegrationManager(config)
        
        # Configuration parameters
        self.z_score_window = config.get_int('ZSCORE_WINDOW', 252)
        self.ma_short_window = config.get_int('MA_SHORT_WINDOW', 20)
        self.ma_long_window = config.get_int('MA_LONG_WINDOW', 50)
        self.lookback_hours = config.get_int('HOURLY_LOOKBACK_HOURS', 24)
        self.z_score_threshold = config.get_float('ZSCORE_THRESHOLD', 2.0)
        
        self.logger.info("HourlyStatisticsManager initialized")
    
    def run_hourly_analysis(
        self, as_of_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Run hourly statistical analysis on cointegrated pairs.
        
        Parameters:
        -----------
        as_of_date : date, optional
            Date to get cointegrated pairs for. Defaults to today.
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results summary
        """
        self.logger.info("Starting hourly statistical analysis")
        
        try:
            if as_of_date is None:
                as_of_date = date.today()
            
            # Get cointegrated pairs from daily analysis
            cointegrated_pairs = self.cointegration_manager.get_cointegrated_pairs(
                as_of_date
            )
            
            if not cointegrated_pairs:
                self.logger.warning(
                    f"No cointegrated pairs found for {as_of_date}"
                )
                return {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'pairs_analyzed': 0,
                    'signals_generated': 0,
                    'message': 'No cointegrated pairs available'
                }
            
            # Get current price data for all symbols
            symbols = self._extract_symbols_from_pairs(cointegrated_pairs)
            current_prices = self._fetch_recent_prices(symbols)
            
            # Calculate statistics for each pair
            analysis_results = []
            signals_generated = 0
            
            for pair_info in cointegrated_pairs:
                try:
                    result = self._analyze_pair_hourly(pair_info, current_prices)
                    if result:
                        analysis_results.append(result)
                        if result.get('has_signal', False):
                            signals_generated += 1
                except Exception as e:
                    self.logger.warning(
                        f"Error analyzing pair {pair_info['symbol1']}-"
                        f"{pair_info['symbol2']}: {e}"
                    )
                    continue
            
            # Store results
            self._store_hourly_results(analysis_results)
            
            # Generate summary
            results_summary = {
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_date': as_of_date.isoformat(),
                'pairs_analyzed': len(analysis_results),
                'signals_generated': signals_generated,
                'total_cointegrated_pairs': len(cointegrated_pairs),
                'success_rate': (
                    len(analysis_results) / len(cointegrated_pairs)
                    if cointegrated_pairs else 0
                )
            }
            
            self.logger.info(
                f"Hourly analysis completed. Analyzed {len(analysis_results)} "
                f"pairs, generated {signals_generated} signals"
            )
            
            return results_summary
            
        except Exception as e:
            self.logger.error(f"Error in hourly analysis: {e}")
            raise
    
    def get_current_signals(
        self, min_z_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get current trading signals from hourly analysis.
        
        Parameters:
        -----------
        min_z_score : float, optional
            Minimum absolute Z-score for signal filtering
            
        Returns:
        --------
        List[Dict[str, Any]]
            Current trading signals
        """
        if min_z_score is None:
            min_z_score = self.z_score_threshold
        
        try:
            connection = self.db_manager.get_connection()
            
            # Get the most recent analysis results
            query = """
                SELECT 
                    symbol1,
                    symbol2,
                    current_z_score,
                    spread_value,
                    ma_short,
                    ma_long,
                    signal_type,
                    signal_strength,
                    analysis_timestamp,
                    metadata
                FROM hourly_statistics 
                WHERE DATE(analysis_timestamp) = CURRENT_DATE
                AND ABS(current_z_score) >= %s
                AND has_signal = true
                ORDER BY analysis_timestamp DESC, ABS(current_z_score) DESC
            """
            
            with connection.cursor() as cursor:
                cursor.execute(query, (min_z_score,))
                results = cursor.fetchall()
                
                signals = []
                for row in results:
                    signal_data = {
                        'symbol1': row[0],
                        'symbol2': row[1],
                        'current_z_score': float(row[2]),
                        'spread_value': float(row[3]),
                        'ma_short': float(row[4]) if row[4] else None,
                        'ma_long': float(row[5]) if row[5] else None,
                        'signal_type': row[6],
                        'signal_strength': float(row[7]),
                        'analysis_timestamp': row[8],
                        'metadata': json.loads(row[9]) if row[9] else {}
                    }
                    signals.append(signal_data)
                
                self.logger.info(f"Retrieved {len(signals)} current signals")
                return signals
                
        except Exception as e:
            self.logger.error(f"Error retrieving current signals: {e}")
            raise
        finally:
            self.db_manager.return_connection(connection)
    
    def _extract_symbols_from_pairs(
        self, pairs: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract unique symbols from pairs list."""
        symbols = set()
        for pair in pairs:
            symbols.add(pair['symbol1'])
            symbols.add(pair['symbol2'])
        return list(symbols)
    
    def _fetch_recent_prices(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch recent price data for symbols."""
        try:
            connection = self.db_manager.get_connection()
            
            # Get sufficient data for statistical calculations
            # Use lookback hours from configuration, with minimum for calculations
            lookback_hours = max(self.lookback_hours, self.z_score_window + 50)
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=lookback_hours)
            
            symbol_placeholders = ','.join(['%s'] * len(symbols))
            query = f"""
                SELECT symbol,
                       (bar_date + bar_time) as timestamp,
                       close_price
                FROM historical_bars_1_hour
                WHERE symbol IN ({symbol_placeholders})
                AND (bar_date + bar_time) >= %s
                ORDER BY bar_date ASC, bar_time ASC
            """
            
            params = symbols + [start_date]
            
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
            
            if not results:
                # Fallback to daily prices if intraday not available
                return self._fetch_daily_prices_fallback(symbols)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                results, columns=['symbol', 'timestamp', 'close_price']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert Decimal to float for mathematical operations
            df['close_price'] = pd.to_numeric(
                df['close_price'], errors='coerce'
            )
            
            # Pivot to get symbols as columns
            price_matrix = df.pivot(
                index='timestamp', columns='symbol', values='close_price'
            )
            
            # Forward fill missing values
            price_matrix = price_matrix.fillna(method='ffill')
            
            self.logger.info(f"Fetched recent prices: {price_matrix.shape}")
            return price_matrix
            
        except Exception as e:
            self.logger.error(f"Error fetching recent prices: {e}")
            # Fallback to daily prices
            return self._fetch_daily_prices_fallback(symbols)
        finally:
            self.db_manager.return_connection(connection)
    
    def _fetch_daily_prices_fallback(self, symbols: List[str]) -> pd.DataFrame:
        """Fallback to daily prices if intraday data not available."""
        try:
            connection = self.db_manager.get_connection()
            
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
            
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
            
            if not results:
                raise ValueError("No price data available for symbols")
            
            df = pd.DataFrame(
                results, columns=['symbol', 'bar_date', 'close_price']
            )
            df['bar_date'] = pd.to_datetime(df['bar_date'])
            
            # Convert Decimal to float for mathematical operations
            df['close_price'] = pd.to_numeric(
                df['close_price'], errors='coerce'
            )
            
            price_matrix = df.pivot(
                index='bar_date', columns='symbol', values='close_price'
            )
            price_matrix = price_matrix.fillna(method='ffill')
            
            return price_matrix
            
        except Exception as e:
            self.logger.error(f"Error in daily prices fallback: {e}")
            raise
        finally:
            self.db_manager.return_connection(connection)
    
    def _analyze_pair_hourly(
        self, pair_info: Dict[str, Any], price_data: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single pair for hourly statistics."""
        try:
            symbol1 = pair_info['symbol1']
            symbol2 = pair_info['symbol2']
            hedge_ratio = pair_info['hedge_ratio']
            
            # Check if symbols are available in price data
            if (symbol1 not in price_data.columns or
                    symbol2 not in price_data.columns):
                return None
            
            # Get price series
            prices1 = price_data[symbol1].dropna()
            prices2 = price_data[symbol2].dropna()
            
            # Align series
            common_index = prices1.index.intersection(prices2.index)
            if len(common_index) < self.ma_long_window:
                return None
            
            prices1_aligned = prices1.loc[common_index]
            prices2_aligned = prices2.loc[common_index]
            
            # Calculate spread
            spread = prices1_aligned - hedge_ratio * prices2_aligned
            
            # Current values
            current_spread = spread.iloc[-1]
            current_price1 = prices1_aligned.iloc[-1]
            current_price2 = prices2_aligned.iloc[-1]
            
            # Calculate Z-score
            z_score_window = min(self.z_score_window, len(spread))
            if z_score_window < 20:  # Minimum window
                return None
            
            spread_window = spread.tail(z_score_window)
            spread_mean = spread_window.mean()
            spread_std = spread_window.std()
            
            if spread_std == 0:
                current_z_score = 0.0
            else:
                current_z_score = (current_spread - spread_mean) / spread_std
            
            # Calculate moving averages
            ma_short = None
            ma_long = None
            
            if len(spread) >= self.ma_short_window:
                ma_short = spread.tail(self.ma_short_window).mean()
            
            if len(spread) >= self.ma_long_window:
                ma_long = spread.tail(self.ma_long_window).mean()
            
            # Generate signal
            signal_info = self._generate_signal(
                current_z_score, current_spread, ma_short, ma_long
            )
            
            # Calculate additional metrics
            spread_volatility = spread_window.std()
            spread_momentum = self._calculate_momentum(spread)
            correlation = prices1_aligned.corr(prices2_aligned)
            
            # Create result
            result = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'current_price1': float(current_price1),
                'current_price2': float(current_price2),
                'hedge_ratio': float(hedge_ratio),
                'spread_value': float(current_spread),
                'current_z_score': float(current_z_score),
                'spread_mean': float(spread_mean),
                'spread_std': float(spread_std),
                'ma_short': float(ma_short) if ma_short is not None else None,
                'ma_long': float(ma_long) if ma_long is not None else None,
                'signal_type': signal_info['type'],
                'signal_strength': float(signal_info['strength']),
                'has_signal': bool(signal_info['has_signal']),
                'analysis_timestamp': datetime.now(),
                'metadata': {
                    'spread_volatility': float(spread_volatility),
                    'spread_momentum': float(spread_momentum),
                    'correlation': float(correlation),
                    'observations_used': len(common_index),
                    'z_score_window': z_score_window,
                    'original_pair_metadata': pair_info.get('metadata', {})
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.warning(
                f"Error in pair analysis {pair_info['symbol1']}-"
                f"{pair_info['symbol2']}: {e}"
            )
            return None
    
    def _generate_signal(
        self, z_score: float, spread: float, ma_short: Optional[float],
        ma_long: Optional[float]
    ) -> Dict[str, Any]:
        """Generate trading signal based on statistics."""
        abs_z_score = abs(z_score)
        
        # Signal strength based on Z-score magnitude
        if abs_z_score >= 3.0:
            strength = 1.0  # Very strong
        elif abs_z_score >= 2.5:
            strength = 0.8  # Strong
        elif abs_z_score >= 2.0:
            strength = 0.6  # Moderate
        elif abs_z_score >= 1.5:
            strength = 0.4  # Weak
        else:
            strength = 0.0  # No signal
        
        # Determine signal type
        has_signal = abs_z_score >= self.z_score_threshold
        
        if not has_signal:
            return {
                'type': 'hold',
                'strength': 0.0,
                'has_signal': bool(False)
            }
        
        # Additional confirmation from moving averages if available
        ma_confirmation = True
        if ma_short is not None and ma_long is not None:
            if z_score > 0:  # Spread is high, expect mean reversion
                ma_confirmation = ma_short > ma_long  # Trend confirmation
            else:  # Spread is low
                ma_confirmation = ma_short < ma_long
        
        # Adjust strength based on MA confirmation
        if ma_confirmation:
            strength = min(1.0, strength * 1.2)
        else:
            strength = strength * 0.8
        
        # Determine signal direction
        if z_score > self.z_score_threshold:
            signal_type = 'short_spread'  # Short symbol1, Long symbol2
        elif z_score < -self.z_score_threshold:
            signal_type = 'long_spread'   # Long symbol1, Short symbol2
        else:
            signal_type = 'hold'
        
        return {
            'type': signal_type,
            'strength': strength,
            'has_signal': bool(has_signal)
        }
    
    def _calculate_momentum(self, series: pd.Series, window: int = 5) -> float:
        """Calculate momentum of the spread."""
        if len(series) < window + 1:
            return 0.0
        
        recent_values = series.tail(window + 1)
        momentum = (recent_values.iloc[-1] - recent_values.iloc[0]) / window
        return momentum
    
    def _store_hourly_results(self, results: List[Dict[str, Any]]) -> None:
        """Store hourly analysis results."""
        if not results:
            return
        
        try:
            connection = self.db_manager.get_connection()
            
            insert_query = """
                INSERT INTO hourly_statistics (
                    symbol1, symbol2, current_price1, current_price2,
                    hedge_ratio, spread_value, current_z_score,
                    spread_mean, spread_std, ma_short, ma_long,
                    signal_type, signal_strength, has_signal,
                    analysis_timestamp, metadata
                ) VALUES %s
            """
            
            values = []
            for result in results:
                values.append((
                    result['symbol1'],
                    result['symbol2'],
                    result['current_price1'],
                    result['current_price2'],
                    result['hedge_ratio'],
                    result['spread_value'],
                    result['current_z_score'],
                    result['spread_mean'],
                    result['spread_std'],
                    result['ma_short'],
                    result['ma_long'],
                    result['signal_type'],
                    result['signal_strength'],
                    result['has_signal'],
                    result['analysis_timestamp'],
                    json.dumps(result['metadata'])
                ))
            
            with connection.cursor() as cursor:
                from psycopg2.extras import execute_values
                execute_values(cursor, insert_query, values)
            
            connection.commit()
            
            self.logger.info(f"Stored {len(results)} hourly analysis results")
            
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Error storing hourly results: {e}")
            raise
        finally:
            self.db_manager.return_connection(connection)
    
    def create_tables(self) -> None:
        """Create necessary database tables."""
        try:
            connection = self.db_manager.get_connection()
            
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS hourly_statistics (
                    id SERIAL PRIMARY KEY,
                    symbol1 VARCHAR(20) NOT NULL,
                    symbol2 VARCHAR(20) NOT NULL,
                    current_price1 DOUBLE PRECISION NOT NULL,
                    current_price2 DOUBLE PRECISION NOT NULL,
                    hedge_ratio DOUBLE PRECISION NOT NULL,
                    spread_value DOUBLE PRECISION NOT NULL,
                    current_z_score DOUBLE PRECISION NOT NULL,
                    spread_mean DOUBLE PRECISION NOT NULL,
                    spread_std DOUBLE PRECISION NOT NULL,
                    ma_short DOUBLE PRECISION,
                    ma_long DOUBLE PRECISION,
                    signal_type VARCHAR(20) NOT NULL,
                    signal_strength DOUBLE PRECISION NOT NULL,
                    has_signal BOOLEAN DEFAULT false,
                    analysis_timestamp TIMESTAMP NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_hourly_stats_timestamp 
                ON hourly_statistics(analysis_timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_hourly_stats_symbols 
                ON hourly_statistics(symbol1, symbol2);
                
                CREATE INDEX IF NOT EXISTS idx_hourly_stats_signals 
                ON hourly_statistics(has_signal, analysis_timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_hourly_stats_zscore 
                ON hourly_statistics(ABS(current_z_score));
            """
            
            with connection.cursor() as cursor:
                cursor.execute(create_table_sql)
            
            connection.commit()
            self.logger.info("Hourly statistics tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
        finally:
            self.db_manager.return_connection(connection)
