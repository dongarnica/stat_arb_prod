#!/usr/bin/env python3
"""
Complete Statistical Arbitrage Analytics with Database Storage.

This script:
1. Gets available symbols from the database
2. Calculates comprehensive pair statistics including Z-scores
3. Writes all results to the pair_statistics table
4. Provides detailed logging and error handling
"""

import sys
import os
import logging
import warnings
from datetime import date, timedelta
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
# from scipy import stats  # Commented out as not used
import psycopg2
from psycopg2.extras import RealDictCursor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('statistical_arbitrage_analytics.log')
        ]
    )

class StatisticalArbitrageAnalytics:
    """
    Complete statistical arbitrage analytics engine.
    """
    
    def __init__(self):
        """Initialize the analytics engine."""
        self.logger = logging.getLogger(__name__)
          # Import components
        from analytics.stats.configuration_manager import ConfigurationManager
        from analytics.stats.data_manager import DataManager
        
        self.config = ConfigurationManager()
        self.data_manager = DataManager(self.config)
        
        # Database connection for writing results
        self.db_connection = None
        self._init_database_connection()
        
        self.logger.info("Statistical Arbitrage Analytics initialized")
    
    def _init_database_connection(self):
        """Initialize database connection for writing results."""
        try:
            db_config = self.config.get_database_config()
            self.db_connection = psycopg2.connect(
                host=db_config['host'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                port=db_config['port']
            )
            self.logger.info("Database connection established for writing results")
        except Exception as e:
            self.logger.error(f"Failed to establish database connection: {e}")
            raise
    
    def calculate_z_score(self, spread_series: pd.Series, window: int = 252) -> float:
        """
        Calculate the current Z-score of the spread.
        
        Parameters:
        -----------
        spread_series : pd.Series
            Time series of spread values
        window : int
            Rolling window for Z-score calculation
            
        Returns:
        --------
        float
            Current Z-score
        """
        if len(spread_series) < window:
            window = len(spread_series)
        
        if window < 2:
            return 0.0
        
        # Use rolling window for mean and std
        rolling_mean = spread_series.rolling(window=window).mean()
        rolling_std = spread_series.rolling(window=window).std()
        
        # Current Z-score
        current_spread = spread_series.iloc[-1]
        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]
        
        if current_std == 0 or pd.isna(current_std):
            return 0.0
        
        z_score = (current_spread - current_mean) / current_std
        return float(z_score)
    
    def calculate_half_life(self, spread_series: pd.Series) -> Optional[float]:
        """
        Calculate half-life of mean reversion.
        
        Parameters:
        -----------
        spread_series : pd.Series
            Time series of spread values
            
        Returns:
        --------
        float or None
            Half-life in days
        """
        try:
            # Calculate log spread
            log_spread = np.log(np.abs(spread_series) + 1e-8)
            
            # Lag the series
            lagged_spread = log_spread.shift(1).dropna()
            current_spread = log_spread[1:]
            
            # Align series
            aligned_data = pd.DataFrame({
                'current': current_spread,
                'lagged': lagged_spread
            }).dropna()
            
            if len(aligned_data) < 10:
                return None
            
            # OLS regression: spread_t = alpha + beta * spread_{t-1} + error
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(
                aligned_data['lagged'], aligned_data['current']
            )
              # Half-life calculation
            if slope >= 1 or slope <= 0:
                return None
            
            half_life = -np.log(2) / np.log(slope)
            # Reasonable bounds (1 day to 2 years)
            if 1 <= half_life <= 730:
                return float(half_life)            
            return None
            
        except Exception as e:
            self.logger.debug(f"Half-life calculation failed: {e}")
            return None

    def test_cointegration(self, price1: pd.Series, price2: pd.Series) -> Tuple[float, bool]:
        """
        Test for cointegration using Engle-Granger test with proper statistical validation.
        
        This method requires statsmodels to be installed and will raise an error
        if it's not available, ensuring no fallback to correlation-based methods.
        
        Parameters:
        -----------
        price1, price2 : pd.Series
            Price series for the two assets
            
        Returns:
        --------
        Tuple[float, bool]
            P-value and whether series are cointegrated
            
        Raises:
        -------
        ImportError
            If statsmodels is not available
        ValueError
            If price series are invalid
        """
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError as e:
            self.logger.error("statsmodels is required for cointegration testing but is not available")
            raise ImportError(
                "statsmodels is required for cointegration analysis. "
                "Please install it with: pip install statsmodels"
            ) from e
        
        try:
            # Align series and ensure we have enough data
            aligned_data = pd.DataFrame({'p1': price1, 'p2': price2}).dropna()
            
            if len(aligned_data) < 50:  # Need more data for reliable analysis
                self.logger.debug(f"Insufficient data for cointegration test: {len(aligned_data)} points")
                return 0.99, False
            
            # Extract aligned price series
            p1 = aligned_data['p1'].values
            p2 = aligned_data['p2'].values
              # Validate input data
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                raise ValueError("Price series cannot contain NaN values after alignment")
            
            # Check correlation before cointegration test to filter out negative correlations
            correlation = aligned_data['p1'].corr(aligned_data['p2'])
            min_correlation_threshold = 0.1  # Minimum positive correlation required
            
            if correlation < min_correlation_threshold:
                self.logger.debug(
                    f"Correlation {correlation:.6f} below minimum threshold {min_correlation_threshold}. "
                    f"Skipping cointegration test to avoid negative-correlation cointegrated pairs."
                )
                return 0.99, False
            
            # Perform Engle-Granger cointegration test
            # Returns: (test_statistic, p_value, critical_values)
            coint_result = coint(p1, p2)
            test_statistic = coint_result[0]
            raw_p_value = coint_result[1]
            critical_values = coint_result[2]
              # Debug logging to track p-value calculation
            self.logger.debug(
                f"Cointegration test - Correlation: {correlation:.6f}, "
                f"Raw p-value: {raw_p_value:.10f}, "
                f"Test stat: {test_statistic:.6f}, "
                f"Critical values: {critical_values}"
            )
            
            # Use the actual p-value from the test
            # Only apply a floor if the p-value is extremely small (computational precision issues)
            if raw_p_value < 1e-16:
                self.logger.warning(
                    f"Extremely small p-value {raw_p_value:.2e} likely due to "
                    f"computational precision, using 1e-16"
                )
                p_value = 1e-16
            else:
                p_value = raw_p_value
            
            # Standard threshold for cointegration (5% significance level)
            is_cointegrated = (p_value < 0.05)
            
            # Additional validation for cointegrated pairs with negative correlation
            if is_cointegrated and correlation < 0:
                self.logger.warning(
                    f"VALIDATION ALERT: Cointegrated pair with negative correlation! "
                    f"Correlation: {correlation:.6f}, P-value: {p_value:.10f}. "
                    f"This should not happen with correlation filter in place."
                )                # Force rejection of negative correlation cointegrated pairs
                is_cointegrated = False
                p_value = 0.99
            
            self.logger.debug(
                f"Cointegration result - Final p-value: {p_value:.10f}, "
                f"Cointegrated: {is_cointegrated}, Correlation: {correlation:.6f}"
            )
            
            return float(p_value), is_cointegrated
            
        except Exception as e:
            self.logger.error(f"Engle-Granger cointegration test failed: {e}")
            return 0.99, False
    
    def calculate_signal_strength(self, z_score: float, correlation: float, 
                                  p_value: float, data_points: int, 
                                  current_spread: float, mean_spread: float, 
                                  entry_threshold: float = 2.0) -> float:
        """
        Calculate overall signal strength with mean-reversion directionality.
        
        For mean-reversion strategies, signal strength should be positive when:
        - Current spread is above mean and z_score is positive (expect reversion down)
        - Current spread is below mean and z_score is negative (expect reversion up)
        
        Parameters:
        -----------
        z_score : float
            Current Z-score
        correlation : float
            Correlation coefficient  
        p_value : float
            Cointegration p-value
        data_points : int
            Number of data points
        current_spread : float
            Current spread value
        mean_spread : float
            Historical mean of the spread
        entry_threshold : float
            Z-score threshold for entries (default 2.0)
            
        Returns:
        --------
        float
            Signal strength between 0 and 1, reflecting mean-reversion opportunity
        """
        # Z-score component (higher absolute Z-score = stronger signal)
        z_component = min(abs(z_score) / 3.0, 1.0)
        
        # Correlation component (higher correlation = stronger)
        corr_component = abs(correlation)
        
        # Cointegration component (lower p-value = stronger)
        coint_component = max(0, 1 - p_value)
        
        # Data quality component (more data = more reliable)
        data_component = min(data_points / 252.0, 1.0)
        
        # Mean-reversion directionality factor
        # This ensures the signal aligns with mean reversion opportunity
        spread_deviation = current_spread - mean_spread
        
        # Check if z_score and spread deviation have the same sign (both positive or both negative)
        # This indicates a proper mean-reversion setup
        if (z_score > 0 and spread_deviation > 0) or (z_score < 0 and spread_deviation < 0):
            # Aligned: current position suggests mean reversion opportunity
            directional_factor = 1.0
        elif abs(z_score) < 0.5:
            # Near equilibrium, directional factor is neutral
            directional_factor = 0.8
        else:
            # Misaligned: z_score and spread suggest conflicting directions
            directional_factor = 0.3
            
        # Entry strength based on how far we are from entry threshold
        if abs(z_score) >= entry_threshold:
            entry_factor = 1.0
        else:
            entry_factor = abs(z_score) / entry_threshold
        
        # Weighted average with directional adjustment
        base_strength = (
            0.35 * z_component +
            0.25 * corr_component +
            0.25 * coint_component +
            0.15 * data_component
        )
        
        # Apply directional and entry factors
        signal_strength = base_strength * directional_factor * entry_factor
        
        return min(signal_strength, 1.0)
    
    def analyze_pair(self, symbol1: str, symbol2: str, 
                     start_date: date, end_date: date) -> Optional[Dict[str, Any]]:
        """
        Perform complete statistical analysis for a pair.
        
        Parameters:
        -----------
        symbol1, symbol2 : str
            Asset symbols
        start_date, end_date : date
            Date range for analysis
            
        Returns:
        --------
        Dict[str, Any] or None
            Complete analysis results
        """
        try:
            # Get pair data
            pair_data = self.data_manager.get_pair_data(symbol1, symbol2, start_date, end_date)
            
            if pair_data.empty:
                self.logger.warning(f"No data for pair {symbol1}/{symbol2}")
                return None
              # Extract price series
            price1 = pair_data['asset_1_close']
            price2 = pair_data['asset_2_close']
            
            # Calculate hedge ratio using regression (price1 = beta * price2 + alpha)
            # This gives us the optimal hedge ratio for the spread calculation
            from sklearn.linear_model import LinearRegression
            import numpy as np
            
            # Reshape for sklearn
            X = price2.values.reshape(-1, 1)
            y = price1.values
            
            # Fit linear regression to get hedge ratio
            reg = LinearRegression().fit(X, y)
            hedge_ratio = reg.coef_[0]
            
            # Calculate spread using hedge ratio: spread = price1 - hedge_ratio * price2
            spread = price1 - hedge_ratio * price2
            
            self.logger.debug(
                f"Pair {symbol1}/{symbol2} - Hedge ratio: {hedge_ratio:.6f}, "
                f"Spread mean: {spread.mean():.6f}, Spread std: {spread.std():.6f}"
            )
            
            # Core Z-score calculation
            z_score = self.calculate_z_score(spread)
            
            # Basic statistics
            correlation = price1.corr(price2)
            current_spread = float(spread.iloc[-1])
            mean_spread = float(spread.mean())
            std_spread = float(spread.std())            # Cointegration test
            coint_pvalue, is_cointegrated = self.test_cointegration(price1, price2)
            
            # Debug logging for p-value tracking
            self.logger.debug(
                f"Pair {symbol1}/{symbol2} - Cointegration p-value: {coint_pvalue:.10f}, "
                f"Cointegrated: {is_cointegrated}"
            )
            
            # Half-life calculation
            half_life = self.calculate_half_life(spread)            # Volatility metrics - dynamically calculated from price returns
            spread_volatility = float(spread.std())
            vol1 = price1.pct_change().std()  # Asset 1 price volatility
            vol2 = price2.pct_change().std()  # Asset 2 price volatility
            
            # Volatility ratio: ratio of asset 1 to asset 2 volatility
            if vol2 > 0:
                volatility_ratio = float(vol1 / vol2)
            else:
                self.logger.warning(f"Asset 2 ({symbol2}) has zero volatility, setting ratio to 1.0")
                volatility_ratio = 1.0
            
            self.logger.debug(
                f"Volatility analysis - Asset 1: {vol1:.6f}, Asset 2: {vol2:.6f}, "
                f"Ratio: {volatility_ratio:.6f}, Spread vol: {spread_volatility:.6f}"
            )
              # Signal strength
            signal_strength = self.calculate_signal_strength(
                z_score, correlation, coint_pvalue, len(pair_data),
                current_spread, mean_spread
            )
            
            # Enhanced logging for signal strength validation
            self.logger.debug(
                f"Signal strength calculation for {symbol1}/{symbol2}: "
                f"z_score={z_score:.4f}, signal_strength={signal_strength:.6f}"
            )
            
            # Validate signal strength bounds
            if signal_strength > 1.0:
                self.logger.error(
                    f"CRITICAL ERROR: Signal strength {signal_strength:.6f} exceeds 1.0 for {symbol1}/{symbol2}! "
                    f"z_score={z_score:.4f}, correlation={correlation:.4f}, "
                    f"p_value={coint_pvalue:.6f}, data_points={len(pair_data)}"
                )
                # Force bound to prevent invalid data
                signal_strength = 1.0
            
            if signal_strength < 0.0:
                self.logger.error(
                    f"CRITICAL ERROR: Signal strength {signal_strength:.6f} is negative for {symbol1}/{symbol2}!"
                )
                # Force to zero to prevent invalid data
                signal_strength = 0.0
              # Confidence level (based on data quality and cointegration)
            confidence_level = 0.5
            if is_cointegrated:
                confidence_level += 0.3
            if len(pair_data) >= 252:
                confidence_level += 0.2
            confidence_level = min(confidence_level, 1.0)
            
            # Calculate returns for Sharpe ratio
            returns = spread.pct_change().dropna()
            if len(returns) > 0 and returns.std() != 0:
                sharpe_ratio = float(returns.mean() / returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0.0
                
            # Mean reversion speed
            mean_reversion_speed = 1.0 / half_life if half_life else None
            
            # Count Z-score breaches
            zscore_breach_counts = self.count_zscore_breaches(spread)
            
            # Log and assert breach counts before saving to validate correctness
            self.logger.debug(
                f"Final Z-score breach counts before saving - "
                f"Above 2.0: {zscore_breach_counts['zscore_breach_count_above_2']}, "
                f"Below -2.0: {zscore_breach_counts['zscore_breach_count_below_neg2']}"
            )
            
            # Assert that breach counts are non-negative and reasonable
            assert zscore_breach_counts['zscore_breach_count_above_2'] >= 0, \
                "Z-score breach count above 2.0 cannot be negative"
            assert zscore_breach_counts['zscore_breach_count_below_neg2'] >= 0, \
                "Z-score breach count below -2.0 cannot be negative"
            
            # Use current timestamp for calculation_date to ensure uniqueness
            from datetime import datetime
            calculation_timestamp = datetime.now()
            
            return {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'calculation_date': calculation_timestamp.date(),
                'z_score': float(z_score),
                'current_spread': float(current_spread),
                'mean_spread': float(mean_spread),
                'std_spread': float(std_spread),
                'correlation': float(correlation),
                'cointegration_pvalue': float(coint_pvalue),
                'is_cointegrated': bool(is_cointegrated),  # Convert numpy.bool_ to Python bool
                'half_life_days': float(half_life) if half_life else None,
                'mean_reversion_speed': float(mean_reversion_speed) if mean_reversion_speed else None,
                'spread_volatility': float(spread_volatility),
                'volatility_ratio': float(volatility_ratio),
                'signal_strength': float(signal_strength),                'confidence_level': float(confidence_level),
                'sharpe_ratio': float(sharpe_ratio),
                'data_points_used': int(len(pair_data)),
                'data_start_date': pair_data.index.min().date(),
                'data_end_date': pair_data.index.max().date(),
                'zscore_breach_count_above_2': zscore_breach_counts['zscore_breach_count_above_2'],
                'zscore_breach_count_below_neg2': zscore_breach_counts['zscore_breach_count_below_neg2']
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol1}/{symbol2}: {e}")
            return None
    
    def save_results_to_database(self, results: Dict[str, Any]) -> bool:
        """
        Save analysis results to the database.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Analysis results
            
        Returns:
        --------
        bool
            Success status
        """
        try:
            cursor = self.db_connection.cursor()
            # Insert or update results
            insert_query = """INSERT INTO pair_statistics (
                symbol1, symbol2, calculation_date, z_score, current_spread,
                mean_spread, std_spread, correlation, cointegration_pvalue,
                is_cointegrated, half_life_days, mean_reversion_speed,
                spread_volatility, volatility_ratio, signal_strength,
                confidence_level, sharpe_ratio, data_points_used,
                data_start_date, data_end_date, zscore_breach_count_above_2,
                zscore_breach_count_below_neg2
            ) VALUES (
                %(symbol1)s, %(symbol2)s, %(calculation_date)s, %(z_score)s,
                %(current_spread)s, %(mean_spread)s, %(std_spread)s,
                %(correlation)s, %(cointegration_pvalue)s, %(is_cointegrated)s,
                %(half_life_days)s, %(mean_reversion_speed)s, %(spread_volatility)s,
                %(volatility_ratio)s, %(signal_strength)s, %(confidence_level)s,
                %(sharpe_ratio)s, %(data_points_used)s, %(data_start_date)s,
                %(data_end_date)s, %(zscore_breach_count_above_2)s,
                %(zscore_breach_count_below_neg2)s
            ) ON CONFLICT (symbol1, symbol2, calculation_date)
            DO UPDATE SET
                z_score = EXCLUDED.z_score,
                current_spread = EXCLUDED.current_spread,
                mean_spread = EXCLUDED.mean_spread,
                std_spread = EXCLUDED.std_spread,
                correlation = EXCLUDED.correlation,
                cointegration_pvalue = EXCLUDED.cointegration_pvalue,
                is_cointegrated = EXCLUDED.is_cointegrated,
                half_life_days = EXCLUDED.half_life_days,
                mean_reversion_speed = EXCLUDED.mean_reversion_speed,
                spread_volatility = EXCLUDED.spread_volatility,
                volatility_ratio = EXCLUDED.volatility_ratio,
                signal_strength = EXCLUDED.signal_strength,
                confidence_level = EXCLUDED.confidence_level,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                data_points_used = EXCLUDED.data_points_used,
                data_start_date = EXCLUDED.data_start_date,
                data_end_date = EXCLUDED.data_end_date,
                zscore_breach_count_above_2 =\
                    EXCLUDED.zscore_breach_count_above_2,
                zscore_breach_count_below_neg2 =\
                    EXCLUDED.zscore_breach_count_below_neg2,
                last_updated = CURRENT_TIMESTAMP"""
            
            cursor.execute(insert_query, results)
            self.db_connection.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            self.db_connection.rollback()
            return False

    def run_full_analysis(self, max_pairs: Optional[int] = 50, min_correlation: float = 0.5) -> None:
        """
        Run complete analysis on available symbols.
        
        Parameters:
        -----------
        max_pairs : Optional[int]
            Maximum number of pairs to analyze. If None, analyze all pairs.
        min_correlation : float
            Minimum correlation threshold for analysis
        """
        self.logger.info("Starting full statistical arbitrage analysis...")
        
        # Force cache refresh for today's calculations
        self.logger.info("Clearing stale cache entries to ensure fresh calculations...")
        self.data_manager.clear_today_cache()
        self.data_manager.clear_stale_cache(max_age_hours=1)
        
        # Get available symbols
        available_symbols = self.data_manager.get_available_symbols()
        self.logger.info(f"Found {len(available_symbols)} symbols in database")
        
        if len(available_symbols) < 2:
            self.logger.error("Need at least 2 symbols for pair analysis")
            return
        
        # Process ALL available symbols if max_pairs is None
        if max_pairs is None:
            symbols_to_analyze = available_symbols
            self.logger.info(f"Analyzing ALL {len(symbols_to_analyze)} symbols")
        else:
            # Use a reasonable subset for analysis
            if len(available_symbols) > 20:
                # Take the first 20 symbols for comprehensive analysis
                symbols_to_analyze = available_symbols[:20]
                self.logger.info(f"Analyzing top {len(symbols_to_analyze)} symbols")
            else:
                symbols_to_analyze = available_symbols
        
        # Generate ALL possible pairs
        pairs_to_analyze = []
        for i in range(len(symbols_to_analyze)):
            for j in range(i + 1, len(symbols_to_analyze)):
                pairs_to_analyze.append((symbols_to_analyze[i], symbols_to_analyze[j]))
                # Only limit pairs if max_pairs is specified
                if max_pairs is not None and len(pairs_to_analyze) >= max_pairs:
                    break
            if max_pairs is not None and len(pairs_to_analyze) >= max_pairs:
                break
        
        self.logger.info(f"Analyzing {len(pairs_to_analyze)} pairs")
        
        # Analysis parameters
        end_date = date.today()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        successful_analyses = 0
        saved_results = 0
        
        for idx, (symbol1, symbol2) in enumerate(pairs_to_analyze, 1):
            self.logger.info(f"Processing pair {idx}/{len(pairs_to_analyze)}: {symbol1}/{symbol2}")
            
            try:
                # Perform analysis
                results = self.analyze_pair(symbol1, symbol2, start_date, end_date)
                
                if results:
                    successful_analyses += 1
                    
                    # Log key metrics
                    self.logger.info(
                        f"  Z-score: {results['z_score']:.3f}, "
                        f"Correlation: {results['correlation']:.3f}, "
                        f"Signal: {results['signal_strength']:.3f}, "
                        f"Cointegrated: {results['is_cointegrated']}"
                    )
                    
                    # Save to database
                    if self.save_results_to_database(results):
                        saved_results += 1
                    else:
                        self.logger.warning(f"Failed to save results for {symbol1}/{symbol2}")
                else:
                    self.logger.warning(f"Analysis failed for {symbol1}/{symbol2}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol1}/{symbol2}: {e}")
        
        self.logger.info("=" * 60)
        self.logger.info("ANALYSIS SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total pairs processed: {len(pairs_to_analyze)}")
        self.logger.info(f"Successful analyses: {successful_analyses}")
        self.logger.info(f"Results saved to database: {saved_results}")
        
        # Query and display top signals        self.display_top_signals()
        
    def display_top_signals(self, limit: int = 10) -> None:
        """Display top trading signals from the database."""
        try:
            cursor = self.db_connection.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                symbol1, symbol2, z_score, correlation, signal_strength,
                confidence_level, is_cointegrated, half_life_days,
                zscore_breach_count_above_2, zscore_breach_count_below_neg2,
                CASE 
                    WHEN ABS(z_score) >= 2.0 THEN 'STRONG_SIGNAL'
                    WHEN ABS(z_score) >= 1.5 THEN 'MEDIUM_SIGNAL'
                    WHEN ABS(z_score) >= 1.0 THEN 'WEAK_SIGNAL'
                    ELSE 'NO_SIGNAL'
                END as signal_category,
                CASE 
                    WHEN z_score > 0 THEN 'SHORT_' || symbol1 || '_LONG_' || symbol2
                    WHEN z_score < 0 THEN 'LONG_' || symbol1 || '_SHORT_' || symbol2
                    ELSE 'NEUTRAL'
                END as trade_direction
            FROM pair_statistics 
            WHERE calculation_date = CURRENT_DATE
            ORDER BY ABS(z_score) DESC, signal_strength DESC
            LIMIT %s
            """
            
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            
            if results:
                self.logger.info("=" * 80)
                self.logger.info("TOP TRADING SIGNALS")
                self.logger.info("=" * 80)
                
                for i, row in enumerate(results, 1):
                    self.logger.info(f"{i:2d}. {row['symbol1']}/{row['symbol2']}")
                    self.logger.info(f"    Z-Score: {row['z_score']:8.3f} | {row['signal_category']}")
                    self.logger.info(f"    Direction: {row['trade_direction']}")
                    self.logger.info(f"    Correlation: {row['correlation']:6.3f} | Signal: {row['signal_strength']:5.3f}")
                    self.logger.info(f"    Cointegrated: {row['is_cointegrated']} | Half-life: {row['half_life_days']} days")
                    
                    # Display Z-Score breach counts if available
                    if 'zscore_breach_count_above_2' in row and 'zscore_breach_count_below_neg2' in row:
                        self.logger.info(f"    Z-Score Breaches: Above 2.0: {row['zscore_breach_count_above_2']}, "
                                        f"Below -2.0: {row['zscore_breach_count_below_neg2']}")
                    
                    self.logger.info("")
            else:
                self.logger.info("No results found in database")
                
        except Exception as e:
            self.logger.error(f"Failed to display top signals: {e}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.db_connection:
            self.db_connection.close()
        self.data_manager.cleanup()

    def count_zscore_breaches(self, spread_series: pd.Series, window: int = 252,
                           upper_threshold: float = 2.0, lower_threshold: float = -2.0) -> Dict[str, int]:
        """
        Count breaches of Z-score above upper threshold and below lower threshold over the entire historical series.
        
        This method calculates a rolling Z-score for each point in the series and counts how many times
        the Z-score exceeds the thresholds throughout the entire history.
        
        Parameters:
        -----------
        spread_series : pd.Series
            Time series of spread values
        window : int
            Rolling window for Z-score calculation (default: 252 trading days)
        upper_threshold : float
            Upper threshold for breach count (default: 2.0)
        lower_threshold : float
            Lower threshold for breach count (default: -2.0)
            
        Returns:
        --------
        Dict[str, int]
            Dictionary with breach counts above and below thresholds
        """
        if len(spread_series) < window:
            window = len(spread_series)
        
        if window < 2 or len(spread_series) < 10:  # Need minimum data for meaningful analysis
            self.logger.debug(f"Insufficient data for Z-score breach counting: {len(spread_series)} points")
            return {
                'zscore_breach_count_above_2': 0,
                'zscore_breach_count_below_neg2': 0
            }
        
        # Calculate rolling Z-scores for the entire series
        # Use a minimum period to ensure we have meaningful data early in the series
        min_periods = max(10, window // 4)  # At least 10 points, or 1/4 of window
        
        rolling_mean = spread_series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = spread_series.rolling(window=window, min_periods=min_periods).std()
        
        # Calculate Z-scores more robustly
        z_scores = pd.Series(index=spread_series.index, dtype=float)
        
        for i in range(len(spread_series)):
            current_mean = rolling_mean.iloc[i]
            current_std = rolling_std.iloc[i]
            
            if (pd.notna(current_mean) and pd.notna(current_std) and 
                current_std > 1e-8):  # Valid statistics
                z_scores.iloc[i] = ((spread_series.iloc[i] - current_mean) / current_std)
            else:
                z_scores.iloc[i] = 0.0
                
        # Remove NaN values and ensure we have valid data
        valid_z_scores = z_scores.dropna()
        
        if len(valid_z_scores) == 0:
            self.logger.warning("No valid z-scores calculated for breach counting")
            return {
                'zscore_breach_count_above_2': 0,
                'zscore_breach_count_below_neg2': 0
            }
        
        # Count breaches over the entire historical series
        above_threshold_count = (valid_z_scores > upper_threshold).sum()
        below_threshold_count = (valid_z_scores < lower_threshold).sum()
        
        # Enhanced logging for debugging
        max_z = valid_z_scores.max() if len(valid_z_scores) > 0 else 0
        min_z = valid_z_scores.min() if len(valid_z_scores) > 0 else 0
        
        self.logger.debug(
            f"Z-score breach analysis - Total valid points: {len(valid_z_scores)}/{len(spread_series)}, "
            f"Above {upper_threshold}: {above_threshold_count}, "
            f"Below {lower_threshold}: {below_threshold_count}, "
            f"Max Z-score: {max_z:.3f}, Min Z-score: {min_z:.3f}, "
            f"Z-score std: {valid_z_scores.std():.3f}"
        )
        
        # Validate reasonableness of breach counts
        total_points = len(valid_z_scores)
        if total_points > 0:
            above_pct = (above_threshold_count / total_points) * 100
            below_pct = (below_threshold_count / total_points) * 100
            
            # Log percentages for insight
            self.logger.debug(
                f"Breach percentages - Above {upper_threshold}: {above_pct:.1f}%, "
                f"Below {lower_threshold}: {below_pct:.1f}%"
            )
            
            # For a normal distribution, expect ~2.3% above 2.0 and ~2.3% below -2.0
            # But financial data often has fat tails, so we allow higher percentages
            if above_pct > 50 or below_pct > 50:
                self.logger.warning(
                    f"Abnormally high breach percentages: Above {upper_threshold}: {above_pct:.1f}%, "
                    f"Below {lower_threshold}: {below_pct:.1f}%"
                )
        
        result = {
            'zscore_breach_count_above_2': int(above_threshold_count),
            'zscore_breach_count_below_neg2': int(below_threshold_count)
        }
        
        # Final validation
        self.logger.debug(f"Final breach counts: {result}")
        assert result['zscore_breach_count_above_2'] >= 0, "Above threshold count cannot be negative"
        assert result['zscore_breach_count_below_neg2'] >= 0, "Below threshold count cannot be negative"
        
        # Additional sanity check: if we have a lot of data, we should see some breaches
        if total_points > 100 and result['zscore_breach_count_above_2'] == 0 and result['zscore_breach_count_below_neg2'] == 0:
            self.logger.warning(
                f"Suspicious: {total_points} data points but zero breaches. "
                f"Max Z: {max_z:.3f}, Min Z: {min_z:.3f}"
            )
        
        return result

def main():
    """Main function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:        # Initialize analytics engine
        analytics = StatisticalArbitrageAnalytics()
        
        # Run full analysis - process ALL available pairs
        analytics.run_full_analysis(max_pairs=None)  # Analyze ALL pairs
        
        logger.info("Statistical arbitrage analysis completed successfully!")
        
        # Cleanup
        analytics.cleanup()
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
