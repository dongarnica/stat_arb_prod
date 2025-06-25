"""
OLS Hedge Ratio Calculation for Pair Trading.

This module calculates optimal hedge ratios using Ordinary Least Squares
regression, which determines the appropriate position sizes for pair trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
from ..base_pair_statistic import BasePairStatistic


class OLSHedgeRatio(BasePairStatistic):
    """
    Calculate hedge ratios using OLS regression for pair trading.
    
    The hedge ratio determines the optimal position size relationship
    between two assets in a pairs trade. A proper hedge ratio helps:
    - Minimize portfolio variance
    - Create market-neutral positions
    - Optimize risk-adjusted returns
    """
    
    # Required class attributes
    name = "ols_hedge_ratio"
    description = "OLS regression hedge ratio for pair trading"
    category = "pair_trading"
    asset_columns = ["asset_1_close", "asset_2_close"]
    required_columns = ["asset_1_close", "asset_2_close"]
    lookback_period = 252
    
    def __init__(self, use_returns: bool = False, **kwargs):
        """
        Initialize OLS hedge ratio calculation.
        
        Parameters:
        -----------
        use_returns : bool, default False
            If True, use price returns instead of price levels
        """
        self.use_returns = use_returns
        super().__init__(**kwargs)
    
    def calculate(self, data1: pd.DataFrame, data2: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate OLS hedge ratio for pair of assets.
        
        Parameters:
        -----------
        data1 : pd.DataFrame
            First asset's data
        data2 : pd.DataFrame  
            Second asset's data
        **kwargs
            Additional parameters
            
        Returns:
        --------
        Dict[str, Any]
            Hedge ratio calculation results
        """
        try:
            # Create combined dataframe for internal calculation
            data = pd.DataFrame({
                'asset_1_close': data1['close'] if 'close' in data1.columns else data1.iloc[:, 0],
                'asset_2_close': data2['close'] if 'close' in data2.columns else data2.iloc[:, 0]
            })
            
            # Call the existing calculate logic
            return self._calculate(data)
            
        except Exception as e:
            return {
                "error": str(e),
                "statistic_name": self.name,
                "symbol_1": kwargs.get('symbol1', 'unknown'),
                "symbol_2": kwargs.get('symbol2', 'unknown')
            }

    def _calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate OLS hedge ratio and related statistics.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with required asset columns
            
        Returns:
        --------
        Dict[str, Any]
            Hedge ratio statistics and regression results
        """
        try:
            # Extract price series
            prices = data[self.asset_columns].dropna()
            
            if len(prices) < 20:
                return self._create_error_result(
                    "Insufficient data for hedge ratio calculation",
                    {"min_required": 20, "actual": len(prices)}
                )
            
            # Prepare data for regression
            if self.use_returns:
                # Use returns for regression
                returns = prices.pct_change().dropna()
                if len(returns) < 10:
                    return self._create_error_result(
                        "Insufficient return data",
                        {"returns_length": len(returns)}
                    )
                y = returns.iloc[:, 0].values  # Asset 1 returns
                x = returns.iloc[:, 1].values  # Asset 2 returns
                data_type = "returns"
            else:
                # Use price levels for regression
                y = prices.iloc[:, 0].values  # Asset 1 prices
                x = prices.iloc[:, 1].values  # Asset 2 prices
                data_type = "prices"
            
            # Perform OLS regression: y = alpha + beta * x
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                X = x.reshape(-1, 1)
                reg = LinearRegression().fit(X, y)
                
                hedge_ratio = reg.coef_[0]  # Beta coefficient
                alpha = reg.intercept_      # Alpha (intercept)
                
                # Calculate predictions and residuals
                y_pred = reg.predict(X)
                residuals = y - y_pred
                
                # Calculate R-squared and other fit statistics
                r_squared = r2_score(y, y_pred)
                
            # Calculate additional hedge ratio statistics
            hedge_stats = self._calculate_hedge_ratio_stats(
                hedge_ratio, residuals, r_squared, len(prices))
              # Calculate spread using the hedge ratio
            if self.use_returns:
                spread_series = (returns.iloc[:, 0] - 
                               hedge_ratio * returns.iloc[:, 1])
            else:
                spread_series = (prices.iloc[:, 0] - 
                               hedge_ratio * prices.iloc[:, 1])
            
            spread_stats = self._calculate_spread_stats(spread_series)
            
            # Trading signals based on current spread
            trading_signals = self._generate_trading_signals(
                spread_series, hedge_ratio, hedge_stats["confidence"])
            
            return {
                # Primary results
                "hedge_ratio": round(hedge_ratio, 6),
                "alpha": round(alpha, 6),
                "r_squared": round(r_squared, 4),
                
                # Hedge ratio statistics
                **hedge_stats,
                
                # Spread statistics
                **spread_stats,
                
                # Trading signals
                **trading_signals,
                
                # Metadata
                "data_type": data_type,
                "sample_size": len(prices),
                "use_returns": self.use_returns
            }
            
        except Exception as e:
            return self._create_error_result(
                f"OLS hedge ratio calculation failed: {str(e)}",
                {"use_returns": self.use_returns}
            )
    
    def _calculate_hedge_ratio_stats(self, hedge_ratio: float, 
                                   residuals: np.ndarray, 
                                   r_squared: float, 
                                   n_obs: int) -> Dict[str, Any]:
        """Calculate hedge ratio statistics and confidence measures."""
        try:
            # Standard error of hedge ratio
            residual_std = np.std(residuals)
            
            # Confidence score based on R-squared and sample size
            confidence = min(100.0, r_squared * 100 * np.sqrt(n_obs / 252))
            
            # Hedge ratio quality assessment
            quality = self._assess_hedge_ratio_quality(r_squared, confidence)
            
            # Residual analysis
            residual_mean = np.mean(residuals)
            residual_std_error = residual_std / np.sqrt(n_obs)
            
            return {
                "confidence": round(confidence, 2),
                "hedge_ratio_quality": quality,
                "residual_std": round(residual_std, 6),
                "residual_mean": round(residual_mean, 6),
                "residual_std_error": round(residual_std_error, 6),
                "tracking_error": round(residual_std * np.sqrt(252), 6)
            }
            
        except Exception:
            return {
                "confidence": 0.0,
                "hedge_ratio_quality": "poor",
                "residual_std": 0.0,
                "residual_mean": 0.0,
                "residual_std_error": 0.0,
                "tracking_error": 0.0
            }
    
    def _calculate_spread_stats(self, spread: pd.Series) -> Dict[str, Any]:
        """Calculate spread statistics."""
        try:
            return {
                "spread_mean": round(spread.mean(), 6),
                "spread_std": round(spread.std(), 6),
                "spread_min": round(spread.min(), 6),
                "spread_max": round(spread.max(), 6),
                "current_spread": round(spread.iloc[-1], 6),
                "spread_zscore": round(
                    (spread.iloc[-1] - spread.mean()) / spread.std(), 4),
                "spread_percentile": round(
                    (spread <= spread.iloc[-1]).mean() * 100, 2)
            }
            
        except Exception:
            return {
                "spread_mean": 0.0,
                "spread_std": 0.0,
                "spread_min": 0.0,
                "spread_max": 0.0,
                "current_spread": 0.0,
                "spread_zscore": 0.0,
                "spread_percentile": 50.0
            }
    
    def _generate_trading_signals(self, spread: pd.Series, 
                                hedge_ratio: float, 
                                confidence: float) -> Dict[str, Any]:
        """Generate trading signals based on spread analysis."""
        try:
            current_spread = spread.iloc[-1]
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Z-score for signal generation
            zscore = (current_spread - spread_mean) / spread_std
            
            # Signal thresholds (can be made configurable)
            entry_threshold = 2.0
            exit_threshold = 0.5
            
            # Generate signals
            if abs(zscore) > entry_threshold and confidence > 60:
                if zscore > 0:
                    signal = "sell_spread"  # Spread too high, expect reversion
                    position = f"short asset1, long asset2 (ratio {hedge_ratio:.3f})"
                else:
                    signal = "buy_spread"   # Spread too low, expect reversion
                    position = f"long asset1, short asset2 (ratio {hedge_ratio:.3f})"
                signal_strength = min(100, abs(zscore) / entry_threshold * 100)
            elif abs(zscore) < exit_threshold:
                signal = "exit_position"
                position = "close existing positions"
                signal_strength = max(0, 100 - abs(zscore) / exit_threshold * 100)
            else:
                signal = "hold"
                position = "maintain current position or wait"
                signal_strength = 0.0
            
            return {
                "trading_signal": signal,
                "position_recommendation": position,
                "signal_strength": round(signal_strength, 2),
                "entry_threshold": entry_threshold,
                "exit_threshold": exit_threshold,
                "zscore_current": round(zscore, 4)
            }
            
        except Exception:
            return {
                "trading_signal": "hold",
                "position_recommendation": "insufficient data",
                "signal_strength": 0.0,
                "entry_threshold": 2.0,
                "exit_threshold": 0.5,
                "zscore_current": 0.0
            }
    
    def _assess_hedge_ratio_quality(self, r_squared: float, 
                                  confidence: float) -> str:
        """Assess hedge ratio quality."""
        if r_squared > 0.8 and confidence > 80:
            return "excellent"
        elif r_squared > 0.6 and confidence > 60:
            return "good"
        elif r_squared > 0.4 and confidence > 40:
            return "fair"
        else:
            return "poor"
    
    def _create_error_result(self, error_msg: str, 
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "hedge_ratio": 1.0,
            "alpha": 0.0,
            "r_squared": 0.0,
            "confidence": 0.0,
            "hedge_ratio_quality": "unknown",
            "trading_signal": "hold",
            "signal_strength": 0.0,
            "error": error_msg,
            **metadata
        }


class RollingOLSHedgeRatio(OLSHedgeRatio):
    """
    Rolling window OLS hedge ratio for dynamic hedging.
    
    Calculates time-varying hedge ratios to adapt to changing
    market conditions and asset relationships.
    """
    
    name = "rolling_ols_hedge_ratio"
    description = "Rolling window OLS hedge ratio calculation"
    
    def __init__(self, window_size: int = 63, step_size: int = 5, **kwargs):
        """
        Initialize rolling hedge ratio calculation.
        
        Parameters:
        -----------
        window_size : int, default 63
            Rolling window size (e.g., 3 months)
        step_size : int, default 5
            Step size for rolling calculation (e.g., weekly)
        """
        self.window_size = window_size
        self.step_size = step_size
        super().__init__(**kwargs)
    
    def _calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate rolling hedge ratio statistics."""
        try:
            prices = data[self.asset_columns].dropna()
            
            if len(prices) < self.window_size + self.step_size:
                return self._create_error_result(
                    "Insufficient data for rolling hedge ratio",
                    {"required": self.window_size + self.step_size, 
                     "actual": len(prices)}
                )
            
            # Calculate rolling hedge ratios
            results = []
            hedge_ratios = []
            
            for i in range(self.window_size, len(prices), self.step_size):
                window_data = prices.iloc[i-self.window_size:i]
                window_result = super()._calculate(pd.DataFrame(window_data))
                
                if "error" not in window_result:
                    results.append(window_result)
                    hedge_ratios.append(window_result["hedge_ratio"])
            
            if not results:
                return self._create_error_result(
                    "No valid rolling windows calculated",
                    {"window_size": self.window_size, 
                     "step_size": self.step_size}
                )
            
            # Aggregate results
            recent_result = results[-1]
            
            # Calculate hedge ratio stability
            avg_hedge_ratio = np.mean(hedge_ratios)
            hedge_ratio_std = np.std(hedge_ratios)
            hedge_ratio_stability = (100 - (hedge_ratio_std / avg_hedge_ratio * 100) 
                                   if avg_hedge_ratio != 0 else 0)
            
            # Calculate other stability metrics
            r_squareds = [r["r_squared"] for r in results]
            confidences = [r["confidence"] for r in results]
            
            return {
                **recent_result,  # Most recent calculation
                "rolling_windows": len(results),
                "avg_hedge_ratio": round(avg_hedge_ratio, 6),
                "hedge_ratio_std": round(hedge_ratio_std, 6),
                "hedge_ratio_stability": round(hedge_ratio_stability, 2),
                "avg_r_squared": round(np.mean(r_squareds), 4),
                "avg_confidence": round(np.mean(confidences), 2),
                "window_size": self.window_size,
                "step_size": self.step_size,
                "relationship_stability": ("stable" if hedge_ratio_stability > 80
                                         else "moderately_stable" 
                                         if hedge_ratio_stability > 60
                                         else "unstable")
            }
            
        except Exception as e:
            return self._create_error_result(
                f"Rolling hedge ratio calculation failed: {str(e)}",
                {"window_size": self.window_size, "step_size": self.step_size}
            )
