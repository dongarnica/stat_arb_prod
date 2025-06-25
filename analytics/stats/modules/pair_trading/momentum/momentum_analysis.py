"""
Momentum Analysis for Pair Trading.

This module calculates momentum-based statistics for trading pairs,
complementing mean reversion strategies with trend and momentum insights.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import date

from ..base_pair_statistic import BasePairStatistic


class MomentumAnalysis(BasePairStatistic):
    """
    Calculate momentum statistics for trading pairs.
    
    This class provides various momentum metrics including:
    - Multi-timeframe momentum
    - RSI-like momentum indicators
    - Momentum divergence
    - Momentum persistence
    """
    
    # Class attributes required by the orchestrator
    name = "momentum_analysis"
    description = "Momentum analysis for pair trading"
    category = "pair_trading"
    required_columns = ["close"]
    
    def __init__(self):
        """Initialize the momentum analysis."""
        super().__init__()
    
    def calculate_momentum_windows(self, prices: pd.Series, windows: list = [5, 10, 20, 50]) -> Dict[str, float]:
        """
        Calculate momentum over multiple timeframes.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        windows : list
            List of lookback windows
            
        Returns:
        --------
        Dict[str, float]
            Momentum values for each window
        """
        momentum = {}
        
        for window in windows:
            if len(prices) > window:
                # Calculate percentage change over window
                current_price = prices.iloc[-1]
                past_price = prices.iloc[-window-1]
                momentum[f'momentum_{window}d'] = (current_price - past_price) / past_price
            else:
                momentum[f'momentum_{window}d'] = 0.0
                
        return momentum
    
    def calculate_rsi_momentum(self, prices: pd.Series, window: int = 14) -> float:
        """
        Calculate RSI-like momentum indicator.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        window : int
            RSI calculation window
            
        Returns:
        --------
        float
            RSI value (0-100)
        """
        if len(prices) < window + 1:
            return 50.0  # Neutral
            
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Calculate RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def calculate_momentum_divergence(self, prices1: pd.Series, prices2: pd.Series, window: int = 20) -> Dict[str, float]:
        """
        Calculate momentum divergence between two assets.
        
        Parameters:
        -----------
        prices1 : pd.Series
            First asset prices
        prices2 : pd.Series
            Second asset prices
        window : int
            Lookback window
            
        Returns:
        --------
        Dict[str, float]
            Divergence metrics
        """
        if len(prices1) < window or len(prices2) < window:
            return {
                'momentum_divergence': 0.0,
                'momentum_correlation': 0.0,
                'divergence_strength': 0.0
            }
        
        # Calculate momentum for both assets
        mom1 = prices1.pct_change(window)
        mom2 = prices2.pct_change(window)
        
        # Recent momentum values
        recent_mom1 = mom1.iloc[-5:].mean()
        recent_mom2 = mom2.iloc[-5:].mean()
        
        # Calculate divergence
        divergence = abs(recent_mom1 - recent_mom2)
        
        # Calculate momentum correlation
        momentum_corr = mom1.tail(window).corr(mom2.tail(window))
        if pd.isna(momentum_corr):
            momentum_corr = 0.0
        
        # Divergence strength (higher when momentum moves in opposite directions)
        divergence_strength = abs(recent_mom1) + abs(recent_mom2) if recent_mom1 * recent_mom2 < 0 else 0.0
        
        return {
            'momentum_divergence': float(divergence),
            'momentum_correlation': float(momentum_corr),
            'divergence_strength': float(divergence_strength)
        }
    
    def calculate_momentum_persistence(self, prices: pd.Series, window: int = 10) -> Dict[str, float]:
        """
        Calculate momentum persistence metrics.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        window : int
            Analysis window
            
        Returns:
        --------
        Dict[str, float]
            Persistence metrics
        """
        if len(prices) < window * 2:
            return {
                'momentum_persistence': 0.0,
                'trend_consistency': 0.0,
                'momentum_acceleration': 0.0
            }
        
        # Calculate short-term momentum
        returns = prices.pct_change()
        
        # Momentum persistence (how often momentum continues in same direction)
        momentum_signs = np.sign(returns.rolling(window).mean())
        sign_changes = (momentum_signs.diff() != 0).sum()
        persistence = 1.0 - (sign_changes / len(momentum_signs))
        
        # Trend consistency (volatility of momentum)
        momentum_values = returns.rolling(window).mean()
        trend_consistency = 1.0 / (1.0 + momentum_values.std()) if momentum_values.std() > 0 else 1.0
        
        # Momentum acceleration (is momentum increasing?)
        recent_momentum = momentum_values.tail(window).mean()
        past_momentum = momentum_values.iloc[-window*2:-window].mean()
        acceleration = recent_momentum - past_momentum
        
        return {
            'momentum_persistence': float(persistence) if not pd.isna(persistence) else 0.0,
            'trend_consistency': float(trend_consistency) if not pd.isna(trend_consistency) else 0.0,
            'momentum_acceleration': float(acceleration) if not pd.isna(acceleration) else 0.0
        }
    
    def calculate(self, data1: pd.DataFrame, data2: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate momentum analysis for pair of assets.
        
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
            Momentum analysis results
        """
        try:
            # Extract close prices
            prices1 = data1['close'] if 'close' in data1.columns else data1.iloc[:, 0]
            prices2 = data2['close'] if 'close' in data2.columns else data2.iloc[:, 0]
            
            # Calculate momentum for both assets
            momentum1 = self.calculate_momentum_windows(prices1)
            momentum2 = self.calculate_momentum_windows(prices2)
            
            # Calculate relative momentum
            relative_momentum = {}
            for window in [5, 10, 20, 50]:
                key = f"momentum_{window}d"
                if key in momentum1 and key in momentum2:
                    relative_momentum[f"relative_{key}"] = momentum1[key] - momentum2[key]
            
            return {
                "asset_1_momentum": momentum1,
                "asset_2_momentum": momentum2, 
                "relative_momentum": relative_momentum,
                "statistic_name": self.name
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "statistic_name": self.name,
                "symbol_1": kwargs.get('symbol1', 'unknown'),
                "symbol_2": kwargs.get('symbol2', 'unknown')
            }

    def _create_success_result(self, results: Dict[str, float], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a successful result dictionary."""
        return {
            'success': True,
            'results': results,
            'metadata': metadata,
            'error': None
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result dictionary."""
        return {
            'success': False,
            'results': {},
            'metadata': {},
            'error': error_message
        }
