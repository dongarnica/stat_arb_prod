"""
Spread Z-score calculation for pair trading signals.

This module calculates Z-scores for different spread types to generate
entry and exit signals for pair trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict
from ..base_pair_statistic import BasePairStatistic


class SpreadZScore(BasePairStatistic):
    """
    Calculate Z-scores for pair trading spreads.
    
    Computes standardized Z-scores for price ratio, log ratio, and
    difference spreads to identify trading opportunities.
    """
    
    name = "spread_zscore"
    description = "Z-score calculation for pair trading spreads"
    category = "pair_trading"
    required_columns = ["close"]
    asset_columns = ["asset_1_close", "asset_2_close"]
    min_data_points = 30
    
    def _validate_pair_input(self, data: pd.DataFrame) -> None:
        """
        Validate pair input data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Combined pair data
            
        Raises:
        -------
        ValueError
            If data is invalid
        """
        if data.empty:
            raise ValueError("Input data is empty")          missing_cols = [col for col in self.asset_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Calculate spread Z-scores for pair of assets.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Combined pair data with asset_1_close, asset_2_close columns
        **kwargs
            Additional parameters
            
        Returns:
        --------
        Dict[str, float]
            Z-score metrics for different spread types
        """
        try:
            # Call the existing calculate logic
            return self._calculate_zscore(data)
            
        except Exception as e:
            return {
                "error": str(e),
                "statistic_name": self.name
            }

    def _calculate_zscore(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Z-scores for different spread types.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing both assets' close prices with columns
            'asset_1_close' and 'asset_2_close'
            
        Returns:
        --------
        dict
            Z-score metrics for different spread types
        """
        # Validate pair input
        self._validate_pair_input(data)
        
        # Get aligned data for both assets
        valid_data = data[self.asset_columns].dropna()
        
        if len(valid_data) < self.min_data_points:
            raise ValueError(
                f"Insufficient paired data points: {len(valid_data)}"
            )
        
        asset1_prices = valid_data[self.asset_columns[0]]
        asset2_prices = valid_data[self.asset_columns[1]]
        
        # Calculate different spread types
        price_ratio_spread = asset1_prices / asset2_prices
        log_ratio_spread = np.log(asset1_prices) - np.log(asset2_prices)
        difference_spread = asset1_prices - asset2_prices
        
        # Calculate current (most recent) values
        current_price_ratio = float(price_ratio_spread.iloc[-1])
        current_log_ratio = float(log_ratio_spread.iloc[-1])
        current_difference = float(difference_spread.iloc[-1])
        
        # Calculate rolling statistics for Z-score
        lookback = min(30, len(valid_data) // 2)
        
        # Price ratio Z-score
        price_ratio_mean = float(price_ratio_spread.rolling(lookback).mean().iloc[-1])
        price_ratio_std = float(price_ratio_spread.rolling(lookback).std().iloc[-1])
        price_ratio_zscore = float(
            (current_price_ratio - price_ratio_mean) / price_ratio_std
            if price_ratio_std > 0 else 0
        )
        
        # Log ratio Z-score
        log_ratio_mean = float(log_ratio_spread.rolling(lookback).mean().iloc[-1])
        log_ratio_std = float(log_ratio_spread.rolling(lookback).std().iloc[-1])
        log_ratio_zscore = float(
            (current_log_ratio - log_ratio_mean) / log_ratio_std
            if log_ratio_std > 0 else 0
        )
        
        # Difference Z-score
        diff_mean = float(difference_spread.rolling(lookback).mean().iloc[-1])
        diff_std = float(difference_spread.rolling(lookback).std().iloc[-1])
        difference_zscore = float(
            (current_difference - diff_mean) / diff_std
            if diff_std > 0 else 0
        )
        
        # Calculate signal strength (0-1 scale)
        max_zscore = max(abs(price_ratio_zscore), abs(log_ratio_zscore), abs(difference_zscore))
        signal_strength = float(min(1.0, max_zscore / 2.0))  # Normalize to 0-1
        
        # Determine trading signal direction
        if price_ratio_zscore > 1.5:
            signal_direction = "short_pair"  # Asset 1 overvalued vs Asset 2
        elif price_ratio_zscore < -1.5:
            signal_direction = "long_pair"   # Asset 1 undervalued vs Asset 2
        else:
            signal_direction = "neutral"
        
        return {
            self.name: price_ratio_zscore,  # Primary Z-score value
            "price_ratio_zscore": price_ratio_zscore,
            "log_ratio_zscore": log_ratio_zscore,
            "difference_zscore": difference_zscore,
            "current_price_ratio": current_price_ratio,
            "current_log_ratio": current_log_ratio,
            "current_difference": current_difference,
            "price_ratio_mean": price_ratio_mean,
            "price_ratio_std": price_ratio_std,
            "signal_strength": signal_strength,
            "signal_direction": signal_direction,
            "lookback_periods": lookback,
            "data_points_used": int(len(valid_data))
        }
