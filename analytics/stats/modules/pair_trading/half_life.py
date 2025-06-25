"""
Half-life calculation for pair trading mean reversion.

This module calculates the half-life of mean reversion for spread series,
indicating how quickly spreads return to their mean value.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from .base_pair_statistic import BasePairStatistic


class HalfLife(BasePairStatistic):
    """
    Calculate half-life of mean reversion for pair trading spreads.
    
    Measures the expected time for a spread to revert halfway back to its
    mean, providing insight into mean reversion speed.
    """
    
    name = "half_life"
    description = "Half-life of mean reversion for pair trading spreads"
    category = "pair_trading"
    required_columns = ["close"]
    asset_columns = ["asset_1_close", "asset_2_close"]
    min_data_points = 30
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate half-life of mean reversion.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing both assets' close prices
            
        Returns:
        --------
        dict
            Half-life statistics and mean reversion metrics
        """
        # Validate pair input
        self._validate_pair_input(data)
        
        # Calculate spread (using parent class method)
        spread = self._calculate_spread(data)
        spread_clean = spread.dropna()
        
        if len(spread_clean) < self.min_data_points:
            raise ValueError(
                f"Insufficient spread data points: {len(spread_clean)}"
            )
        
        # Calculate half-life using parent class method
        half_life_periods = self._calculate_half_life(spread_clean)
        
        # Calculate additional mean reversion metrics
        spread_mean = float(spread_clean.mean())
        spread_std = float(spread_clean.std())
        current_spread = float(spread_clean.iloc[-1])
        
        # Calculate mean reversion strength (R-squared of AR(1) model)
        spread_lag = spread_clean.shift(1).dropna()
        spread_current = spread_clean[1:].values
        
        if len(spread_lag) > 5:
            reg = LinearRegression()
            reg.fit(spread_lag.values.reshape(-1, 1), spread_current)
            mean_reversion_strength = float(reg.score(
                spread_lag.values.reshape(-1, 1), spread_current
            ))
        else:
            mean_reversion_strength = 0.0
        
        # Calculate current deviation from mean in standard deviations
        current_z_score = float(
            (current_spread - spread_mean) / spread_std
            if spread_std > 0 else 0
        )
        
        # Estimate mean reversion signal strength
        if half_life_periods < np.inf and half_life_periods > 0:
            reversion_speed = 1.0 / half_life_periods
            signal_strength = min(1.0, abs(current_z_score) * reversion_speed)
        else:
            reversion_speed = 0.0
            signal_strength = 0.0
        
        return {
            self.name: float(half_life_periods),
            "half_life_periods": float(half_life_periods),
            "mean_reversion_strength": mean_reversion_strength,
            "reversion_speed": float(reversion_speed),
            "current_spread": current_spread,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "current_z_score": current_z_score,
            "signal_strength": float(signal_strength),
            "is_mean_reverting": bool(half_life_periods < np.inf),
            "data_points_used": int(len(spread_clean))
        }
