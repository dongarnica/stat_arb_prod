"""
Spread Acceleration Analysis for Entry Timing.
"""

import pandas as pd
import numpy as np
from typing import Dict
from ..base_pair_statistic import BasePairStatistic


class SpreadAcceleration(BasePairStatistic):
    """Calculate spread acceleration for precise entry timing."""
    
    name = "spread_acceleration"
    description = "Second derivative analysis for spread entry timing"
    category = "pair_trading"
    required_columns = ["close"]
    asset_columns = ["asset_1_close", "asset_2_close"]
    min_data_points = 50
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate spread acceleration metrics."""
        self._validate_pair_input(data)
        
        valid_data = data[self.asset_columns].dropna()
        
        if len(valid_data) < self.min_data_points:
            raise ValueError(f"Insufficient paired data points: {len(valid_data)}")
        
        asset1_prices = valid_data[self.asset_columns[0]]
        asset2_prices = valid_data[self.asset_columns[1]]
        
        log_spread = np.log(asset1_prices) - np.log(asset2_prices)
        spread_velocity = log_spread.diff()
        spread_acceleration = spread_velocity.diff()
        
        valid_acceleration = spread_acceleration.dropna()
        if len(valid_acceleration) < 10:
            raise ValueError("Insufficient data for acceleration calculation")
        
        current_acceleration = float(spread_acceleration.iloc[-1])
        
        return {
            self.name: current_acceleration,
            "data_points_used": int(len(valid_data))
        }
        
    def _validate_pair_input(self, data: pd.DataFrame) -> None:
        """Validate that data contains required pair columns."""
        missing_cols = [col for col in self.asset_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
