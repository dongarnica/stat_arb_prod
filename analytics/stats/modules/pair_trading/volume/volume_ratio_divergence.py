"""
Volume Ratio Divergence Analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict
from ..base_pair_statistic import BasePairStatistic


class VolumeRatioDivergence(BasePairStatistic):
    """Analyze volume ratio divergence for breakout confirmation."""
    
    name = "volume_ratio_divergence"
    description = "Volume pattern analysis for breakout vs noise detection"
    category = "pair_trading"
    required_columns = ["close", "volume"]
    asset_columns = ["asset_1_close", "asset_2_close", "asset_1_volume", "asset_2_volume"]
    min_data_points = 50
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume ratio divergence metrics."""
        self._validate_pair_input(data)
        
        valid_data = data[self.asset_columns].dropna()
        
        if len(valid_data) < self.min_data_points:
            raise ValueError(f"Insufficient paired data points: {len(valid_data)}")
        
        asset1_prices = valid_data['asset_1_close']
        asset2_prices = valid_data['asset_2_close']
        asset1_volume = valid_data['asset_1_volume']
        asset2_volume = valid_data['asset_2_volume']
        
        log_spread = np.log(asset1_prices) - np.log(asset2_prices)
        spread_change = log_spread.diff()
        
        volume_ratio = asset1_volume / (asset2_volume + 1e-8)
        current_spread_change = float(spread_change.iloc[-1])
        current_volume_ratio = float(volume_ratio.iloc[-1])
        
        # Simple divergence calculation
        divergence_ratio = float(abs(current_spread_change) * current_volume_ratio)
        
        return {
            self.name: divergence_ratio,
            "current_spread_change": current_spread_change,
            "current_volume_ratio": current_volume_ratio,
            "data_points_used": int(len(valid_data))
        }
        
    def _validate_pair_input(self, data: pd.DataFrame) -> None:
        """Validate that data contains required pair columns."""
        missing_cols = [col for col in self.asset_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
