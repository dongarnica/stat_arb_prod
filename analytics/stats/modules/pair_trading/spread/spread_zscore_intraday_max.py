"""
Intraday Maximum Spread Z-Score Analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, time
from ..base_pair_statistic import BasePairStatistic


class SpreadZScoreIntradayMax(BasePairStatistic):
    """Track intraday maximum spread Z-score extremes."""
    
    name = "spread_zscore_intraday_max"
    description = "Intraday maximum spread Z-score for extreme detection"
    category = "pair_trading"
    required_columns = ["close"]
    asset_columns = ["asset_1_close", "asset_2_close"]
    min_data_points = 100
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate intraday maximum spread Z-score metrics."""
        self._validate_pair_input(data)
        
        valid_data = data[self.asset_columns].dropna()
        
        if len(valid_data) < self.min_data_points:
            raise ValueError(f"Insufficient paired data points: {len(valid_data)}")
        
        asset1_prices = valid_data[self.asset_columns[0]]
        asset2_prices = valid_data[self.asset_columns[1]]
        
        log_spread = np.log(asset1_prices) - np.log(asset2_prices)
        
        lookback = 30
        spread_mean = log_spread.rolling(lookback).mean()
        spread_std = log_spread.rolling(lookback).std()
        spread_zscore = (log_spread - spread_mean) / spread_std
        
        valid_zscores = spread_zscore.dropna()
        if len(valid_zscores) < 10:
            raise ValueError("Insufficient data for Z-score calculation")
        
        current_zscore = float(spread_zscore.iloc[-1])
        intraday_max_abs_zscore = float(valid_zscores.tail(50).abs().max())
        
        return {
            self.name: intraday_max_abs_zscore,
            "current_zscore": current_zscore,
            "data_points_used": int(len(valid_data))
        }
        
    def _validate_pair_input(self, data: pd.DataFrame) -> None:
        """Validate that data contains required pair columns."""
        missing_cols = [col for col in self.asset_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
