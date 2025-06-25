"""
Base Pair Statistic Class.

Provides the foundation for all pair trading statistics calculations.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import date

from ...base_statistic import BaseStatistic


class BasePairStatistic(BaseStatistic):
    """
    Base class for pair trading statistics.
    
    This class provides common functionality for all pair trading
    statistics including data validation, caching, and result formatting.
    """
    
    def __init__(self):
        """Initialize the base pair statistic."""
        super().__init__()
        self.requires_pair = True
    
    def validate_pair_data(self, data1: pd.DataFrame, data2: pd.DataFrame) -> bool:
        """
        Validate that both datasets are suitable for pair analysis.
        
        Parameters:
        -----------
        data1 : pd.DataFrame
            First asset's price data
        data2 : pd.DataFrame
            Second asset's price data
            
        Returns:
        --------
        bool
            True if data is valid for pair analysis
        """
        if data1.empty or data2.empty:
            return False
            
        # Check for minimum data points
        if len(data1) < 30 or len(data2) < 30:
            return False
            
        # Check for required columns
        required_cols = ['close']
        if not all(col in data1.columns for col in required_cols):
            return False
        if not all(col in data2.columns for col in required_cols):
            return False
            
        return True
    
    def align_data(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align two datasets by their common time index.
        
        Parameters:
        -----------
        data1 : pd.DataFrame
            First asset's data
        data2 : pd.DataFrame
            Second asset's data
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Aligned datasets
        """
        # Find common dates
        common_index = data1.index.intersection(data2.index)
        
        if len(common_index) == 0:
            raise ValueError("No common dates found between datasets")
            
        # Align data
        aligned_data1 = data1.loc[common_index].sort_index()
        aligned_data2 = data2.loc[common_index].sort_index()
        
        return aligned_data1, aligned_data2
    
    def calculate(self, data1: pd.DataFrame, data2: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate the pair trading statistic.
        
        This method should be overridden by subclasses.
        
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
            Calculation results
        """
        raise NotImplementedError("Subclasses must implement calculate method")
