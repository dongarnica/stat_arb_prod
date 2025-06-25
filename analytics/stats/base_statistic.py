"""
Base statistic class for the modular statistics framework.
Provides standardized interface for all statistical calculations.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Any
import warnings


class BaseStatistic(ABC):
    """
    Abstract base class for all statistical calculations.
    
    All statistic modules must inherit from this class and implement
    the calculate() method to ensure consistent interface and behavior.
    """
    
    # Class attributes to be defined by subclasses
    name: str = None
    description: str = None
    category: str = None
    required_columns: List[str] = None
    min_data_points: int = 1
    
    def __init__(self, **kwargs):
        """Initialize base statistic with common setup."""
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.config = kwargs
        
        # Validate class attributes are properly defined
        self._validate_class_attributes()
    
    def _validate_class_attributes(self) -> None:
        """Validate that required class attributes are defined."""
        if self.name is None:
            raise ValueError(
                f"{self.__class__.__name__} must define 'name' attribute"
            )
        if self.description is None:
            raise ValueError(
                f"{self.__class__.__name__} must define 'description'"
            )
        if self.category is None:
            raise ValueError(
                f"{self.__class__.__name__} must define 'category' attribute"
            )
        if self.required_columns is None:
            raise ValueError(
                f"{self.__class__.__name__} must define 'required_columns'"
            )
        
        # Validate category
        valid_categories = [
            'basic', 'technical', 'statistical', 'mean_reversion',
            'risk', 'correlation', 'pair_trading'
        ]
        if self.category not in valid_categories:
            raise ValueError(
                f"Category '{self.category}' must be one of {valid_categories}"
            )
    
    def _validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input data meets requirements.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input OHLCV data to validate
            
        Raises:
        -------
        ValueError
            If data doesn't meet requirements
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        if len(data) < self.min_data_points:
            raise ValueError(
                f"Insufficient data points. Required: {self.min_data_points}, "
                f"Available: {len(data)}"
            )
        
        # Check required columns exist
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate required columns are numeric
        for col in self.required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column '{col}' must be numeric")
        
        # Check for all NaN values in required columns
        for col in self.required_columns:
            if data[col].isna().all():
                raise ValueError(f"Column '{col}' contains only NaN values")
        
        # Warn if significant NaN values present
        for col in self.required_columns:
            nan_pct = data[col].isna().sum() / len(data)
            if nan_pct > 0.1:  # More than 10% NaN
                warnings.warn(
                    f"Column '{col}' has {nan_pct:.1%} NaN values",
                    UserWarning
                )
    
    def _validate_result(self, result: Dict[str, Any]) -> None:
        """
        Validate calculation result.
        
        Parameters:
        -----------
        result : dict
            Calculation result to validate
            
        Raises:
        -------
        ValueError
            If result is invalid
        """
        if not isinstance(result, dict):
            raise ValueError("Result must be a dictionary")
        
        if self.name not in result:
            raise ValueError(f"Result must contain key '{self.name}'")
        
        # Convert numpy types to native Python types for JSON compatibility
        for key, value in result.items():
            if isinstance(value, (np.integer, np.floating)):
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    result[key] = float(value.item())
                else:
                    result[key] = value.tolist()
    
    def _handle_missing_data(
        self, data: pd.DataFrame, method: str = 'drop'
    ) -> pd.DataFrame:
        """
        Handle missing data in required columns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        method : str
            Method to handle missing data ('drop', 'forward_fill', 
            'backward_fill')
            
        Returns:
        --------
        pd.DataFrame
            Data with missing values handled
        """
        # Work with a copy to avoid modifying original data
        data_clean = data.copy()
        
        if method == 'drop':
            # Drop rows with NaN in any required column
            data_clean = data_clean.dropna(subset=self.required_columns)
        elif method == 'forward_fill':
            data_clean[self.required_columns] = (
                data_clean[self.required_columns].fillna(method='ffill')
            )
        elif method == 'backward_fill':
            data_clean[self.required_columns] = (
                data_clean[self.required_columns].fillna(method='bfill')
            )
        else:
            raise ValueError(f"Unknown missing data method: {method}")
        
        return data_clean
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this statistic.
        
        Returns:
        --------
        dict
            Metadata dictionary
        """
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'required_columns': self.required_columns,
            'min_data_points': self.min_data_points,
            'class_name': self.__class__.__name__
        }
    
    @abstractmethod
    def calculate(
        self, data: pd.DataFrame
    ) -> Dict[str, Union[float, int, List[float]]]:
        """
        Calculate the statistic.
        
        This method must be implemented by all subclasses.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data with datetime index
            
        Returns:
        --------
        dict
            Dictionary with calculated statistic values
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the statistic."""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"category='{self.category}')"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"category='{self.category}', "
            f"required_columns={self.required_columns}, "
            f"min_data_points={self.min_data_points})"
        )
