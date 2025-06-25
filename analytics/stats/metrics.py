"""
Advanced Statistical Metrics for Hourly Statistics.

This module computes and provides advanced statistical metrics including
half-life of mean reversion, Sharpe ratio, and Z-score breach counts
for spread series in the statistical arbitrage engine.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any


def compute_hourly_statistics(
    spread: pd.Series, zscore: pd.Series
) -> Dict[str, Any]:
    """
    Compute advanced statistical metrics for hourly statistics.
    
    This function calculates half-life of mean reversion, Sharpe ratio,
    and Z-score breach counts for a given spread series and its corresponding
    Z-score values.
    
    Parameters:
    -----------
    spread : pd.Series
        The price spread series between two cointegrated assets
    zscore : pd.Series
        The Z-score series calculated from the spread
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the following metrics:
        - half_life: Half-life of mean reversion (float)
        - sharpe_ratio: Sharpe ratio of spread returns (float)
        - zscore_over_2: Count of Z-scores above +2 (int)
        - zscore_under_minus_2: Count of Z-scores below -2 (int)
        
    Raises:
    -------
    ValueError
        If input Series is empty or contains NaN values
        If insufficient data for calculations
    """
    logger = logging.getLogger(__name__)
    
    # Input validation
    if spread.empty or zscore.empty:
        raise ValueError("Input Series cannot be empty")
    
    if spread.isna().any():
        raise ValueError("Spread series contains NaN values")
    
    if zscore.isna().any():
        raise ValueError("Z-score series contains NaN values")
    
    if len(spread) < 2:
        raise ValueError(
            "Insufficient data: spread series must have at least "
            "2 observations"
        )
    
    try:
        # Calculate half-life of mean reversion using AR(1) regression
        half_life = _calculate_half_life(spread)
        
        # Calculate Sharpe ratio of spread returns
        sharpe_ratio = _calculate_sharpe_ratio(spread)
        
        # Count Z-score breaches
        zscore_over_2 = int((zscore > 2).sum())
        zscore_under_minus_2 = int((zscore < -2).sum())
        
        result = {
            'half_life': half_life,
            'sharpe_ratio': sharpe_ratio,
            'zscore_over_2': zscore_over_2,
            'zscore_under_minus_2': zscore_under_minus_2
        }
        
        logger.debug(f"Computed hourly statistics: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error computing hourly statistics: {e}")
        raise


def _calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate half-life of mean reversion using AR(1) regression.
    
    Uses the formula: half_life = -log(2) / log(beta)
    where beta is the coefficient from regressing spread[t] on spread[t-1].
    
    Parameters:
    -----------
    spread : pd.Series
        The price spread series
        
    Returns:
    --------
    float
        Half-life in the same units as the spread series index
        
    Raises:
    -------
    ValueError
        If regression cannot be performed or beta coefficient is invalid
    """
    try:
        # Create lagged spread series
        spread_lag = spread.shift(1).dropna()
        # Remove first observation to align with lag
        spread_current = spread[1:]
        
        if len(spread_current) < 2:
            raise ValueError("Insufficient data for half-life calculation")
        
        # Perform AR(1) regression:
        # spread[t] = alpha + beta * spread[t-1] + error
        # Using numpy's least squares solution
        X = np.column_stack([np.ones(len(spread_lag)), spread_lag.values])
        y = spread_current.values
        
        # Solve for coefficients [alpha, beta]
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        beta = coefficients[1]
        
        # Validate beta coefficient
        if beta >= 1.0:
            # Non-stationary or very slow mean reversion
            return np.inf
        
        if beta <= 0.0:
            # Invalid beta for mean reversion calculation
            raise ValueError(
                "Invalid beta coefficient for half-life calculation"
            )
        
        # Calculate half-life: -log(2) / log(beta)
        half_life = -np.log(2) / np.log(beta)
        
        # Ensure half-life is positive and finite
        if not np.isfinite(half_life) or half_life <= 0:
            return np.inf
        
        return float(half_life)
        
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Half-life calculation failed: {e}"
        )
        return np.inf


def _calculate_sharpe_ratio(spread: pd.Series) -> float:
    """
    Calculate Sharpe ratio of spread returns.
    
    Uses the formula: sharpe_ratio = mean_return / std_return
    Assumes zero risk-free rate.
    
    Parameters:
    -----------
    spread : pd.Series
        The price spread series
        
    Returns:
    --------
    float
        Sharpe ratio of the spread returns
        
    Raises:
    -------
    ValueError
        If insufficient data for calculation
    """
    try:
        # Calculate returns (percentage change)
        spread_returns = spread.pct_change().dropna()
        
        if len(spread_returns) < 2:
            raise ValueError("Insufficient data for Sharpe ratio calculation")
        
        # Calculate mean and standard deviation of returns
        mean_return = spread_returns.mean()
        std_return = spread_returns.std()
        
        # Handle zero standard deviation case
        if std_return == 0 or np.isnan(std_return):
            return 0.0
        
        # Calculate Sharpe ratio
        sharpe_ratio = mean_return / std_return
        
        # Ensure finite result
        if not np.isfinite(sharpe_ratio):
            return 0.0
        
        return float(sharpe_ratio)
        
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Sharpe ratio calculation failed: {e}"
        )
        return 0.0


def validate_metrics_result(result: Dict[str, Any]) -> bool:
    """
    Validate the structure and content of metrics result.
    
    Parameters:
    -----------
    result : Dict[str, Any]
        Result dictionary from compute_hourly_statistics
        
    Returns:
    --------
    bool
        True if result is valid, False otherwise
    """
    required_fields = [
        'half_life', 'sharpe_ratio', 'zscore_over_2', 'zscore_under_minus_2'
    ]
    
    # Check all required fields are present
    if not all(field in result for field in required_fields):
        return False
    
    # Check data types
    try:
        # half_life and sharpe_ratio should be numeric
        float(result['half_life'])
        float(result['sharpe_ratio'])
        
        # Z-score counts should be non-negative integers
        if result['zscore_over_2'] < 0 or result['zscore_under_minus_2'] < 0:
            return False
        
        int(result['zscore_over_2'])
        int(result['zscore_under_minus_2'])
        
        return True
        
    except (ValueError, TypeError):
        return False
