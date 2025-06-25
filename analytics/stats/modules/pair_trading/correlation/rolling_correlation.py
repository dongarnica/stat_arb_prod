"""
Rolling Correlation Analysis for Pair Trading.

This module calculates rolling correlations between asset pairs,
which is essential for monitoring pair relationship stability
and detecting regime changes in pair trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
from ..base_pair_statistic import BasePairStatistic


class RollingCorrelation(BasePairStatistic):
    """
    Calculate rolling correlation between asset pairs.
    
    Rolling correlation analysis helps with:
    - Monitoring pair relationship stability
    - Detecting correlation breakdowns
    - Timing entry and exit points
    - Risk management decisions
    """
    
    # Required class attributes
    name = "rolling_correlation"
    description = "Rolling correlation analysis for pair trading"
    category = "correlation"
    asset_columns = ["asset_1_close", "asset_2_close"]
    required_columns = ["asset_1_close", "asset_2_close"]
    lookback_period = 252
    
    def __init__(self, window_size: int = 21, min_periods: int = 10, **kwargs):
        """
        Initialize rolling correlation calculation.
        
        Parameters:
        -----------
        window_size : int, default 21
            Rolling window size for correlation calculation
        min_periods : int, default 10
            Minimum periods required for valid correlation
        """
        if window_size < min_periods:
            raise ValueError("window_size must be >= min_periods")
        
        self.window_size = window_size
        self.min_periods = min_periods
        super().__init__(**kwargs)
    
    def calculate(self, data1: pd.DataFrame, data2: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate rolling correlation for pair of assets.
        
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
        try:
            # Validate inputs
            if not self.validate_pair_data(data1, data2):
                return {
                    "error": "Invalid pair data",
                    "statistic_name": self.name,
                    "symbol_1": kwargs.get('symbol1', 'unknown'),
                    "symbol_2": kwargs.get('symbol2', 'unknown')
                }
            
            # Align data
            aligned_data1, aligned_data2 = self.align_data(data1, data2)
            
            # Create combined dataframe for _calculate method
            combined_data = pd.DataFrame({
                'asset_1_close': aligned_data1['close'],
                'asset_2_close': aligned_data2['close']
            })
            
            # Call the existing _calculate method
            return self._calculate(combined_data)
            
        except Exception as e:
            return {
                "error": str(e),
                "statistic_name": self.name,
                "symbol_1": kwargs.get('symbol1', 'unknown'),
                "symbol_2": kwargs.get('symbol2', 'unknown')
            }
    
    def _calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate rolling correlation statistics.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with required asset columns
            
        Returns:
        --------
        Dict[str, Any]
            Rolling correlation analysis results
        """
        try:
            # Extract price series
            prices = data[self.asset_columns].dropna()
            
            if len(prices) < self.window_size:
                return self._create_error_result(
                    "Insufficient data for rolling correlation",
                    {"min_required": self.window_size, "actual": len(prices)}
                )
            
            # Calculate returns for correlation analysis
            returns = prices.pct_change().dropna()
            
            if len(returns) < self.min_periods:
                return self._create_error_result(
                    "Insufficient return data for correlation",
                    {"min_required": self.min_periods,
                     "actual": len(returns)}
                )
            
            # Calculate rolling correlation
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                rolling_corr = returns.iloc[:, 0].rolling(
                    window=self.window_size,
                    min_periods=self.min_periods
                ).corr(returns.iloc[:, 1])
            
            # Remove NaN values
            rolling_corr = rolling_corr.dropna()
            
            if len(rolling_corr) == 0:
                return self._create_error_result(
                    "No valid correlation values calculated",
                    {"window_size": self.window_size}
                )
            
            # Calculate correlation statistics
            corr_stats = self._calculate_correlation_stats(rolling_corr)
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(rolling_corr)
            
            # Generate trading assessment
            trading_assessment = self._assess_trading_suitability(
                rolling_corr, corr_stats)
            
            return {
                **corr_stats,
                **stability_metrics,
                **trading_assessment,
                "window_size": self.window_size,
                "min_periods": self.min_periods,
                "sample_size": len(rolling_corr)
            }
            
        except Exception as e:
            return self._create_error_result(
                f"Rolling correlation calculation failed: {str(e)}",
                {"window_size": self.window_size, 
                 "min_periods": self.min_periods}
            )
    
    def _calculate_correlation_stats(self, 
                                   rolling_corr: pd.Series) -> Dict[str, Any]:
        """Calculate basic correlation statistics."""
        try:
            current_corr = rolling_corr.iloc[-1]
            mean_corr = rolling_corr.mean()
            std_corr = rolling_corr.std()
            min_corr = rolling_corr.min()
            max_corr = rolling_corr.max()
            
            # Calculate percentiles
            corr_25 = rolling_corr.quantile(0.25)
            corr_75 = rolling_corr.quantile(0.75)
            
            return {
                "current_correlation": round(current_corr, 4),
                "mean_correlation": round(mean_corr, 4),
                "correlation_std": round(std_corr, 4),
                "min_correlation": round(min_corr, 4),
                "max_correlation": round(max_corr, 4),
                "correlation_range": round(max_corr - min_corr, 4),
                "correlation_25th": round(corr_25, 4),
                "correlation_75th": round(corr_75, 4),
                "correlation_percentile": round(
                    (rolling_corr <= current_corr).mean() * 100, 2)
            }
            
        except Exception:
            return {
                "current_correlation": 0.0,
                "mean_correlation": 0.0,
                "correlation_std": 0.0,
                "min_correlation": 0.0,
                "max_correlation": 0.0,
                "correlation_range": 0.0,
                "correlation_25th": 0.0,
                "correlation_75th": 0.0,
                "correlation_percentile": 50.0
            }
    
    def _calculate_stability_metrics(self, 
                                   rolling_corr: pd.Series) -> Dict[str, Any]:
        """Calculate correlation stability metrics."""
        try:
            # Coefficient of variation
            if rolling_corr.mean() != 0:
                cv = abs(rolling_corr.std() / rolling_corr.mean())
            else:
                cv = float('inf')
            
            # Stability score (0-100, higher is more stable)
            stability_score = max(0, min(100, (1 - cv) * 100))
            
            # Count significant correlation changes
            corr_changes = rolling_corr.diff().abs()
            significant_changes = (corr_changes > 0.1).sum()
            change_frequency = significant_changes / len(rolling_corr) * 100
            
            # Trend analysis
            if len(rolling_corr) > 1:
                trend_slope = np.polyfit(range(len(rolling_corr)), 
                                       rolling_corr.values, 1)[0]
            else:
                trend_slope = 0.0
            
            # Regime classification
            regime = self._classify_correlation_regime(rolling_corr)
            
            return {
                "stability_score": round(stability_score, 2),
                "coefficient_of_variation": round(cv, 4),
                "significant_changes": int(significant_changes),
                "change_frequency_pct": round(change_frequency, 2),
                "trend_slope": round(trend_slope, 6),
                "correlation_regime": regime,
                "is_stable": stability_score > 70
            }
            
        except Exception:
            return {
                "stability_score": 0.0,
                "coefficient_of_variation": float('inf'),
                "significant_changes": 0,
                "change_frequency_pct": 0.0,
                "trend_slope": 0.0,
                "correlation_regime": "unknown",
                "is_stable": False
            }
    
    def _classify_correlation_regime(self, 
                                   rolling_corr: pd.Series) -> str:
        """Classify the correlation regime."""
        try:
            mean_corr = rolling_corr.mean()
            std_corr = rolling_corr.std()
            
            if mean_corr > 0.7 and std_corr < 0.1:
                return "high_stable"
            elif mean_corr > 0.5 and std_corr < 0.15:
                return "moderate_stable"
            elif mean_corr > 0.3 and std_corr < 0.2:
                return "low_stable"
            elif std_corr > 0.3:
                return "highly_volatile"
            elif mean_corr < 0.0:
                return "negative_correlation"
            else:
                return "low_unstable"
                
        except Exception:
            return "unknown"
    
    def _assess_trading_suitability(self, rolling_corr: pd.Series, 
                                  corr_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Assess pair trading suitability based on correlation."""
        try:
            current_corr = corr_stats["current_correlation"]
            mean_corr = corr_stats["mean_correlation"]
            stability_score = corr_stats.get("stability_score", 0)
            
            # Trading suitability assessment
            if mean_corr > 0.6 and stability_score > 70:
                suitability = "excellent"
                confidence = 95
            elif mean_corr > 0.4 and stability_score > 60:
                suitability = "good"
                confidence = 80
            elif mean_corr > 0.2 and stability_score > 50:
                suitability = "fair"
                confidence = 60
            else:
                suitability = "poor"
                confidence = 30
            
            # Risk assessment
            if current_corr < 0.2:
                risk_level = "high"
                risk_warning = "Very low correlation - pair relationship weak"
            elif current_corr < mean_corr - 2 * corr_stats["correlation_std"]:
                risk_level = "elevated"
                risk_warning = "Correlation significantly below historical mean"
            elif stability_score < 50:
                risk_level = "moderate"
                risk_warning = "Correlation is unstable - monitor closely"
            else:
                risk_level = "low"
                risk_warning = "Correlation within normal parameters"
            
            # Trading recommendation
            if suitability in ["excellent", "good"] and risk_level == "low":
                recommendation = "suitable_for_trading"
            elif suitability in ["good", "fair"] and risk_level in ["low", "moderate"]:
                recommendation = "proceed_with_caution"
            else:
                recommendation = "not_recommended"
            
            return {
                "trading_suitability": suitability,
                "suitability_confidence": confidence,
                "risk_level": risk_level,
                "risk_warning": risk_warning,
                "trading_recommendation": recommendation
            }
            
        except Exception:
            return {
                "trading_suitability": "unknown",
                "suitability_confidence": 0,
                "risk_level": "unknown",
                "risk_warning": "Unable to assess risk",
                "trading_recommendation": "insufficient_data"
            }
    
    def _create_error_result(self, error_msg: str, 
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "current_correlation": 0.0,
            "mean_correlation": 0.0,
            "stability_score": 0.0,
            "trading_suitability": "unknown",
            "trading_recommendation": "insufficient_data",
            "error": error_msg,
            **metadata
        }


class CorrelationBreakdownDetector(RollingCorrelation):
    """
    Specialized correlation analyzer for detecting breakdown events.
    
    Focuses on identifying when correlation relationships
    significantly deteriorate, which is critical for risk management.
    """
    
    name = "correlation_breakdown_detector"
    description = "Detect correlation breakdown events in pair trading"
    
    def __init__(self, breakdown_threshold: float = 0.3, 
                 lookback_period: int = 63, **kwargs):
        """
        Initialize correlation breakdown detector.
        
        Parameters:
        -----------
        breakdown_threshold : float, default 0.3
            Correlation level below which breakdown is flagged
        lookback_period : int, default 63
            Period to look back for breakdown detection
        """
        self.breakdown_threshold = breakdown_threshold
        self.lookback_period = lookback_period
        super().__init__(**kwargs)
    
    def _calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation breakdown analysis."""
        try:
            # Get base correlation analysis
            base_result = super()._calculate(data)
            
            if "error" in base_result:
                return base_result
            
            # Extract rolling correlation data
            prices = data[self.asset_columns].dropna()
            returns = prices.pct_change().dropna()
            
            rolling_corr = returns.iloc[:, 0].rolling(
                window=self.window_size,
                min_periods=self.min_periods
            ).corr(returns.iloc[:, 1]).dropna()
            
            # Breakdown detection analysis
            breakdown_analysis = self._detect_breakdowns(rolling_corr)
            
            return {
                **base_result,
                **breakdown_analysis,
                "breakdown_threshold": self.breakdown_threshold,
                "lookback_period": self.lookback_period
            }
            
        except Exception as e:
            return self._create_error_result(
                f"Correlation breakdown detection failed: {str(e)}",
                {"breakdown_threshold": self.breakdown_threshold}
            )
    
    def _detect_breakdowns(self, rolling_corr: pd.Series) -> Dict[str, Any]:
        """Detect correlation breakdown events."""
        try:
            # Current correlation
            current_corr = rolling_corr.iloc[-1]
            
            # Check for current breakdown
            is_breakdown = current_corr < self.breakdown_threshold
            
            # Historical breakdown analysis
            breakdown_periods = rolling_corr < self.breakdown_threshold
            breakdown_count = breakdown_periods.sum()
            breakdown_frequency = breakdown_count / len(rolling_corr) * 100
            
            # Recent breakdown analysis (last lookback_period)
            recent_period = min(self.lookback_period, len(rolling_corr))
            recent_corr = rolling_corr.iloc[-recent_period:]
            recent_breakdowns = (recent_corr < self.breakdown_threshold).sum()
            recent_breakdown_freq = recent_breakdowns / recent_period * 100
            
            # Severity assessment
            if is_breakdown:
                severity = self._assess_breakdown_severity(current_corr)
            else:
                severity = "no_breakdown"
            
            # Risk alert
            risk_alert = self._generate_risk_alert(
                is_breakdown, recent_breakdown_freq, severity)
            
            return {
                "is_breakdown": is_breakdown,
                "breakdown_severity": severity,
                "breakdown_count": int(breakdown_count),
                "breakdown_frequency_pct": round(breakdown_frequency, 2),
                "recent_breakdown_frequency_pct": round(
                    recent_breakdown_freq, 2),
                "risk_alert": risk_alert
            }
            
        except Exception:
            return {
                "is_breakdown": False,
                "breakdown_severity": "unknown",
                "breakdown_count": 0,
                "breakdown_frequency_pct": 0.0,
                "recent_breakdown_frequency_pct": 0.0,
                "risk_alert": "analysis_failed"
            }
    
    def _assess_breakdown_severity(self, correlation: float) -> str:
        """Assess the severity of correlation breakdown."""
        if correlation < 0.0:
            return "severe_negative"
        elif correlation < 0.1:
            return "severe"
        elif correlation < 0.2:
            return "moderate"
        elif correlation < self.breakdown_threshold:
            return "mild"
        else:
            return "no_breakdown"
    
    def _generate_risk_alert(self, is_breakdown: bool, 
                           recent_freq: float, severity: str) -> str:
        """Generate risk alert based on breakdown analysis."""
        if is_breakdown and severity in ["severe", "severe_negative"]:
            return "critical_risk_exit_positions"
        elif is_breakdown and recent_freq > 30:
            return "high_risk_reduce_exposure"
        elif is_breakdown:
            return "moderate_risk_monitor_closely"
        elif recent_freq > 20:
            return "elevated_risk_watch_correlation"
        else:
            return "normal_risk_level"
