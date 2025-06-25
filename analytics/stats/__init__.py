"""
Statistics System Package.

This package provides a comprehensive statistics calculation system for
financial data analysis, with support for both single-asset and pair
trading statistics, including cointegration analysis and hourly monitoring.
"""

from .base_statistic import BaseStatistic
from .configuration_manager import ConfigurationManager
from .database_manager import DatabaseManager
from .result_validator import ResultValidator
from .performance_monitor import PerformanceMonitor
from .audit_logger import AuditLogger
from .data_manager import DataManager
from .orchestrator import StatisticsOrchestrator
from .cointegration_manager import CointegrationManager
from .hourly_statistics_manager import HourlyStatisticsManager
from .scheduler import StatisticalArbitrageScheduler

__all__ = [
    'BaseStatistic',
    'StatisticsOrchestrator',
    'ConfigurationManager',
    'DatabaseManager',
    'ResultValidator',
    'PerformanceMonitor',
    'AuditLogger',
    'DataManager',
    'CointegrationManager',
    'HourlyStatisticsManager',
    'StatisticalArbitrageScheduler'
]

__version__ = '1.1.0'
