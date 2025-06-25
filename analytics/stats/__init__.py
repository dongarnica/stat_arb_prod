"""
Statistics System Package.

This package provides a comprehensive statistics calculation system for
financial data analysis, with support for both single-asset and pair
trading statistics.
"""

from .base_statistic import BaseStatistic
from .configuration_manager import ConfigurationManager
from .database_manager import DatabaseManager
from .result_validator import ResultValidator
from .performance_monitor import PerformanceMonitor
from .audit_logger import AuditLogger
from .data_manager import DataManager
from .orchestrator import StatisticsOrchestrator

__all__ = [
    'BaseStatistic',
    'StatisticsOrchestrator',
    'ConfigurationManager',
    'DatabaseManager',
    'ResultValidator',
    'PerformanceMonitor',
    'AuditLogger',
    'DataManager'
]

__version__ = '1.0.0'
