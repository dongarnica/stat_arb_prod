"""Analytics package - Modern statistics system."""

__version__ = "1.0.0"

from .stats import (
    BaseStatistic,
    StatisticsOrchestrator,
    ConfigurationManager,
    DatabaseManager,
    ResultValidator,
    PerformanceMonitor,
    AuditLogger,
    DataManager
)

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
