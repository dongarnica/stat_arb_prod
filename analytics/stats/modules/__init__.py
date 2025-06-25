"""
Modular statistics modules for financial analysis.

This package contains specialized statistics modules organized by category:
- basic: Fundamental statistics
- correlation: Correlation analysis
- mean_reversion: Mean reversion statistics
- pair_trading: Pair trading specific statistics
- risk: Risk assessment metrics
- statistical: Advanced statistical tests
- technical: Technical analysis indicators
"""

# Import main modules for easy access
from . import basic
from . import correlation
from . import mean_reversion
from . import pair_trading
from . import risk
from . import statistical
from . import technical

__all__ = [
    'basic',
    'correlation',
    'mean_reversion',
    'pair_trading',
    'risk',
    'statistical',
    'technical'
]
