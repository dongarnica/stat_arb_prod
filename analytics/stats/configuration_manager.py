"""
Configuration Manager for Statistics System.

Handles environment variables, .env files, and runtime settings.
"""

import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None


class ConfigurationManager:
    """
    Manages configuration for the statistics system.
    
    Features:
    - Environment variable support
    - .env file loading via python-dotenv
    - JSON configuration files
    - Type-safe getters with defaults
    - Runtime configuration updates
    """
    
    def __init__(self, config_path: Optional[str] = None,
                 env_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to JSON configuration file
        env_file : str, optional
            Path to .env file (defaults to .env in current directory)
        """
        self.logger = logging.getLogger(__name__)
        self.config_data: Dict[str, Any] = {}
        
        # Load from .env file first
        self._load_env_file(env_file)
        
        # Load from JSON file if provided (overrides .env)
        if config_path and Path(config_path).exists():
            self._load_config_file(config_path)
        
        # Load remaining environment variables
        self._load_environment_variables()
        
        self.logger.info("ConfigurationManager initialized")
    
    def _load_env_file(self, env_file: Optional[str] = None) -> None:
        """Load configuration from .env file."""
        if not DOTENV_AVAILABLE:
            self.logger.warning(
                "python-dotenv not available, skipping .env file loading")
            return
        
        # Determine .env file path
        if env_file:
            env_path = Path(env_file)
        else:
            # Look for .env in current directory and parent directories
            current_dir = Path.cwd()
            env_path = current_dir / '.env'
            
            # Also check parent directories up to 3 levels
            if not env_path.exists():
                for parent in current_dir.parents[:3]:
                    potential_path = parent / '.env'
                    if potential_path.exists():
                        env_path = potential_path
                        break
        
        if env_path.exists():
            try:
                load_dotenv(env_path)
                self.logger.info(f"Loaded .env file from {env_path}")
            except Exception as e:
                self.logger.error(f"Failed to load .env file {env_path}: {e}")
        else:
            self.logger.debug("No .env file found")
    
    def _load_config_file(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                self.config_data.update(file_config)
            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_path}: {e}")
    
    def _load_environment_variables(self) -> None:
        """Load relevant environment variables."""
        # Direct mappings from .env variables
        env_vars = [
            'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_SSLMODE',
            'MLFLOW_TRACKING_URI', 'MLFLOW_EXPERIMENT_NAME', 'MLFLOW_ARTIFACT_ROOT',
            'ASSET_SYMBOLS', 'DEFAULT_TIMEFRAME', 'CORRELATION_THRESHOLD', 
            'COINTEGRATION_SIGNIFICANCE_LEVEL', 'COINTEGRATION_MIN_OBSERVATIONS',
            'COINTEGRATION_LOOKBACK_DAYS', 'COINTEGRATION_TEST_METHOD',
            'RISK_FREE_RATE', 'MAX_DRAWDOWN_THRESHOLD',
            'VOLATILITY_LOOKBACK', 'MIN_OBSERVATIONS', 'LOG_LEVEL', 'LOG_FORMAT',
            'ENVIRONMENT', 'CACHE_ENABLED', 'BATCH_SIZE', 'PARALLEL_PROCESSING',
            'SECRET_KEY', 'API_RATE_LIMIT', 'DATA_RETENTION_DAYS',
            'STRATEGY_NAME', 'STRATEGY_TYPE', 'STRATEGY_SYMBOLS', 'STRATEGY_TIMEFRAME',
            'INITIAL_CAPITAL', 'ENTRY_ZSCORE', 'EXIT_ZSCORE', 'MIN_SIGNAL_STRENGTH',
            'MAX_POSITION_PCT',
            # Hourly Analysis Settings
            'ZSCORE_WINDOW', 'MA_SHORT_WINDOW', 'MA_LONG_WINDOW', 
            'HOURLY_LOOKBACK_HOURS', 'ZSCORE_THRESHOLD'
        ]
        
        for env_var in env_vars:
            value = os.getenv(env_var)
            if value is not None:
                self.config_data[env_var] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.config_data.get(key, default)
    
    def get_str(self, key: str, default: str = '') -> str:
        """Get string configuration value."""
        value = self.get(key, default)
        return str(value) if value is not None else default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer configuration value."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float configuration value."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean configuration value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value) if value is not None else default
    
    def get_list(self, key: str, default: List[Any] = None, separator: str = ',') -> List[Any]:
        """Get list configuration value."""
        if default is None:
            default = []
        
        value = self.get(key, default)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # Try to parse as JSON list or separated values
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                # Try separator-separated
                return [item.strip() for item in value.split(separator)
                        if item.strip()]
        
        return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value at runtime."""
        self.config_data[key] = value
        self.logger.debug(f"Configuration updated: {key} = {value}")
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update multiple configuration values."""
        self.config_data.update(config_dict)
        self.logger.debug(f"Configuration updated with {len(config_dict)} items")
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database connection configuration."""
        return {
            'host': self.get_str('DB_HOST', 'localhost'),
            'port': self.get_int('DB_PORT', 5432),
            'database': self.get_str('DB_NAME', 'trading_db'),
            'user': self.get_str('DB_USER', 'postgres'),
            'password': self.get_str('DB_PASSWORD', ''),
            'sslmode': self.get_str('DB_SSLMODE', 'prefer')
        }
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return {
            'tracking_uri': self.get_str('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
            'experiment_name': self.get_str('MLFLOW_EXPERIMENT_NAME', 'statistics_experiment'),
            'artifact_root': self.get_str('MLFLOW_ARTIFACT_ROOT', './mlruns')
        }
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return {
            'asset_symbols': self.get_list('ASSET_SYMBOLS', ['SPY', 'QQQ']),
            'default_timeframe': self.get_str('DEFAULT_TIMEFRAME', '1 day'),
            'correlation_threshold': self.get_float('CORRELATION_THRESHOLD', 0.7),
            'cointegration_significance': self.get_float('COINTEGRATION_SIGNIFICANCE', 0.05),
            'min_observations': self.get_int('MIN_OBSERVATIONS', 252)
        }
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk configuration."""
        return {
            'risk_free_rate': self.get_float('RISK_FREE_RATE', 0.045),
            'max_drawdown_threshold': self.get_float('MAX_DRAWDOWN_THRESHOLD', 0.15),
            'volatility_lookback': self.get_int('VOLATILITY_LOOKBACK', 30)
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'level': self.get_str('LOG_LEVEL', 'INFO'),
            'format': self.get_str(
                'LOG_FORMAT',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration."""
        return {
            'batch_size': self.get_int('BATCH_SIZE', 50),
            'parallel_processing': self.get_bool('PARALLEL_PROCESSING', True),
            'cache_enabled': self.get_bool('CACHE_ENABLED', True)
        }
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration."""
        return {
            'name': self.get_str('STRATEGY_NAME', 'default_strategy'),
            'type': self.get_str('STRATEGY_TYPE', 'pairs_trading'),
            'symbols': self.get_list('STRATEGY_SYMBOLS', ['SPY', 'IVV']),
            'timeframe': self.get_str('STRATEGY_TIMEFRAME', '1 day'),
            'initial_capital': self.get_float('INITIAL_CAPITAL', 30000.0),
            'entry_zscore': self.get_float('ENTRY_ZSCORE', 2.0),
            'exit_zscore': self.get_float('EXIT_ZSCORE', 0.25),
            'min_signal_strength': self.get_float('MIN_SIGNAL_STRENGTH', 2.2),
            'max_position_pct': self.get_float('MAX_POSITION_PCT', 0.31)
        }
    
    def validate_required_config(self) -> bool:
        """
        Validate that all required configuration is present.
        
        Returns:
        --------
        bool
            True if all required config is valid
        """
        required_keys = [
            'DB_HOST',
            'DB_NAME',
            'DB_USER'
        ]
        
        missing_keys = []
        for key in required_keys:
            if not self.get_str(key):
                missing_keys.append(key)
        
        if missing_keys:
            self.logger.error(
                f"Missing required configuration: {missing_keys}")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self.config_data.copy()
    
    def __str__(self) -> str:
        """String representation (excluding sensitive values)."""
        safe_config = {}
        sensitive_keys = ['password', 'secret', 'key', 'token']
        
        for key, value in self.config_data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                safe_config[key] = '[REDACTED]'
            else:
                safe_config[key] = value
        
        return json.dumps(safe_config, indent=2)
