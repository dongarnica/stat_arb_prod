#!/usr/bin/env python3
"""
Strategy Configuration Switcher
Easily switch between different strategy profiles by updating .env file.
"""

import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StrategyConfigSwitcher:
    """Manages switching between different strategy configurations."""
    
    def __init__(self):
        self.env_file = Path('.env')
        self.backup_dir = Path('.env_backups')
        self.backup_dir.mkdir(exist_ok=True)
        
        # Predefined strategy profiles
        self.profiles = {
            'conservative': {
                'ENTRY_ZSCORE': '2.2',
                'EXIT_ZSCORE': '0.3',
                'MAX_POSITION_PCT': '0.27',
                'BASE_POSITION_PCT': '0.23',
                'LOOKBACK_WINDOW': '60',
                'PORTFOLIO_STOP_LOSS': '0.03',
                'TRADE_STOP_LOSS': '0.01',
                'MAX_SIZING_MULTIPLIER': '1.5',
                'BACKTEST_DAYS': '365'
            },
            'balanced': {
                'ENTRY_ZSCORE': '2.0',
                'EXIT_ZSCORE': '0.25',
                'MAX_POSITION_PCT': '0.31',
                'BASE_POSITION_PCT': '0.26',
                'LOOKBACK_WINDOW': '50',
                'PORTFOLIO_STOP_LOSS': '0.04',
                'TRADE_STOP_LOSS': '0.012',
                'MAX_SIZING_MULTIPLIER': '1.7',
                'BACKTEST_DAYS': '365'
            },
            'aggressive': {
                'ENTRY_ZSCORE': '1.8',
                'EXIT_ZSCORE': '0.2',
                'MAX_POSITION_PCT': '0.35',
                'BASE_POSITION_PCT': '0.30',
                'LOOKBACK_WINDOW': '45',
                'PORTFOLIO_STOP_LOSS': '0.05',
                'TRADE_STOP_LOSS': '0.015',
                'MAX_SIZING_MULTIPLIER': '2.0',
                'BACKTEST_DAYS': '450'
            },
            'enhanced': {
                'ENTRY_ZSCORE': '1.8',
                'EXIT_ZSCORE': '0.2',
                'MAX_POSITION_PCT': '0.35',
                'BASE_POSITION_PCT': '0.30',
                'LOOKBACK_WINDOW': '45',
                'PORTFOLIO_STOP_LOSS': '0.05',
                'TRADE_STOP_LOSS': '0.015',
                'MAX_SIZING_MULTIPLIER': '2.0',
                'TRAILING_STOP_ENABLED': 'true',
                'TRAILING_STOP_PCT': '0.005',
                'PROFIT_TARGET_MULTIPLIER': '1.5',
                'BACKTEST_DAYS': '450'
            }
        }
    
    def backup_current_config(self):
        """Create a backup of the current .env file."""
        if self.env_file.exists():
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f'.env_backup_{timestamp}'
            shutil.copy2(self.env_file, backup_file)
            logger.info(f"Current configuration backed up to {backup_file}")
            return backup_file
        else:
            logger.warning("No .env file found to backup")
            return None
    
    def switch_to_profile(self, profile_name: str):
        """Switch to a predefined strategy profile."""
        if profile_name not in self.profiles:
            available = ', '.join(self.profiles.keys())
            raise ValueError(f"Unknown profile '{profile_name}'. Available: {available}")
        
        # Backup current configuration
        self.backup_current_config()
        
        # Read current .env file
        current_config = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        current_config[key] = value
        
        # Update with profile parameters
        profile_config = self.profiles[profile_name]
        current_config.update(profile_config)
        
        # Update strategy name to reflect profile
        current_config['STRATEGY_NAME'] = f"SPY_IVV_Pairs_30K_{profile_name.title()}"
        
        # Write updated configuration
        self._write_env_file(current_config)
        
        logger.info(f"Switched to '{profile_name}' strategy profile")
        self._log_profile_summary(profile_name, profile_config)
    
    def _write_env_file(self, config_dict: dict):
        """Write configuration dictionary to .env file."""
        with open(self.env_file, 'w') as f:
            f.write("# Environment Variables for Stat Arb Backtest\n\n")
            
            # Group configurations
            sections = {
                'Database': ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_SSLMODE'],
                'MLflow': ['MLFLOW_TRACKING_URI', 'MLFLOW_EXPERIMENT_NAME', 'MLFLOW_ARTIFACT_ROOT'],
                'Analysis': ['ASSET_SYMBOLS', 'DEFAULT_TIMEFRAME', 'CORRELATION_THRESHOLD', 'COINTEGRATION_SIGNIFICANCE'],
                'Risk': ['RISK_FREE_RATE', 'MAX_DRAWDOWN_THRESHOLD', 'VOLATILITY_LOOKBACK', 'MIN_OBSERVATIONS'],
                'System': ['LOG_LEVEL', 'LOG_FORMAT', 'ENVIRONMENT', 'CACHE_ENABLED', 'BATCH_SIZE', 'PARALLEL_PROCESSING'],
                'Security': ['SECRET_KEY', 'API_RATE_LIMIT', 'DATA_RETENTION_DAYS'],
                'Strategy': ['STRATEGY_NAME', 'STRATEGY_TYPE', 'STRATEGY_SYMBOLS', 'STRATEGY_TIMEFRAME', 'INITIAL_CAPITAL',
                           'ENTRY_ZSCORE', 'EXIT_ZSCORE', 'MIN_SIGNAL_STRENGTH', 'MAX_POSITION_PCT', 'MIN_POSITION_PCT',
                           'BASE_POSITION_PCT', 'DYNAMIC_SIZING', 'MAX_SIZING_MULTIPLIER', 'PORTFOLIO_STOP_LOSS',
                           'TRADE_STOP_LOSS', 'DAILY_LOSS_LIMIT', 'LOOKBACK_WINDOW', 'COMMISSION_PER_TRADE',
                           'BID_ASK_SPREAD_PCT', 'VOLUME_CONFIRMATION', 'PROFIT_TARGET_MULTIPLIER',
                           'TRAILING_STOP_ENABLED', 'TRAILING_STOP_PCT', 'BACKTEST_DAYS', 'COMMISSION_MODEL']
            }
            
            for section_name, keys in sections.items():
                f.write(f"# {section_name} configuration\n")
                for key in keys:
                    if key in config_dict:
                        f.write(f"{key}={config_dict[key]}\n")
                f.write("\n")
            
            # Write any remaining keys
            written_keys = set()
            for keys in sections.values():
                written_keys.update(keys)
            
            remaining_keys = set(config_dict.keys()) - written_keys
            if remaining_keys:
                f.write("# Additional configuration\n")
                for key in sorted(remaining_keys):
                    f.write(f"{key}={config_dict[key]}\n")
    
    def _log_profile_summary(self, profile_name: str, profile_config: dict):
        """Log a summary of the profile configuration."""
        logger.info(f"=== {profile_name.upper()} PROFILE SUMMARY ===")
        logger.info(f"  Entry Z-Score: {profile_config.get('ENTRY_ZSCORE', 'N/A')}")
        logger.info(f"  Exit Z-Score: {profile_config.get('EXIT_ZSCORE', 'N/A')}")
        logger.info(f"  Max Position: {float(profile_config.get('MAX_POSITION_PCT', 0))*100:.1f}%")
        logger.info(f"  Base Position: {float(profile_config.get('BASE_POSITION_PCT', 0))*100:.1f}%")
        logger.info(f"  Lookback Window: {profile_config.get('LOOKBACK_WINDOW', 'N/A')} days")
        logger.info(f"  Portfolio Stop: {float(profile_config.get('PORTFOLIO_STOP_LOSS', 0))*100:.1f}%")
        logger.info(f"  Trade Stop: {float(profile_config.get('TRADE_STOP_LOSS', 0))*100:.1f}%")
    
    def list_profiles(self):
        """List all available strategy profiles."""
        logger.info("Available strategy profiles:")
        for name, config in self.profiles.items():
            logger.info(f"\n{name.upper()}:")
            self._log_profile_summary(name, config)
    
    def show_current_config(self):
        """Show current strategy configuration from .env file."""
        if not self.env_file.exists():
            logger.warning("No .env file found")
            return
        
        logger.info("Current strategy configuration:")
        strategy_keys = ['STRATEGY_NAME', 'ENTRY_ZSCORE', 'EXIT_ZSCORE', 'MAX_POSITION_PCT', 
                        'BASE_POSITION_PCT', 'LOOKBACK_WINDOW', 'PORTFOLIO_STOP_LOSS']
        
        with open(self.env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key in strategy_keys:
                        logger.info(f"  {key}: {value}")


def main():
    """Main CLI interface for strategy configuration switching."""
    import sys
    
    switcher = StrategyConfigSwitcher()
    
    if len(sys.argv) < 2:
        print("Usage: python config_switcher.py [list|current|switch <profile>]")
        print("Example: python config_switcher.py switch balanced")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        switcher.list_profiles()
    elif command == 'current':
        switcher.show_current_config()
    elif command == 'switch':
        if len(sys.argv) < 3:
            print("Please specify a profile to switch to")
            switcher.list_profiles()
            return
        profile = sys.argv[2].lower()
        try:
            switcher.switch_to_profile(profile)
            print(f"\n✅ Successfully switched to '{profile}' profile")
            print("Run 'python run_parameterized_backtest.py' to test the new configuration")
        except ValueError as e:
            print(f"❌ Error: {e}")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: list, current, switch <profile>")


if __name__ == "__main__":
    main()
