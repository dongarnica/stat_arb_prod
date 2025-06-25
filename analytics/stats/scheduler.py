"""
Statistical Arbitrage Scheduler.

Coordinates daily cointegration analysis and hourly statistical calculations
for efficient pair trading analysis.
"""

import logging
import schedule
import time
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
import json
import sys
import os

from .configuration_manager import ConfigurationManager
from .cointegration_manager import CointegrationManager
from .hourly_statistics_manager import HourlyStatisticsManager
from .audit_logger import AuditLogger


class StatisticalArbitrageScheduler:
    """
    Main scheduler for statistical arbitrage analysis.
    
    Features:
    - Daily cointegration analysis scheduling
    - Hourly statistical calculations
    - Error handling and recovery
    - Performance monitoring
    - Audit logging
    """
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize the scheduler.
        
        Parameters:
        -----------
        config : ConfigurationManager
            Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize managers
        self.cointegration_manager = CointegrationManager(config)
        self.hourly_manager = HourlyStatisticsManager(config)
        self.audit_logger = AuditLogger(config)
        
        # Configuration
        self.daily_time = config.get_str('DAILY_ANALYSIS_TIME', '06:00')
        self.hourly_enabled = config.get_bool('HOURLY_ANALYSIS_ENABLED', True)
        self.max_retries = config.get_int('MAX_RETRIES', 3)
        self.retry_delay = config.get_int('RETRY_DELAY_MINUTES', 30)
        
        self.logger.info("StatisticalArbitrageScheduler initialized")
    
    def setup_schedules(self) -> None:
        """Setup the daily and hourly schedules."""
        try:
            # Schedule daily cointegration analysis
            schedule.every().day.at(self.daily_time).do(
                self._run_daily_analysis_with_retry
            )
            
            if self.hourly_enabled:
                # Schedule hourly analysis (every hour during trading hours)
                trading_hours = self._get_trading_hours()
                for hour in trading_hours:
                    schedule.every().day.at(f"{hour:02d}:00").do(
                        self._run_hourly_analysis_with_retry
                    )
            
            self.logger.info(
                f"Schedules setup: Daily at {self.daily_time}, "
                f"Hourly: {self.hourly_enabled}"
            )
            
        except Exception as e:
            self.logger.error(f"Error setting up schedules: {e}")
            raise
    
    def run_scheduler(self) -> None:
        """Run the main scheduler loop."""
        self.logger.info("Starting scheduler...")
        
        try:
            # Initialize database tables
            self._initialize_tables()
            
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
            raise
    
    def run_daily_analysis_now(self, symbols: Optional[list] = None) -> Dict[str, Any]:
        """
        Run daily cointegration analysis immediately.
        
        Parameters:
        -----------
        symbols : list, optional
            List of symbols to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        self.logger.info("Running daily analysis on demand")
        return self._run_daily_analysis(symbols)
    
    def run_hourly_analysis_now(self, as_of_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Run hourly analysis immediately.
        
        Parameters:
        -----------
        as_of_date : date, optional
            Date for cointegrated pairs lookup
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        self.logger.info("Running hourly analysis on demand")
        return self._run_hourly_analysis(as_of_date)
    
    def _run_daily_analysis_with_retry(self) -> None:
        """Run daily analysis with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._run_daily_analysis()
                return
            except Exception as e:
                self.logger.error(
                    f"Daily analysis attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    self.logger.info(
                        f"Retrying in {self.retry_delay} minutes..."
                    )
                    time.sleep(self.retry_delay * 60)
                else:
                    self.logger.error("All daily analysis attempts failed")
                    self.audit_logger.log_system_error(
                        "daily_analysis_failed",
                        f"Failed after {self.max_retries} attempts: {e}"
                    )
    
    def _run_hourly_analysis_with_retry(self) -> None:
        """Run hourly analysis with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._run_hourly_analysis()
                return
            except Exception as e:
                self.logger.error(
                    f"Hourly analysis attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying in 15 minutes...")
                    time.sleep(15 * 60)  # Shorter retry for hourly
                else:
                    self.logger.error("All hourly analysis attempts failed")
                    self.audit_logger.log_system_error(
                        "hourly_analysis_failed",
                        f"Failed after {self.max_retries} attempts: {e}"
                    )
    
    def _run_daily_analysis(self, symbols: Optional[list] = None) -> Dict[str, Any]:
        """Run the daily cointegration analysis."""
        start_time = datetime.now()
        
        try:
            self.audit_logger.log_system_event(
                "daily_analysis_started",
                {"start_time": start_time.isoformat()}
            )
            
            # Run cointegration analysis
            results = self.cointegration_manager.run_daily_cointegration_analysis(
                symbols
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log success
            self.audit_logger.log_system_event(
                "daily_analysis_completed",
                {
                    "duration_seconds": duration,
                    "results": results
                }
            )
            
            self.logger.info(
                f"Daily analysis completed in {duration:.2f} seconds. "
                f"Found {results['cointegrated_pairs']} cointegrated pairs."
            )
            
            return results
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.audit_logger.log_system_error(
                "daily_analysis_error",
                {
                    "duration_seconds": duration,
                    "error": str(e)
                }
            )
            
            raise
    
    def _run_hourly_analysis(self, as_of_date: Optional[date] = None) -> Dict[str, Any]:
        """Run the hourly statistical analysis."""
        start_time = datetime.now()
        
        try:
            self.audit_logger.log_system_event(
                "hourly_analysis_started",
                {"start_time": start_time.isoformat()}
            )
            
            # Run hourly analysis
            results = self.hourly_manager.run_hourly_analysis(as_of_date)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log success
            self.audit_logger.log_system_event(
                "hourly_analysis_completed",
                {
                    "duration_seconds": duration,
                    "results": results
                }
            )
            
            self.logger.info(
                f"Hourly analysis completed in {duration:.2f} seconds. "
                f"Analyzed {results['pairs_analyzed']} pairs, "
                f"generated {results['signals_generated']} signals."
            )
            
            return results
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.audit_logger.log_system_error(
                "hourly_analysis_error",
                {
                    "duration_seconds": duration,
                    "error": str(e)
                }
            )
            
            raise
    
    def _get_trading_hours(self) -> list:
        """Get trading hours for hourly analysis."""
        # Default US market hours (9:30 AM - 4:00 PM EST)
        # Adjust based on your market requirements
        trading_hours_config = self.config.get_str(
            'TRADING_HOURS', '9,10,11,12,13,14,15,16'
        )
        
        try:
            hours = [int(h.strip()) for h in trading_hours_config.split(',')]
            return sorted(hours)
        except Exception as e:
            self.logger.warning(f"Invalid trading hours config: {e}")
            return [9, 10, 11, 12, 13, 14, 15, 16]  # Default hours
    
    def _initialize_tables(self) -> None:
        """Initialize all required database tables."""
        try:
            self.logger.info("Initializing database tables...")
            
            self.cointegration_manager.create_tables()
            self.hourly_manager.create_tables()
            
            self.logger.info("Database tables initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing tables: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and recent performance."""
        try:
            # Get recent analysis results
            today = date.today()
            yesterday = today - timedelta(days=1)
            
            # Get cointegrated pairs count
            cointegrated_pairs = self.cointegration_manager.get_cointegrated_pairs(
                today
            )
            
            # Get recent signals
            current_signals = self.hourly_manager.get_current_signals()
            
            # Get recent audit events
            recent_events = self.audit_logger.get_recent_events(hours=24)
            
            status = {
                'system_time': datetime.now().isoformat(),
                'cointegrated_pairs_today': len(cointegrated_pairs),
                'current_signals': len(current_signals),
                'recent_events': len(recent_events),
                'scheduler_running': True,
                'next_daily_analysis': self._get_next_scheduled_time('daily'),
                'next_hourly_analysis': self._get_next_scheduled_time('hourly'),
                'configuration': {
                    'daily_time': self.daily_time,
                    'hourly_enabled': self.hourly_enabled,
                    'trading_hours': self._get_trading_hours()
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                'system_time': datetime.now().isoformat(),
                'error': str(e),
                'scheduler_running': False
            }
    
    def _get_next_scheduled_time(self, analysis_type: str) -> Optional[str]:
        """Get next scheduled time for analysis type."""
        try:
            if analysis_type == 'daily':
                for job in schedule.jobs:
                    if 'daily_analysis' in str(job.job_func):
                        return job.next_run.isoformat() if job.next_run else None
            elif analysis_type == 'hourly':
                hourly_jobs = [
                    job for job in schedule.jobs
                    if 'hourly_analysis' in str(job.job_func)
                ]
                if hourly_jobs:
                    next_job = min(hourly_jobs, key=lambda x: x.next_run or datetime.max)
                    return next_job.next_run.isoformat() if next_job.next_run else None
            
            return None
            
        except Exception:
            return None


def main():
    """Main entry point for the scheduler."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('statistical_arbitrage_scheduler.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration
        config = ConfigurationManager()
        
        # Initialize scheduler
        scheduler = StatisticalArbitrageScheduler(config)
        
        # Setup schedules
        scheduler.setup_schedules()
        
        # Run scheduler
        scheduler.run_scheduler()
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
