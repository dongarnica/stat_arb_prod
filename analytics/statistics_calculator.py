"""
Statistics Calculator - Main Entry Point.

This module provides the main interface for calculating financial statistics
across all available modules, with support for both single-asset and pair
trading analysis. Results are stored in PostgreSQL for AI model consumption.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

try:
    # Try relative imports first (when used as module)
    from .stats.orchestrator import StatisticsOrchestrator
    from .stats.database_manager import DatabaseManager
    from .stats.configuration_manager import ConfigurationManager
    from .stats.result_validator import ResultValidator
    from .stats.performance_monitor import PerformanceMonitor
    from .stats.audit_logger import AuditLogger
    from .stats.data_manager import DataManager
except ImportError:
    # Fall back to direct imports (when run directly)
    from stats.orchestrator import StatisticsOrchestrator
    from stats.database_manager import DatabaseManager
    from stats.configuration_manager import ConfigurationManager
    from stats.result_validator import ResultValidator
    from stats.performance_monitor import PerformanceMonitor
    from stats.audit_logger import AuditLogger
    from stats.data_manager import DataManager


class StatisticsCalculator:
    """
    Main statistics calculator that orchestrates all statistic calculations.
    
    Features:
    - Dynamic loading of all available statistic modules
    - Batch processing for multiple symbols
    - Parallel execution of independent calculations
    - Database persistence with transaction management
    - Comprehensive error handling and logging
    - Performance monitoring and caching
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the statistics calculator.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to custom configuration file
        """
        # Initialize core components
        self.config = ConfigurationManager(config_path)
        self.db_manager = DatabaseManager(self.config)
        self.orchestrator = StatisticsOrchestrator(self.config)
        self.validator = ResultValidator(self.config)
        self.performance = PerformanceMonitor(self.config)
        self.audit = AuditLogger(self.config)
        self.data_manager = DataManager(self.config)
        
        # Initialize logging
        self._setup_logging()
        
        # Validate system readiness
        self._validate_system()
        
        self.logger.info("StatisticsCalculator initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config.get('STATISTICS_LOG_LEVEL', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('statistics_calculator.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _validate_system(self) -> None:
        """Validate system readiness before processing."""
        try:
            # Check database connectivity
            self.db_manager.validate_connection()
            
            # Verify statistics table exists
            self.db_manager.ensure_schema()
            
            # Load and validate statistic modules
            modules_loaded = self.orchestrator.load_modules()
            self.logger.info(f"Loaded {modules_loaded} statistic modules")
            
            # Test data access
            self.data_manager.validate_data_access()
            
        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            raise RuntimeError(f"System not ready: {e}")
    
    def calculate_single_asset_statistics(self, 
                                        symbol: str,
                                        start_date: Optional[date] = None,
                                        end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Calculate all enabled statistics for a single asset.
        
        Parameters:
        -----------
        symbol : str
            ETF symbol to analyze
        start_date : date, optional
            Start date for analysis
        end_date : date, optional
            End date for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Results summary with success/failure details
        """
        start_time = time.time()
        self.logger.info(f"Starting single asset calculation for {symbol}")
        
        try:
            # Retrieve price data
            price_data = self.data_manager.get_asset_data(
                symbol, start_date, end_date)
            
            if price_data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return self._create_empty_result(symbol)
            
            # Execute single-asset statistics
            results = self.orchestrator.calculate_single_asset_statistics(
                price_data, symbol)
            
            # Validate and process results
            validated_results = self._process_calculation_results(
                results, symbol, None, price_data)
            
            # Write to database if not in dry-run mode
            if not self.config.get_bool('STATISTICS_DRY_RUN', False):
                self.db_manager.write_statistics(validated_results)
            
            # Log performance metrics
            duration = time.time() - start_time
            self.performance.record_calculation(symbol, None, duration)
            
            # Audit log
            self.audit.log_calculation(symbol, None, 
                                     len(validated_results), duration)
            
            self.logger.info(
                f"Completed single asset calculation for {symbol} "
                f"in {duration:.2f}s")
            
            return {
                'symbol': symbol,
                'status': 'success',
                'statistics_calculated': len(validated_results),
                'duration_seconds': duration,
                'data_points': len(price_data)
            }
            
        except Exception as e:
            self.logger.error(
                f"Single asset calculation failed for {symbol}: {e}")
            self.audit.log_error(symbol, None, str(e))
            
            return {
                'symbol': symbol,
                'status': 'error',
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }
    
    def calculate_pair_statistics(self, 
                                symbol1: str, 
                                symbol2: str,
                                start_date: Optional[date] = None,
                                end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Calculate all enabled pair trading statistics.
        
        Parameters:
        -----------
        symbol1 : str
            First ETF symbol
        symbol2 : str
            Second ETF symbol
        start_date : date, optional
            Start date for analysis
        end_date : date, optional
            End date for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Results summary with success/failure details
        """
        start_time = time.time()
        pair_key = f"{symbol1}_{symbol2}"
        self.logger.info(f"Starting pair calculation for {pair_key}")
        
        try:
            # Retrieve price data for both assets
            data1 = self.data_manager.get_asset_data(
                symbol1, start_date, end_date)
            data2 = self.data_manager.get_asset_data(
                symbol2, start_date, end_date)
            
            if data1.empty or data2.empty:
                self.logger.warning(f"Insufficient data for pair {pair_key}")
                return self._create_empty_result(symbol1, symbol2)
            
            # Align data for pair analysis
            pair_data = self.data_manager.align_pair_data(data1, data2)
            
            # Execute pair trading statistics
            results = self.orchestrator.calculate_pair_statistics(
                pair_data, symbol1, symbol2)
            
            # Validate and process results
            validated_results = self._process_calculation_results(
                results, symbol1, symbol2, pair_data)
            
            # Write to database if not in dry-run mode
            if not self.config.get_bool('STATISTICS_DRY_RUN', False):
                self.db_manager.write_statistics(validated_results)
            
            # Log performance metrics
            duration = time.time() - start_time
            self.performance.record_calculation(symbol1, symbol2, duration)
            
            # Audit log
            self.audit.log_calculation(symbol1, symbol2, 
                                     len(validated_results), duration)
            
            self.logger.info(
                f"Completed pair calculation for {pair_key} "
                f"in {duration:.2f}s")
            
            return {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'status': 'success',
                'statistics_calculated': len(validated_results),
                'duration_seconds': duration,
                'data_points': len(pair_data)
            }
            
        except Exception as e:
            self.logger.error(
                f"Pair calculation failed for {pair_key}: {e}")
            self.audit.log_error(symbol1, symbol2, str(e))
            
            return {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'status': 'error',
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }
    
    def calculate_batch(self, 
                       symbols: List[str],
                       pairs: Optional[List[Tuple[str, str]]] = None,
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Calculate statistics for multiple symbols and pairs in batch.
        
        Parameters:
        -----------
        symbols : List[str]
            List of ETF symbols for single-asset analysis
        pairs : List[Tuple[str, str]], optional
            List of symbol pairs for pair trading analysis
        start_date : date, optional
            Start date for analysis
        end_date : date, optional
            End date for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Batch processing results summary
        """
        start_time = time.time()
        total_tasks = len(symbols) + (len(pairs) if pairs else 0)
        
        self.logger.info(
            f"Starting batch calculation: {len(symbols)} symbols, "
            f"{len(pairs) if pairs else 0} pairs")
        
        results = {
            'single_asset_results': [],
            'pair_results': [],
            'summary': {
                'total_tasks': total_tasks,
                'successful_tasks': 0,
                'failed_tasks': 0,
                'start_time': datetime.now(),
                'duration_seconds': 0
            }
        }
        
        # Get parallel processing configuration
        max_workers = self.config.get_int('STATISTICS_PARALLEL_WORKERS', 4)
        batch_size = self.config.get_int('STATISTICS_BATCH_SIZE', 10)
        
        try:
            # Process single-asset statistics
            if symbols:
                single_results = self._process_batch_single_assets(
                    symbols, start_date, end_date, max_workers, batch_size)
                results['single_asset_results'] = single_results
            
            # Process pair statistics
            if pairs:
                pair_results = self._process_batch_pairs(
                    pairs, start_date, end_date, max_workers, batch_size)
                results['pair_results'] = pair_results
            
            # Calculate summary statistics
            all_results = results['single_asset_results'] + results['pair_results']
            successful = sum(1 for r in all_results if r['status'] == 'success')
            failed = sum(1 for r in all_results if r['status'] == 'error')
            
            results['summary'].update({
                'successful_tasks': successful,
                'failed_tasks': failed,
                'duration_seconds': time.time() - start_time,
                'end_time': datetime.now()
            })
            
            self.logger.info(
                f"Batch calculation completed: {successful} successful, "
                f"{failed} failed, {time.time() - start_time:.2f}s total")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch calculation failed: {e}")
            results['summary']['error'] = str(e)
            return results
    
    def _process_batch_single_assets(self, 
                                   symbols: List[str],
                                   start_date: Optional[date],
                                   end_date: Optional[date],
                                   max_workers: int,
                                   batch_size: int) -> List[Dict[str, Any]]:
        """Process single assets in parallel batches."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks in batches
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                # Submit batch tasks
                future_to_symbol = {
                    executor.submit(
                        self.calculate_single_asset_statistics,
                        symbol, start_date, end_date
                    ): symbol for symbol in batch_symbols
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.debug(f"Completed {symbol}: {result['status']}")
                    except Exception as e:
                        self.logger.error(f"Task failed for {symbol}: {e}")
                        results.append({
                            'symbol': symbol,
                            'status': 'error',
                            'error': str(e)
                        })
        
        return results
    
    def _process_batch_pairs(self, 
                           pairs: List[Tuple[str, str]],
                           start_date: Optional[date],
                           end_date: Optional[date],
                           max_workers: int,
                           batch_size: int) -> List[Dict[str, Any]]:
        """Process pairs in parallel batches."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks in batches
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # Submit batch tasks
                future_to_pair = {
                    executor.submit(
                        self.calculate_pair_statistics,
                        symbol1, symbol2, start_date, end_date
                    ): (symbol1, symbol2) for symbol1, symbol2 in batch_pairs
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_pair):
                    symbol1, symbol2 = future_to_pair[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.logger.debug(
                            f"Completed {symbol1}_{symbol2}: {result['status']}")
                    except Exception as e:
                        self.logger.error(
                            f"Task failed for {symbol1}_{symbol2}: {e}")
                        results.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'status': 'error',
                            'error': str(e)
                        })
        
        return results
    
    def _process_calculation_results(self, 
                                   results: List[Dict[str, Any]],
                                   symbol1: str,
                                   symbol2: Optional[str],
                                   data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process and validate calculation results."""
        processed_results = []
        
        for result in results:
            try:
                # Validate result structure and values
                validated_result = self.validator.validate_result(result)
                
                # Add metadata
                validated_result.update({
                    'symbol': symbol1,
                    'pair_symbol': symbol2,
                    'calculation_timestamp': datetime.now(),
                    'data_start_date': data.index[0].date() if not data.empty else None,
                    'data_end_date': data.index[-1].date() if not data.empty else None,
                    'data_points_used': len(data)
                })
                
                processed_results.append(validated_result)
                
            except Exception as e:
                self.logger.warning(
                    f"Result validation failed for {result.get('statistic_name', 'unknown')}: {e}")
                continue
        
        return processed_results
    
    def _create_empty_result(self, symbol1: str, 
                           symbol2: Optional[str] = None) -> Dict[str, Any]:
        """Create empty result for insufficient data."""
        return {
            'symbol1' if symbol2 else 'symbol': symbol1,
            'symbol2': symbol2,
            'status': 'no_data',
            'statistics_calculated': 0,
            'data_points': 0,
            'error': 'Insufficient data available'
        }
    
    def generate_summary_report(self, 
                              start_date: Optional[date] = None,
                              end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Generate summary report of calculated statistics.
        
        Parameters:
        -----------
        start_date : date, optional
            Start date for report period
        end_date : date, optional
            End date for report period
            
        Returns:
        --------
        Dict[str, Any]
            Summary report with statistics and performance metrics
        """
        self.logger.info("Generating summary report")
        
        try:
            # Get statistics summary from database
            stats_summary = self.db_manager.get_statistics_summary(
                start_date, end_date)
            
            # Get performance metrics
            perf_metrics = self.performance.get_summary_metrics(
                start_date, end_date)
            
            # Get audit summary
            audit_summary = self.audit.get_summary(start_date, end_date)
            
            report = {
                'report_generated': datetime.now(),
                'period_start': start_date,
                'period_end': end_date,
                'statistics_summary': stats_summary,
                'performance_metrics': perf_metrics,
                'audit_summary': audit_summary,
                'system_health': self._get_system_health()
            }
            
            self.logger.info("Summary report generated successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"Summary report generation failed: {e}")
            return {'error': str(e)}
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        return {
            'database_connection': self.db_manager.check_health(),
            'modules_loaded': self.orchestrator.get_module_count(),
            'cache_status': self.data_manager.get_cache_status(),
            'memory_usage': self.performance.get_memory_usage()
        }
    
    def cleanup(self) -> None:
        """Cleanup resources and connections."""
        self.logger.info("Cleaning up resources")
        
        try:
            self.db_manager.close_connections()
            self.data_manager.clear_cache()
            self.performance.save_metrics()
            self.audit.flush_logs()
            
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
