"""
Audit Logger for Statistics System.

Provides comprehensive audit logging for all statistics operations.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, date
import threading
from collections import deque


class AuditLogger:
    """
    Handles audit logging for the statistics system.
    
    Features:
    - Comprehensive operation logging
    - Structured audit trails
    - Error tracking and analysis
    - Performance audit metrics
    - Configurable retention policies
    """
    
    def __init__(self, config):
        """
        Initialize audit logger.
        
        Parameters:
        -----------
        config : ConfigurationManager
            Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Audit storage
        self._lock = threading.Lock()
        self.audit_history = deque(maxlen=10000)
        
        # Configuration
        self.audit_file_path = config.get_str(
            'STATISTICS_AUDIT_FILE', 'statistics_audit.log')
        self.retention_days = config.get_int(
            'STATISTICS_AUDIT_RETENTION_DAYS', 30)
        self.detailed_logging = config.get_bool(
            'STATISTICS_AUDIT_DETAILED', True)
        
        # Initialize audit file logger
        self._setup_audit_file_logger()
        
        self.logger.info("AuditLogger initialized")
    
    def _setup_audit_file_logger(self) -> None:
        """Setup dedicated audit file logger."""
        self.audit_file_logger = logging.getLogger('statistics_audit')
        self.audit_file_logger.setLevel(logging.INFO)
        
        # Create file handler if not exists
        if not self.audit_file_logger.handlers:
            handler = logging.FileHandler(self.audit_file_path)
            formatter = logging.Formatter(
                '%(asctime)s - AUDIT - %(message)s')
            handler.setFormatter(formatter)
            self.audit_file_logger.addHandler(handler)
    
    def log_calculation(self, symbol: str, pair_symbol: Optional[str],
                        statistics_count: int, duration: float) -> None:
        """
        Log a statistics calculation operation.
        
        Parameters:
        -----------
        symbol : str
            Primary symbol
        pair_symbol : str, optional
            Second symbol for pair trading
        statistics_count : int
            Number of statistics calculated
        duration : float
            Calculation duration in seconds
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'calculation',
            'operation_type': 'pair' if pair_symbol else 'single',
            'symbol': symbol,
            'pair_symbol': pair_symbol,
            'statistics_count': statistics_count,
            'duration_seconds': duration,
            'status': 'success'
        }
        
        self._write_audit_entry(audit_entry)
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log an audit event.
        
        Parameters:
        -----------
        event_type : str
            Type of the event
        data : Dict[str, Any]
            Event data
        """
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'level': 'INFO',
                'data': data
            }
            
            with self._lock:
                self.audit_history.append(event)
            
            # Log to file
            self.audit_file_logger.info(json.dumps(event))
            
        except Exception as e:
            self.logger.error(f"Failed to log event {event_type}: {e}")
    
    def log_error(self, symbol: str, pair_symbol: Optional[str],
                  error_message: str, operation: str = 'calculation') -> None:
        """
        Log an error occurrence.
        
        Parameters:
        -----------
        symbol : str
            Primary symbol
        pair_symbol : str, optional
            Second symbol for pair trading
        error_message : str
            Error description
        operation : str
            Operation that failed
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'operation_type': 'pair' if pair_symbol else 'single',
            'symbol': symbol,
            'pair_symbol': pair_symbol,
            'status': 'error',
            'error_message': error_message
        }
        
        self._write_audit_entry(audit_entry)
    
    def log_system_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log a system audit event.
        
        Parameters:
        -----------
        event_type : str
            Type of the event
        data : Dict[str, Any]
            Event data
        """
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'level': 'INFO',
                'data': data
            }
            
            with self._lock:
                self.audit_history.append(event)
            
            # Log to file
            self.audit_file_logger.info(json.dumps(event))
            
        except Exception as e:
            self.logger.error(f"Failed to log system event {event_type}: {e}")
    
    def log_system_error(self, error_type: str, error_data: Any) -> None:
        """
        Log a system error event.
        
        Parameters:
        -----------
        error_type : str
            Type of the error
        error_data : Any
            Error data (can be string, dict, etc.)
        """
        try:
            if isinstance(error_data, str):
                data = {'error': error_data}
            else:
                data = error_data if isinstance(error_data, dict) else {'error': str(error_data)}
            
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': error_type,
                'level': 'ERROR',
                'data': data
            }
            
            with self._lock:
                self.audit_history.append(event)
            
            # Log to file
            self.audit_file_logger.error(json.dumps(event))
            
        except Exception as e:
            self.logger.error(f"Failed to log system error {error_type}: {e}")

    def log_database_operation(self, operation: str, table: str,
                               records_affected: int,
                               success: bool) -> None:
        """
        Log database operations.
        
        Parameters:
        -----------
        operation : str
            Database operation type
        table : str
            Target table
        records_affected : int
            Number of records affected
        success : bool
            Operation success status
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'database_operation',
            'db_operation': operation,
            'table': table,
            'records_affected': records_affected,
            'status': 'success' if success else 'error'
        }
        
        self._write_audit_entry(audit_entry)
    
    def log_performance_milestone(self, milestone_type: str,
                                  metrics: Dict[str, Any]) -> None:
        """
        Log performance milestones and metrics.
        
        Parameters:
        -----------
        milestone_type : str
            Type of performance milestone
        metrics : Dict[str, Any]
            Performance metrics data
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'performance_milestone',
            'milestone_type': milestone_type,
            'metrics': metrics,
            'status': 'info'
        }
        
        self._write_audit_entry(audit_entry)
    
    def log_batch_operation(self, operation: str, batch_size: int,
                            successful_items: int, failed_items: int,
                            duration: float) -> None:
        """
        Log batch operation summary.
        
        Parameters:
        -----------
        operation : str
            Batch operation type
        batch_size : int
            Total items in batch
        successful_items : int
            Number of successful items
        failed_items : int
            Number of failed items
        duration : float
            Total batch duration
        """
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'batch_operation',
            'batch_operation': operation,
            'batch_size': batch_size,
            'successful_items': successful_items,
            'failed_items': failed_items,
            'success_rate': (successful_items / batch_size
                             if batch_size > 0 else 0),
            'duration_seconds': duration,
            'status': 'success' if failed_items == 0 else 'partial_failure'
        }
        
        self._write_audit_entry(audit_entry)
    
    def _write_audit_entry(self, audit_entry: Dict[str, Any]) -> None:
        """Write audit entry to storage and file."""
        try:
            # Add to memory storage
            with self._lock:
                self.audit_history.append(audit_entry)
            
            # Write to audit file
            if self.detailed_logging:
                self.audit_file_logger.info(json.dumps(audit_entry))
            else:
                # Simplified logging
                simple_entry = {
                    'timestamp': audit_entry['timestamp'],
                    'operation': audit_entry['operation'],
                    'status': audit_entry['status']
                }
                self.audit_file_logger.info(json.dumps(simple_entry))
            
        except Exception as e:
            self.logger.error(f"Failed to write audit entry: {e}")
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get audit summary for recent period.
        
        Parameters:
        -----------
        hours : int
            Number of hours to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Audit summary statistics
        """
        cutoff_time = (datetime.now().timestamp() - (hours * 3600))
        
        with self._lock:
            recent_entries = [
                entry for entry in self.audit_history
                if datetime.fromisoformat(
                    entry['timestamp']).timestamp() > cutoff_time
            ]
        
        if not recent_entries:
            return {
                'period_hours': hours,
                'total_operations': 0,
                'success_rate': 0
            }
        
        # Count operations by type
        operation_counts = {}
        status_counts = {'success': 0, 'error': 0, 'warning': 0, 'info': 0}
        
        for entry in recent_entries:
            operation = entry.get('operation', 'unknown')
            status = entry.get('status', 'unknown')
            
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate metrics
        total_operations = len(recent_entries)
        success_operations = status_counts['success']
        error_operations = status_counts['error']
        
        return {
            'period_hours': hours,
            'total_operations': total_operations,
            'operation_breakdown': operation_counts,
            'status_breakdown': status_counts,
            'success_rate': (success_operations / total_operations
                             if total_operations > 0 else 0),
            'error_rate': (error_operations / total_operations
                           if total_operations > 0 else 0),
            'recent_errors': self._get_recent_errors(recent_entries)
        }
    
    def _get_recent_errors(self, entries: List[Dict[str, Any]],
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent error entries."""
        error_entries = [
            entry for entry in entries
            if entry.get('status') == 'error'
        ]
        
        # Return most recent errors
        return sorted(error_entries,
                      key=lambda x: x['timestamp'],
                      reverse=True)[:limit]
    
    def get_error_analysis(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze error patterns over specified period.
        
        Parameters:
        -----------
        days : int
            Number of days to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Error analysis report
        """
        cutoff_time = (datetime.now().timestamp() - (days * 24 * 3600))
        
        with self._lock:
            period_entries = [
                entry for entry in self.audit_history
                if datetime.fromisoformat(
                    entry['timestamp']).timestamp() > cutoff_time
            ]
        
        error_entries = [
            entry for entry in period_entries
            if entry.get('status') == 'error'
        ]
        
        if not error_entries:
            return {
                'period_days': days,
                'total_errors': 0,
                'error_patterns': {}
            }
        
        # Analyze error patterns
        error_patterns = {}
        symbol_errors = {}
        operation_errors = {}
        
        for entry in error_entries:
            # Error message patterns
            error_msg = entry.get('error_message', 'Unknown error')
            error_patterns[error_msg] = error_patterns.get(error_msg, 0) + 1
            
            # Symbol-specific errors
            symbol = entry.get('symbol', 'Unknown')
            symbol_errors[symbol] = symbol_errors.get(symbol, 0) + 1
            
            # Operation-specific errors
            operation = entry.get('operation', 'Unknown')
            operation_errors[operation] = operation_errors.get(operation, 0) + 1
        
        return {
            'period_days': days,
            'total_errors': len(error_entries),
            'error_patterns': dict(sorted(error_patterns.items(),
                                          key=lambda x: x[1], reverse=True)[:10]),
            'symbol_errors': dict(sorted(symbol_errors.items(),
                                         key=lambda x: x[1], reverse=True)[:10]),
            'operation_errors': dict(sorted(operation_errors.items(),
                                            key=lambda x: x[1], reverse=True)),
            'error_rate_trend': self._calculate_error_trend(error_entries, days)
        }
    
    def _calculate_error_trend(self, error_entries: List[Dict[str, Any]],
                               days: int) -> List[Dict[str, Any]]:
        """Calculate error trend over the period."""
        # Group errors by day
        daily_errors = {}
        
        for entry in error_entries:
            entry_date = datetime.fromisoformat(entry['timestamp']).date()
            daily_errors[entry_date] = daily_errors.get(entry_date, 0) + 1
        
        # Create trend data
        trend_data = []
        for i in range(days):
            check_date = (datetime.now().date() -
                          __import__('datetime').timedelta(days=i))
            error_count = daily_errors.get(check_date, 0)
            trend_data.append({
                'date': check_date.isoformat(),
                'error_count': error_count
            })
        
        return sorted(trend_data, key=lambda x: x['date'])
    
    def get_summary(self, start_date: Optional[date] = None,
                    end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Get comprehensive audit summary.
        
        Parameters:
        -----------
        start_date : date, optional
            Start date for analysis
        end_date : date, optional
            End date for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive audit summary
        """
        # Default to last 24 hours if no dates specified
        if not start_date and not end_date:
            hours = 24
            recent_summary = self.get_audit_summary(hours)
            error_analysis = self.get_error_analysis(7)
            
            return {
                'summary_period': f'Last {hours} hours',
                'audit_summary': recent_summary,
                'error_analysis': error_analysis,
                'audit_health': self._get_audit_health(recent_summary)
            }
        
        # Date-based analysis would be implemented here
        # For now, return recent summary
        return self.get_audit_summary(24)
    
    def _get_audit_health(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate audit health score."""
        total_ops = summary.get('total_operations', 0)
        success_rate = summary.get('success_rate', 0)
        error_rate = summary.get('error_rate', 0)
        
        # Calculate health score
        health_score = 100
        
        if error_rate > 0.1:  # More than 10% errors
            health_score -= 30
        elif error_rate > 0.05:  # More than 5% errors
            health_score -= 15
        
        if success_rate < 0.9:  # Less than 90% success
            health_score -= 20
        
        if total_ops == 0:
            health_score = 0
        
        health_status = 'healthy'
        if health_score < 50:
            health_status = 'unhealthy'
        elif health_score < 75:
            health_status = 'warning'
        
        return {
            'health_score': max(0, health_score),
            'health_status': health_status,
            'total_operations': total_ops,
            'success_rate': success_rate,
            'error_rate': error_rate
        }
    
    def flush_logs(self) -> None:
        """Flush audit logs to ensure all entries are written."""
        try:
            for handler in self.audit_file_logger.handlers:
                handler.flush()
            self.logger.info("Audit logs flushed")
        except Exception as e:
            self.logger.error(f"Failed to flush audit logs: {e}")
    
    def cleanup_old_entries(self, days_to_keep: int = None) -> int:
        """
        Clean up old audit entries from memory.
        
        Parameters:
        -----------
        days_to_keep : int, optional
            Number of days to retain (uses config default if None)
            
        Returns:
        --------
        int
            Number of entries removed
        """
        if days_to_keep is None:
            days_to_keep = self.retention_days
        
        cutoff_time = (datetime.now().timestamp() -
                       (days_to_keep * 24 * 3600))
        
        with self._lock:
            original_count = len(self.audit_history)
            
            # Filter out old entries
            filtered_entries = [
                entry for entry in self.audit_history
                if datetime.fromisoformat(
                    entry['timestamp']).timestamp() > cutoff_time
            ]
            
            # Update the deque
            self.audit_history.clear()
            self.audit_history.extend(filtered_entries)
            
            removed_count = original_count - len(self.audit_history)
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old audit entries")
        
        return removed_count
