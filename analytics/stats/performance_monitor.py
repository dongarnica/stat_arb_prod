"""
Performance Monitor for Statistics System.

Tracks performance metrics, memory usage, and system health.
"""

import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, date
from collections import defaultdict, deque
import threading


class PerformanceMonitor:
    """
    Monitors performance metrics for the statistics system.
    
    Features:
    - Execution time tracking
    - Memory usage monitoring
    - Throughput calculations
    - System health metrics
    - Performance history
    """
    
    def __init__(self, config):
        """
        Initialize performance monitor.
        
        Parameters:
        -----------
        config : ConfigurationManager
            Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance data storage
        self._lock = threading.Lock()
        self.calculation_times = defaultdict(list)
        self.memory_history = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        # Configuration
        self.max_history_size = config.get_int(
            'STATISTICS_PERF_HISTORY_SIZE', 1000)
        self.memory_check_interval = config.get_int(
            'STATISTICS_MEMORY_CHECK_INTERVAL', 60)
        
        # Start monitoring thread
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("PerformanceMonitor initialized")
    
    def record_calculation(self, symbol: str, pair_symbol: Optional[str],
                           duration: float) -> None:
        """
        Record calculation performance metrics.
        
        Parameters:
        -----------
        symbol : str
            Primary symbol
        pair_symbol : str, optional
            Second symbol for pair trading
        duration : float
            Calculation duration in seconds
        """
        with self._lock:
            key = f"{symbol}_{pair_symbol}" if pair_symbol else symbol
            self.calculation_times[key].append({
                'timestamp': datetime.now(),
                'duration': duration,
                'type': 'pair' if pair_symbol else 'single'
            })
            
            # Limit history size
            if len(self.calculation_times[key]) > self.max_history_size:
                self.calculation_times[key] = (
                    self.calculation_times[key][-self.max_history_size:])
    
    def get_calculation_stats(self, symbol: Optional[str] = None
                              ) -> Dict[str, Any]:
        """
        Get calculation performance statistics.
        
        Parameters:
        -----------
        symbol : str, optional
            Specific symbol to analyze (None for all)
            
        Returns:
        --------
        Dict[str, Any]
            Performance statistics
        """
        with self._lock:
            if symbol:
                # Stats for specific symbol
                if symbol not in self.calculation_times:
                    return {'error': f'No data for symbol {symbol}'}
                
                times = [entry['duration'] for entry in
                         self.calculation_times[symbol]]
                return self._calculate_stats(times, symbol)
            else:
                # Overall stats
                all_times = []
                for entries in self.calculation_times.values():
                    all_times.extend([entry['duration'] for entry in entries])
                
                return self._calculate_stats(all_times, 'overall')
    
    def _calculate_stats(self, times: List[float], context: str
                         ) -> Dict[str, Any]:
        """Calculate statistical metrics from timing data."""
        if not times:
            return {'context': context, 'count': 0}
        
        times_sorted = sorted(times)
        count = len(times)
        
        return {
            'context': context,
            'count': count,
            'min_duration': min(times),
            'max_duration': max(times),
            'avg_duration': sum(times) / count,
            'median_duration': (
                times_sorted[count // 2] if count % 2 == 1
                else (times_sorted[count // 2 - 1] + times_sorted[
                    count // 2]) / 2
            ),
            'p95_duration': times_sorted[int(count * 0.95)] if count > 0 else 0,
            'p99_duration': times_sorted[int(count * 0.99)] if count > 0 else 0,
            'total_duration': sum(times)
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            return {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'process_memory_percent': memory_percent,
                'system_memory_total_gb': (system_memory.total /
                                           1024 / 1024 / 1024),
                'system_memory_available_gb': (system_memory.available /
                                               1024 / 1024 / 1024),
                'system_memory_percent': system_memory.percent,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Failed to get memory usage: {e}")
            return {'error': str(e)}
    
    def record_throughput(self, operations_count: int,
                          duration: float) -> None:
        """
        Record throughput metrics.
        
        Parameters:
        -----------
        operations_count : int
            Number of operations completed
        duration : float
            Time taken in seconds
        """
        throughput = operations_count / duration if duration > 0 else 0
        
        with self._lock:
            self.throughput_history.append({
                'timestamp': datetime.now(),
                'operations': operations_count,
                'duration': duration,
                'throughput': throughput
            })
    
    def get_throughput_stats(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Get throughput statistics for recent period.
        
        Parameters:
        -----------
        minutes : int
            Number of minutes to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Throughput statistics
        """
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        
        with self._lock:
            recent_data = [
                entry for entry in self.throughput_history
                if entry['timestamp'].timestamp() > cutoff_time
            ]
        
        if not recent_data:
            return {
                'period_minutes': minutes,
                'data_points': 0,
                'avg_throughput': 0
            }
        
        throughputs = [entry['throughput'] for entry in recent_data]
        total_operations = sum(entry['operations'] for entry in recent_data)
        total_duration = sum(entry['duration'] for entry in recent_data)
        
        return {
            'period_minutes': minutes,
            'data_points': len(recent_data),
            'total_operations': total_operations,
            'total_duration': total_duration,
            'avg_throughput': sum(throughputs) / len(throughputs),
            'max_throughput': max(throughputs),
            'min_throughput': min(throughputs),
            'overall_throughput': (total_operations / total_duration
                                   if total_duration > 0 else 0)
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory_info = self.get_memory_usage()
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            
            # Recent performance
            recent_calcs = self.get_calculation_stats()
            recent_throughput = self.get_throughput_stats(30)
            
            # Health score calculation
            health_score = self._calculate_health_score(
                cpu_percent, memory_info, recent_calcs, recent_throughput)
            
            return {
                'timestamp': datetime.now(),
                'health_score': health_score,
                'cpu_percent': cpu_percent,
                'memory_info': memory_info,
                'disk_usage_percent': ((disk_usage.used /
                                        disk_usage.total) * 100),
                'recent_calculations': recent_calcs,
                'recent_throughput': recent_throughput
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {'error': str(e), 'health_score': 0}
    
    def _calculate_health_score(self, cpu_percent: float,
                                memory_info: Dict[str, Any],
                                calc_stats: Dict[str, Any],
                                throughput_stats: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # CPU penalty
        if cpu_percent > 80:
            score -= 30
        elif cpu_percent > 60:
            score -= 15
        
        # Memory penalty
        if 'process_memory_percent' in memory_info:
            mem_percent = memory_info['process_memory_percent']
            if mem_percent > 80:
                score -= 25
            elif mem_percent > 60:
                score -= 10
        
        # Performance penalty
        if calc_stats.get('count', 0) > 0:
            avg_duration = calc_stats.get('avg_duration', 0)
            if avg_duration > 10:  # seconds
                score -= 20
            elif avg_duration > 5:
                score -= 10
        
        # Throughput bonus/penalty
        if throughput_stats.get('data_points', 0) > 0:
            avg_throughput = throughput_stats.get('avg_throughput', 0)
            if avg_throughput > 50:  # operations per second
                score += 5
            elif avg_throughput < 10:
                score -= 10
        
        return max(0, min(100, score))
    
    def _background_monitoring(self) -> None:
        """Background thread for continuous monitoring."""
        while self._monitoring_active:
            try:
                # Record memory usage
                memory_info = self.get_memory_usage()
                if 'error' not in memory_info:
                    with self._lock:
                        self.memory_history.append(memory_info)
                
                # Sleep for configured interval
                time.sleep(self.memory_check_interval)
                
            except Exception as e:
                self.logger.error(f"Background monitoring error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def get_summary_metrics(self, start_date: Optional[date] = None,
                            end_date: Optional[date] = None
                            ) -> Dict[str, Any]:
        """
        Get summary performance metrics for a period.
        
        Parameters:
        -----------
        start_date : date, optional
            Start date for analysis
        end_date : date, optional
            End date for analysis
            
        Returns:
        --------
        Dict[str, Any]
            Summary performance metrics
        """
        # Current system health
        health = self.get_system_health()
        
        # Overall calculation stats
        calc_stats = self.get_calculation_stats()
        
        # Recent throughput
        throughput = self.get_throughput_stats(1440)  # 24 hours
        
        # Memory statistics
        memory_stats = self._get_memory_stats()
        
        return {
            'report_period': {
                'start_date': start_date,
                'end_date': end_date
            },
            'current_health': health,
            'calculation_performance': calc_stats,
            'throughput_metrics': throughput,
            'memory_statistics': memory_stats,
            'monitoring_status': {
                'monitoring_active': self._monitoring_active,
                'memory_history_size': len(self.memory_history),
                'calculation_history_size': sum(
                    len(entries) for entries in self.calculation_times.values())
            }
        }
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics from history."""
        with self._lock:
            if not self.memory_history:
                return {'error': 'No memory history available'}
            
            memory_values = [
                entry['process_memory_mb'] for entry in self.memory_history
                if 'process_memory_mb' in entry
            ]
            
            if not memory_values:
                return {'error': 'No valid memory data'}
            
            # Simple trend calculation
            if len(memory_values) > 10:
                recent_avg = sum(memory_values[-5:]) / 5
                older_avg = sum(memory_values[-10:-5]) / 5
                trend = 'increasing' if recent_avg > older_avg else 'stable'
            else:
                trend = 'stable'
            
            return {
                'current_memory_mb': memory_values[-1],
                'avg_memory_mb': sum(memory_values) / len(memory_values),
                'max_memory_mb': max(memory_values),
                'min_memory_mb': min(memory_values),
                'memory_trend': trend,
                'samples_count': len(memory_values)
            }
    
    def save_metrics(self) -> None:
        """Save performance metrics for persistence."""
        # In a production system, this would save to database or file
        self.logger.info("Performance metrics saved")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
