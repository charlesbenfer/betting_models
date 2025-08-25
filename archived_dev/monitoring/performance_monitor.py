#!/usr/bin/env python3
"""
Performance Monitoring System
============================

Tracks performance metrics, response times, and system efficiency.
"""

import time
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
import json
import logging

@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    timestamp: str
    operation: str
    component: str
    duration_ms: float
    success: bool
    details: Optional[Dict[str, Any]] = None

@dataclass
class ThroughputMetric:
    """Throughput measurement over time window."""
    timestamp: str
    operation: str
    count: int
    time_window_seconds: int
    rate_per_second: float

class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self, max_metrics: int = 50000):
        self.max_metrics = max_metrics
        self.metrics = deque(maxlen=max_metrics)
        self.throughput_metrics = deque(maxlen=max_metrics // 10)
        
        # Real-time counters
        self.operation_counters = defaultdict(int)
        self.operation_timers = defaultdict(list)
        
        # Active operations (for tracking concurrent operations)
        self.active_operations = {}
        self.lock = threading.Lock()
        
        # Setup logging
        self.setup_performance_logging()
        
        self.logger.info("Performance monitor initialized")
    
    def setup_performance_logging(self):
        """Setup performance-specific logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('performance_monitor')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_dir / "performance.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
    
    def start_operation(self, operation: str, component: str, 
                       operation_id: Optional[str] = None) -> str:
        """Start timing an operation."""
        if operation_id is None:
            operation_id = f"{operation}_{component}_{int(time.time()*1000)}"
        
        with self.lock:
            self.active_operations[operation_id] = {
                'operation': operation,
                'component': component,
                'start_time': time.time(),
                'start_timestamp': datetime.now().isoformat()
            }
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True,
                     details: Optional[Dict[str, Any]] = None):
        """End timing an operation and record metrics."""
        with self.lock:
            if operation_id not in self.active_operations:
                self.logger.warning(f"Unknown operation ID: {operation_id}")
                return
            
            op_data = self.active_operations.pop(operation_id)
        
        # Calculate duration
        end_time = time.time()
        duration_ms = (end_time - op_data['start_time']) * 1000
        
        # Create metric
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            operation=op_data['operation'],
            component=op_data['component'],
            duration_ms=round(duration_ms, 3),
            success=success,
            details=details
        )
        
        # Store metric
        self.metrics.append(metric)
        
        # Update counters
        op_key = f"{op_data['component']}.{op_data['operation']}"
        self.operation_counters[op_key] += 1
        self.operation_timers[op_key].append(duration_ms)
        
        # Keep only recent timings (last 1000 per operation)
        if len(self.operation_timers[op_key]) > 1000:
            self.operation_timers[op_key] = self.operation_timers[op_key][-1000:]
        
        self.logger.debug(f"Operation completed: {op_key} in {duration_ms:.1f}ms")
    
    def record_throughput(self, operation: str, count: int, 
                         time_window_seconds: int = 60):
        """Record throughput metric."""
        rate = count / time_window_seconds if time_window_seconds > 0 else 0
        
        throughput_metric = ThroughputMetric(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            count=count,
            time_window_seconds=time_window_seconds,
            rate_per_second=round(rate, 3)
        )
        
        self.throughput_metrics.append(throughput_metric)
        
        self.logger.info(f"Throughput recorded: {operation} - {rate:.1f}/sec ({count} in {time_window_seconds}s)")
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        if not recent_metrics:
            return {
                'time_period_hours': hours,
                'total_operations': 0,
                'operations_by_component': {},
                'average_response_times': {},
                'success_rates': {}
            }
        
        # Group by operation
        by_operation = defaultdict(list)
        for metric in recent_metrics:
            op_key = f"{metric.component}.{metric.operation}"
            by_operation[op_key].append(metric)
        
        # Calculate statistics
        operations_by_component = defaultdict(int)
        average_response_times = {}
        success_rates = {}
        p95_response_times = {}
        p99_response_times = {}
        
        for op_key, metrics_list in by_operation.items():
            component = metrics_list[0].component
            operations_by_component[component] += len(metrics_list)
            
            # Response time statistics
            durations = [m.duration_ms for m in metrics_list]
            average_response_times[op_key] = round(statistics.mean(durations), 2)
            
            if len(durations) >= 20:  # Only calculate percentiles if we have enough data
                sorted_durations = sorted(durations)
                p95_idx = int(len(sorted_durations) * 0.95)
                p99_idx = int(len(sorted_durations) * 0.99)
                p95_response_times[op_key] = round(sorted_durations[p95_idx], 2)
                p99_response_times[op_key] = round(sorted_durations[p99_idx], 2)
            
            # Success rate
            successful = sum(1 for m in metrics_list if m.success)
            success_rates[op_key] = round(successful / len(metrics_list), 3)
        
        return {
            'time_period_hours': hours,
            'total_operations': len(recent_metrics),
            'operations_by_component': dict(operations_by_component),
            'average_response_times_ms': average_response_times,
            'p95_response_times_ms': p95_response_times,
            'p99_response_times_ms': p99_response_times,
            'success_rates': success_rates,
            'active_operations_count': len(self.active_operations)
        }
    
    def get_slow_operations(self, threshold_ms: float = 1000, 
                           hours: int = 1) -> List[Dict[str, Any]]:
        """Get operations that exceeded response time threshold."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        slow_operations = []
        for metric in self.metrics:
            if (datetime.fromisoformat(metric.timestamp) > cutoff_time and
                metric.duration_ms > threshold_ms):
                slow_operations.append({
                    'timestamp': metric.timestamp,
                    'operation': f"{metric.component}.{metric.operation}",
                    'duration_ms': metric.duration_ms,
                    'success': metric.success,
                    'details': metric.details
                })
        
        # Sort by duration (slowest first)
        slow_operations.sort(key=lambda x: x['duration_ms'], reverse=True)
        
        return slow_operations
    
    def get_throughput_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get throughput summary for time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_throughput = [
            m for m in self.throughput_metrics
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        if not recent_throughput:
            return {'time_period_hours': hours, 'throughput_data': {}}
        
        # Group by operation
        by_operation = defaultdict(list)
        for metric in recent_throughput:
            by_operation[metric.operation].append(metric)
        
        throughput_data = {}
        for operation, metrics_list in by_operation.items():
            rates = [m.rate_per_second for m in metrics_list]
            throughput_data[operation] = {
                'average_rate_per_second': round(statistics.mean(rates), 2),
                'max_rate_per_second': round(max(rates), 2),
                'min_rate_per_second': round(min(rates), 2),
                'measurement_count': len(metrics_list)
            }
        
        return {
            'time_period_hours': hours,
            'throughput_data': throughput_data
        }
    
    def check_performance_thresholds(self) -> List[Dict[str, Any]]:
        """Check if any performance thresholds are exceeded."""
        issues = []
        
        # Define thresholds
        thresholds = {
            'api_client.get_odds': 5000,  # 5 seconds for odds API
            'api_client.get_starters': 3000,  # 3 seconds for MLB API
            'model_system.predict': 1000,  # 1 second for predictions
            'feature_calculation.batch': 10000,  # 10 seconds for batch features
            'database.query': 500  # 500ms for database queries
        }
        
        summary = self.get_performance_summary(hours=1)
        
        for operation, threshold_ms in thresholds.items():
            avg_response = summary['average_response_times_ms'].get(operation)
            if avg_response and avg_response > threshold_ms:
                issues.append({
                    'operation': operation,
                    'average_response_time_ms': avg_response,
                    'threshold_ms': threshold_ms,
                    'exceeded_by_ms': round(avg_response - threshold_ms, 2),
                    'severity': 'warning' if avg_response < threshold_ms * 2 else 'critical'
                })
        
        return issues
    
    def export_performance_data(self, output_dir: str = "logs", hours: int = 24):
        """Export performance data to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Export metrics
        recent_metrics = [
            asdict(m) for m in self.metrics
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        # Export throughput data
        recent_throughput = [
            asdict(m) for m in self.throughput_metrics
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"performance_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_period_hours': hours,
            'summary': self.get_performance_summary(hours),
            'throughput_summary': self.get_throughput_summary(hours),
            'slow_operations': self.get_slow_operations(hours=hours),
            'performance_issues': self.check_performance_thresholds(),
            'metrics': recent_metrics,
            'throughput_metrics': recent_throughput
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Performance data exported to {filename}")

class PerformanceDecorator:
    """Decorator for automatic performance monitoring."""
    
    def __init__(self, monitor: PerformanceMonitor, component: str):
        self.monitor = monitor
        self.component = component
    
    def __call__(self, operation: str):
        """Decorator that monitors function performance."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                op_id = self.monitor.start_operation(operation, self.component)
                
                try:
                    result = func(*args, **kwargs)
                    self.monitor.end_operation(op_id, success=True)
                    return result
                except Exception as e:
                    self.monitor.end_operation(op_id, success=False, 
                                             details={'error': str(e)})
                    raise
            
            return wrapper
        return decorator

def main():
    """Example usage of performance monitoring."""
    # Create performance monitor
    monitor = PerformanceMonitor()
    
    # Create decorator for easy use
    api_perf = PerformanceDecorator(monitor, 'api_client')
    
    @api_perf('get_odds')
    def simulate_api_call():
        """Simulate API call with variable delay."""
        import random
        time.sleep(random.uniform(0.1, 2.0))
        return "api_response"
    
    # Simulate some operations
    print("Simulating performance monitoring...")
    
    for i in range(20):
        try:
            result = simulate_api_call()
            
            # Also demonstrate manual timing
            op_id = monitor.start_operation('process_result', 'data_processor')
            time.sleep(0.05)  # Simulate processing
            monitor.end_operation(op_id, success=True)
            
        except Exception as e:
            print(f"Operation failed: {e}")
    
    # Record some throughput
    monitor.record_throughput('predictions_generated', 150, 60)
    monitor.record_throughput('api_calls_completed', 45, 60)
    
    # Get performance summary
    summary = monitor.get_performance_summary(hours=1)
    print(f"\nPerformance Summary:")
    print(f"Total Operations: {summary['total_operations']}")
    print(f"Average Response Times: {summary['average_response_times_ms']}")
    print(f"Success Rates: {summary['success_rates']}")
    
    # Check for slow operations
    slow_ops = monitor.get_slow_operations(threshold_ms=500, hours=1)
    print(f"\nSlow Operations (>500ms): {len(slow_ops)}")
    
    # Check thresholds
    issues = monitor.check_performance_thresholds()
    print(f"Performance Issues: {len(issues)}")
    
    # Export data
    monitor.export_performance_data(hours=1)
    print("Performance data exported to logs/")

if __name__ == "__main__":
    main()