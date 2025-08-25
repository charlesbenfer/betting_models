#!/usr/bin/env python3
"""
Production System Monitor
=========================

Comprehensive monitoring system for the baseball HR prediction system.
Tracks performance, errors, API health, and system resources.
"""

import os
import sys
import time
import json
import psutil
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api_client import SafeAPIClient
from config import config

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    load_average: List[float]
    
@dataclass
class APIMetrics:
    """API performance metrics."""
    timestamp: str
    service: str
    endpoint: str
    response_time_ms: float
    status_code: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class PredictionMetrics:
    """Prediction system metrics."""
    timestamp: str
    predictions_generated: int
    average_prediction_time_ms: float
    betting_opportunities_found: int
    model_type: str
    feature_count: int
    api_calls_made: int
    cache_hit_rate: float

class SystemMonitor:
    """Production system monitor with real-time metrics collection."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Metrics storage (in-memory with configurable retention)
        self.system_metrics = deque(maxlen=retention_hours * 60)  # 1 per minute
        self.api_metrics = deque(maxlen=retention_hours * 240)    # 4 per minute
        self.prediction_metrics = deque(maxlen=retention_hours * 12)  # 1 per 5 minutes
        self.error_log = deque(maxlen=1000)  # Last 1000 errors
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # Setup logging
        self.setup_monitoring_logging()
        
        # Initialize API client for health checks
        api_key = os.getenv("THEODDS_API_KEY", "").strip()
        self.api_client = SafeAPIClient(api_key) if api_key else None
        
        self.logger.info("System monitor initialized")
    
    def setup_monitoring_logging(self):
        """Setup monitoring-specific logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create monitoring logger
        self.logger = logging.getLogger('monitor')
        self.logger.setLevel(logging.INFO)
        
        # File handler for monitoring logs
        monitor_log = log_dir / "monitoring.log"
        handler = logging.FileHandler(monitor_log)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_gb = (memory.total - memory.available) / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Load average
            load_avg = list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
            
            metrics = SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=round(memory_used_gb, 2),
                memory_available_gb=round(memory_available_gb, 2),
                disk_percent=round(disk_percent, 2),
                disk_free_gb=round(disk_free_gb, 2),
                load_average=load_avg
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return None
    
    def test_api_health(self) -> List[APIMetrics]:
        """Test health of all API endpoints."""
        api_tests = []
        
        if not self.api_client:
            return api_tests
        
        # Test The Odds API
        start_time = time.time()
        try:
            events = self.api_client.api_client.get_mlb_events()
            response_time = (time.time() - start_time) * 1000
            
            api_tests.append(APIMetrics(
                timestamp=datetime.now().isoformat(),
                service="theodds_api",
                endpoint="events",
                response_time_ms=round(response_time, 2),
                status_code=200,
                success=True
            ))
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            api_tests.append(APIMetrics(
                timestamp=datetime.now().isoformat(),
                service="theodds_api",
                endpoint="events",
                response_time_ms=round(response_time, 2),
                status_code=500,
                success=False,
                error_message=str(e)
            ))
        
        # Test MLB Stats API
        start_time = time.time()
        try:
            starters = self.api_client.get_probable_starters()
            response_time = (time.time() - start_time) * 1000
            
            api_tests.append(APIMetrics(
                timestamp=datetime.now().isoformat(),
                service="mlb_stats_api",
                endpoint="probable_starters",
                response_time_ms=round(response_time, 2),
                status_code=200,
                success=True
            ))
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            api_tests.append(APIMetrics(
                timestamp=datetime.now().isoformat(),
                service="mlb_stats_api",
                endpoint="probable_starters",
                response_time_ms=round(response_time, 2),
                status_code=500,
                success=False,
                error_message=str(e)
            ))
        
        return api_tests
    
    def record_prediction_metrics(self, metrics: PredictionMetrics):
        """Record prediction system performance metrics."""
        self.prediction_metrics.append(metrics)
        
        # Update counters
        self.counters['total_predictions'] += metrics.predictions_generated
        self.counters['total_opportunities'] += metrics.betting_opportunities_found
        self.counters['api_calls'] += metrics.api_calls_made
        
        self.logger.info(f"Prediction metrics recorded: {metrics.predictions_generated} predictions, "
                        f"{metrics.betting_opportunities_found} opportunities")
    
    def record_error(self, error_type: str, error_message: str, component: str = "unknown"):
        """Record system error for tracking."""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'component': component
        }
        
        self.error_log.append(error_record)
        self.counters['total_errors'] += 1
        
        self.logger.error(f"Error recorded - {component}: {error_type} - {error_message}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        now = datetime.now()
        
        # Recent system metrics (last 5 minutes)
        recent_system = [m for m in self.system_metrics 
                        if datetime.fromisoformat(m.timestamp) > now - timedelta(minutes=5)]
        
        # Recent API metrics (last 5 minutes) 
        recent_api = [m for m in self.api_metrics
                     if datetime.fromisoformat(m.timestamp) > now - timedelta(minutes=5)]
        
        # Recent errors (last hour)
        recent_errors = [e for e in self.error_log
                        if datetime.fromisoformat(e['timestamp']) > now - timedelta(hours=1)]
        
        health_status = {
            'timestamp': now.isoformat(),
            'overall_status': 'healthy',
            'components': {
                'system_resources': self._assess_system_health(recent_system),
                'api_connectivity': self._assess_api_health(recent_api),
                'error_rate': self._assess_error_rate(recent_errors)
            },
            'metrics_summary': {
                'system_metrics_count': len(self.system_metrics),
                'api_metrics_count': len(self.api_metrics),
                'prediction_metrics_count': len(self.prediction_metrics),
                'total_errors': len(self.error_log)
            },
            'counters': dict(self.counters)
        }
        
        # Determine overall status
        component_statuses = [comp['status'] for comp in health_status['components'].values()]
        if 'critical' in component_statuses:
            health_status['overall_status'] = 'critical'
        elif 'warning' in component_statuses:
            health_status['overall_status'] = 'warning'
        
        return health_status
    
    def _assess_system_health(self, recent_metrics: List[SystemMetrics]) -> Dict[str, Any]:
        """Assess system resource health."""
        if not recent_metrics:
            return {'status': 'unknown', 'message': 'No recent metrics'}
        
        latest = recent_metrics[-1]
        
        # Define thresholds
        if latest.cpu_percent > 90 or latest.memory_percent > 90 or latest.disk_percent > 90:
            status = 'critical'
            message = f"Resource usage critical: CPU {latest.cpu_percent}%, Memory {latest.memory_percent}%, Disk {latest.disk_percent}%"
        elif latest.cpu_percent > 70 or latest.memory_percent > 70 or latest.disk_percent > 80:
            status = 'warning'
            message = f"Resource usage high: CPU {latest.cpu_percent}%, Memory {latest.memory_percent}%, Disk {latest.disk_percent}%"
        else:
            status = 'healthy'
            message = f"Resources normal: CPU {latest.cpu_percent}%, Memory {latest.memory_percent}%, Disk {latest.disk_percent}%"
        
        return {
            'status': status,
            'message': message,
            'latest_metrics': asdict(latest)
        }
    
    def _assess_api_health(self, recent_metrics: List[APIMetrics]) -> Dict[str, Any]:
        """Assess API connectivity health."""
        if not recent_metrics:
            return {'status': 'unknown', 'message': 'No recent API tests'}
        
        success_count = sum(1 for m in recent_metrics if m.success)
        total_count = len(recent_metrics)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / total_count if total_count > 0 else 0
        
        if success_rate < 0.5:
            status = 'critical'
            message = f"API health critical: {success_rate:.1%} success rate"
        elif success_rate < 0.8 or avg_response_time > 5000:
            status = 'warning' 
            message = f"API health degraded: {success_rate:.1%} success, {avg_response_time:.0f}ms avg response"
        else:
            status = 'healthy'
            message = f"APIs healthy: {success_rate:.1%} success, {avg_response_time:.0f}ms avg response"
        
        return {
            'status': status,
            'message': message,
            'success_rate': success_rate,
            'average_response_time_ms': round(avg_response_time, 2),
            'total_tests': total_count
        }
    
    def _assess_error_rate(self, recent_errors: List[Dict]) -> Dict[str, Any]:
        """Assess system error rate."""
        error_count = len(recent_errors)
        
        if error_count > 50:
            status = 'critical'
            message = f"High error rate: {error_count} errors in last hour"
        elif error_count > 10:
            status = 'warning'
            message = f"Elevated error rate: {error_count} errors in last hour"
        else:
            status = 'healthy'
            message = f"Normal error rate: {error_count} errors in last hour"
        
        # Group errors by type
        error_types = defaultdict(int)
        for error in recent_errors:
            error_types[error['type']] += 1
        
        return {
            'status': status,
            'message': message,
            'error_count': error_count,
            'error_types': dict(error_types)
        }
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring in background thread."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"Started monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop running in background thread."""
        api_check_counter = 0
        
        while self.monitoring_active:
            try:
                # Collect system metrics every interval
                system_metrics = self.collect_system_metrics()
                if system_metrics:
                    self.system_metrics.append(system_metrics)
                
                # Test API health every 5 intervals (5 minutes if interval is 60s)
                api_check_counter += 1
                if api_check_counter >= 5:
                    api_tests = self.test_api_health()
                    self.api_metrics.extend(api_tests)
                    api_check_counter = 0
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.record_error("monitoring_error", str(e), "system_monitor")
                time.sleep(interval_seconds)
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Clean system metrics
        while (self.system_metrics and 
               datetime.fromisoformat(self.system_metrics[0].timestamp) < cutoff_time):
            self.system_metrics.popleft()
        
        # Clean API metrics  
        while (self.api_metrics and
               datetime.fromisoformat(self.api_metrics[0].timestamp) < cutoff_time):
            self.api_metrics.popleft()
    
    def export_metrics(self, output_dir: str = "logs"):
        """Export current metrics to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export system metrics
        system_data = [asdict(m) for m in self.system_metrics]
        with open(output_path / f"system_metrics_{timestamp}.json", 'w') as f:
            json.dump(system_data, f, indent=2)
        
        # Export API metrics
        api_data = [asdict(m) for m in self.api_metrics]
        with open(output_path / f"api_metrics_{timestamp}.json", 'w') as f:
            json.dump(api_data, f, indent=2)
        
        # Export health status
        health_data = self.get_system_health()
        with open(output_path / f"health_status_{timestamp}.json", 'w') as f:
            json.dump(health_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_path}")

def main():
    """Main monitoring function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='System Monitor for Baseball HR Prediction')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=int, default=3600, help='Duration to run in seconds')
    parser.add_argument('--export', action='store_true', help='Export metrics at end')
    
    args = parser.parse_args()
    
    # Create and start monitor
    monitor = SystemMonitor()
    
    try:
        print(f"Starting monitoring for {args.duration} seconds...")
        monitor.start_monitoring(args.interval)
        
        # Run for specified duration
        time.sleep(args.duration)
        
        # Print final health status
        health = monitor.get_system_health()
        print(f"\nFinal System Health: {health['overall_status'].upper()}")
        
        if args.export:
            monitor.export_metrics()
            print("Metrics exported to logs/")
        
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()