#!/usr/bin/env python3
"""
Health Check System
==================

Comprehensive health checks for all system components.
"""

import os
import sys
import time
import json
import psutil
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from api_client import SafeAPIClient
from config import config
from modeling import EnhancedDualModelSystem

@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: str  # healthy, warning, critical, unknown
    response_time_ms: float
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class HealthChecker:
    """Comprehensive system health checker."""
    
    def __init__(self):
        self.api_key = os.getenv("THEODDS_API_KEY", "").strip()
        self.health_history = []
        
    def check_system_resources(self) -> HealthCheck:
        """Check system resource health."""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            status = "healthy"
            issues = []
            
            if cpu_percent > 90:
                status = "critical"
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                status = "warning"
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            if memory.percent > 90:
                status = "critical"
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > 80:
                if status != "critical":
                    status = "warning"
                issues.append(f"Memory usage high: {memory.percent:.1f}%")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 95:
                status = "critical"
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > 85:
                if status != "critical":
                    status = "warning"
                issues.append(f"Disk usage high: {disk_percent:.1f}%")
            
            message = "; ".join(issues) if issues else "System resources normal"
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_percent': round(disk_percent, 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'load_average': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None
            }
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="system_resources",
                status=status,
                response_time_ms=round(response_time, 2),
                message=message,
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="system_resources",
                status="critical",
                response_time_ms=round(response_time, 2),
                message=f"Failed to check system resources: {e}",
                details={'error': str(e)}
            )
    
    def check_api_connectivity(self) -> List[HealthCheck]:
        """Check connectivity to all external APIs."""
        checks = []
        
        if not self.api_key:
            return [HealthCheck(
                name="api_connectivity",
                status="critical",
                response_time_ms=0,
                message="No API key configured"
            )]
        
        try:
            api_client = SafeAPIClient(self.api_key)
            
            # Test The Odds API
            start_time = time.time()
            try:
                odds_available = api_client.is_odds_available()
                if odds_available:
                    # Try to get actual data
                    odds_df = api_client.get_todays_odds()
                    response_time = (time.time() - start_time) * 1000
                    
                    checks.append(HealthCheck(
                        name="theodds_api",
                        status="healthy",
                        response_time_ms=round(response_time, 2),
                        message=f"API operational, {len(odds_df)} odds available",
                        details={'odds_count': len(odds_df)}
                    ))
                else:
                    response_time = (time.time() - start_time) * 1000
                    checks.append(HealthCheck(
                        name="theodds_api",
                        status="warning",
                        response_time_ms=round(response_time, 2),
                        message="API not available"
                    ))
                    
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                checks.append(HealthCheck(
                    name="theodds_api",
                    status="critical",
                    response_time_ms=round(response_time, 2),
                    message=f"API check failed: {e}",
                    details={'error': str(e)}
                ))
            
            # Test MLB Stats API
            start_time = time.time()
            try:
                starters_df = api_client.get_probable_starters()
                response_time = (time.time() - start_time) * 1000
                
                checks.append(HealthCheck(
                    name="mlb_stats_api",
                    status="healthy",
                    response_time_ms=round(response_time, 2),
                    message=f"API operational, {len(starters_df)} starters available",
                    details={'starters_count': len(starters_df)}
                ))
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                checks.append(HealthCheck(
                    name="mlb_stats_api",
                    status="critical",
                    response_time_ms=round(response_time, 2),
                    message=f"MLB API check failed: {e}",
                    details={'error': str(e)}
                ))
            
        except Exception as e:
            checks.append(HealthCheck(
                name="api_connectivity",
                status="critical",
                response_time_ms=0,
                message=f"API client initialization failed: {e}",
                details={'error': str(e)}
            ))
        
        return checks
    
    def check_model_system(self) -> HealthCheck:
        """Check model system health."""
        start_time = time.time()
        
        try:
            # Initialize model system
            model_system = EnhancedDualModelSystem(str(config.MODEL_DIR))
            
            # Try to load models
            model_system.load()
            
            # Get model info
            model_info = model_system.get_model_info()
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status
            if not model_info['core_model_loaded'] and not model_info['enhanced_model_loaded']:
                status = "critical"
                message = "No models loaded"
            elif model_info['enhanced_model_loaded']:
                status = "healthy"
                message = f"Enhanced model loaded ({model_info['enhanced_features_count']} features)"
            elif model_info['core_model_loaded']:
                status = "warning"
                message = f"Only core model loaded ({model_info['core_features_count']} features)"
            else:
                status = "unknown"
                message = "Model status unclear"
            
            return HealthCheck(
                name="model_system",
                status=status,
                response_time_ms=round(response_time, 2),
                message=message,
                details=model_info
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="model_system",
                status="critical",
                response_time_ms=round(response_time, 2),
                message=f"Model system check failed: {e}",
                details={'error': str(e)}
            )
    
    def check_data_availability(self) -> HealthCheck:
        """Check data availability and freshness."""
        start_time = time.time()
        
        try:
            data_dir = Path(config.DATA_DIR)
            cache_dir = Path(config.CACHE_DIR)
            
            # Check if directories exist
            if not data_dir.exists():
                return HealthCheck(
                    name="data_availability",
                    status="critical",
                    response_time_ms=0,
                    message="Data directory missing"
                )
            
            # Check for matchup database
            matchup_db_path = data_dir / "matchup_database.db"
            matchup_db_exists = matchup_db_path.exists()
            matchup_db_size = matchup_db_path.stat().st_size if matchup_db_exists else 0
            
            # Check for recent cache files
            cache_files = list(cache_dir.glob("*.parquet")) if cache_dir.exists() else []
            recent_cache_files = []
            
            cutoff_time = datetime.now() - timedelta(days=7)
            for cache_file in cache_files:
                file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_mtime > cutoff_time:
                    recent_cache_files.append({
                        'name': cache_file.name,
                        'modified': file_mtime.isoformat(),
                        'size_mb': round(cache_file.stat().st_size / (1024*1024), 2)
                    })
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status
            if not matchup_db_exists:
                status = "critical"
                message = "Matchup database missing"
            elif matchup_db_size < 1024*1024:  # Less than 1MB
                status = "warning"
                message = f"Matchup database too small ({matchup_db_size} bytes)"
            elif len(recent_cache_files) == 0:
                status = "warning"
                message = "No recent cache files found"
            else:
                status = "healthy"
                message = f"Data available: {len(recent_cache_files)} recent cache files"
            
            details = {
                'matchup_database': {
                    'exists': matchup_db_exists,
                    'size_bytes': matchup_db_size,
                    'size_mb': round(matchup_db_size / (1024*1024), 2)
                },
                'cache_files': {
                    'total_count': len(cache_files),
                    'recent_count': len(recent_cache_files),
                    'recent_files': recent_cache_files[:5]  # Show first 5
                }
            }
            
            return HealthCheck(
                name="data_availability",
                status=status,
                response_time_ms=round(response_time, 2),
                message=message,
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="data_availability",
                status="critical",
                response_time_ms=round(response_time, 2),
                message=f"Data availability check failed: {e}",
                details={'error': str(e)}
            )
    
    def check_log_health(self) -> HealthCheck:
        """Check log files and error rates."""
        start_time = time.time()
        
        try:
            log_dir = Path("logs")
            
            if not log_dir.exists():
                return HealthCheck(
                    name="log_health",
                    status="warning",
                    response_time_ms=0,
                    message="Log directory missing"
                )
            
            # Check recent log files
            log_files = list(log_dir.glob("*.log"))
            recent_errors = 0
            recent_warnings = 0
            
            # Check the most recent log files for error patterns
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            for log_file in log_files:
                try:
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime > cutoff_time:
                        # Quick scan for error patterns
                        with open(log_file, 'r') as f:
                            for line in f.readlines()[-1000:]:  # Check last 1000 lines
                                if 'ERROR' in line:
                                    recent_errors += 1
                                elif 'WARNING' in line:
                                    recent_warnings += 1
                except Exception:
                    continue  # Skip files we can't read
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status
            if recent_errors > 50:
                status = "critical"
                message = f"High error rate: {recent_errors} errors in last hour"
            elif recent_errors > 10:
                status = "warning"
                message = f"Elevated error rate: {recent_errors} errors, {recent_warnings} warnings"
            else:
                status = "healthy"
                message = f"Normal log activity: {recent_errors} errors, {recent_warnings} warnings"
            
            details = {
                'log_files_count': len(log_files),
                'recent_errors': recent_errors,
                'recent_warnings': recent_warnings,
                'log_directory': str(log_dir)
            }
            
            return HealthCheck(
                name="log_health",
                status=status,
                response_time_ms=round(response_time, 2),
                message=message,
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="log_health",
                status="critical",
                response_time_ms=round(response_time, 2),
                message=f"Log health check failed: {e}",
                details={'error': str(e)}
            )
    
    def run_all_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status."""
        start_time = time.time()
        
        all_checks = []
        
        # Run individual checks
        all_checks.append(self.check_system_resources())
        all_checks.extend(self.check_api_connectivity())
        all_checks.append(self.check_model_system())
        all_checks.append(self.check_data_availability())
        all_checks.append(self.check_log_health())
        
        total_time = (time.time() - start_time) * 1000
        
        # Determine overall status
        statuses = [check.status for check in all_checks]
        if 'critical' in statuses:
            overall_status = 'critical'
        elif 'warning' in statuses:
            overall_status = 'warning'
        elif 'unknown' in statuses:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        # Count by status
        status_counts = {}
        for status in ['healthy', 'warning', 'critical', 'unknown']:
            status_counts[status] = statuses.count(status)
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'total_checks': len(all_checks),
            'status_counts': status_counts,
            'total_check_time_ms': round(total_time, 2),
            'checks': [asdict(check) for check in all_checks]
        }
        
        # Store in history
        self.health_history.append(health_report)
        
        # Keep only last 100 health checks
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return health_report
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_checks = [
            check for check in self.health_history
            if datetime.fromisoformat(check['timestamp']) > cutoff_time
        ]
        
        if not recent_checks:
            return {'message': 'No recent health check data'}
        
        # Analyze trends
        status_over_time = []
        for check in recent_checks:
            status_over_time.append({
                'timestamp': check['timestamp'],
                'overall_status': check['overall_status'],
                'critical_count': check['status_counts'].get('critical', 0),
                'warning_count': check['status_counts'].get('warning', 0)
            })
        
        return {
            'time_period_hours': hours,
            'total_health_checks': len(recent_checks),
            'status_over_time': status_over_time
        }
    
    def export_health_data(self, output_dir: str = "logs"):
        """Export health check data."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"health_checks_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'latest_health_report': self.health_history[-1] if self.health_history else None,
            'health_trends': self.get_health_trends(hours=24),
            'all_health_checks': self.health_history
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Health data exported to {filename}")

def main():
    """Run health checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='System Health Checker')
    parser.add_argument('--export', action='store_true', help='Export health data')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring')
    parser.add_argument('--interval', type=int, default=60, help='Watch interval in seconds')
    
    args = parser.parse_args()
    
    checker = HealthChecker()
    
    if args.watch:
        print(f"Starting continuous health monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                health_report = checker.run_all_health_checks()
                
                print(f"\n[{health_report['timestamp']}] Overall Status: {health_report['overall_status'].upper()}")
                print(f"Checks: {health_report['status_counts']}")
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\nHealth monitoring stopped")
    else:
        # Single health check run
        health_report = checker.run_all_health_checks()
        
        print(f"System Health Report - {health_report['timestamp']}")
        print("=" * 60)
        print(f"Overall Status: {health_report['overall_status'].upper()}")
        print(f"Total Checks: {health_report['total_checks']}")
        print(f"Status Distribution: {health_report['status_counts']}")
        print(f"Check Time: {health_report['total_check_time_ms']:.1f}ms")
        
        print("\nDetailed Results:")
        for check in health_report['checks']:
            status_emoji = {'healthy': '✅', 'warning': '⚠️', 'critical': '❌', 'unknown': '❓'}
            emoji = status_emoji.get(check['status'], '❓')
            print(f"  {emoji} {check['name']}: {check['message']} ({check['response_time_ms']:.1f}ms)")
        
        if args.export:
            checker.export_health_data()

if __name__ == "__main__":
    main()