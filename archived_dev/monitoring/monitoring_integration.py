#!/usr/bin/env python3
"""
Monitoring Integration
=====================

Integrated monitoring system that combines all monitoring components.
"""

import os
import sys
import time
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.system_monitor import SystemMonitor, PredictionMetrics
from monitoring.error_tracker import ErrorTracker, ConsoleAlertHandler
from monitoring.performance_monitor import PerformanceMonitor
from monitoring.health_checks import HealthChecker

class IntegratedMonitoringSystem:
    """Integrated production monitoring system."""
    
    def __init__(self):
        # Initialize all monitoring components
        self.system_monitor = SystemMonitor(retention_hours=48)
        self.error_tracker = ErrorTracker(max_errors=20000)
        self.performance_monitor = PerformanceMonitor(max_metrics=100000)
        self.health_checker = HealthChecker()
        
        # Setup alert handlers
        self.error_tracker.add_alert_handler(ConsoleAlertHandler())
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        
        print("🔧 Integrated monitoring system initialized")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start comprehensive monitoring."""
        if self.monitoring_active:
            print("⚠️  Monitoring already active")
            return
        
        print(f"🚀 Starting integrated monitoring (interval: {interval_seconds}s)")
        
        # Start individual monitors
        self.system_monitor.start_monitoring(interval_seconds)
        
        # Start integrated monitoring loop
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._integrated_monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        
        print("✅ Integrated monitoring started successfully")
    
    def stop_monitoring(self):
        """Stop all monitoring."""
        print("🛑 Stopping integrated monitoring...")
        
        self.monitoring_active = False
        self.system_monitor.stop_monitoring()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        print("✅ Monitoring stopped")
    
    def _integrated_monitoring_loop(self, interval_seconds: int):
        """Main integrated monitoring loop."""
        health_check_counter = 0
        
        while self.monitoring_active:
            try:
                # Run health checks every 5 intervals (5 minutes if interval is 60s)
                health_check_counter += 1
                if health_check_counter >= 5:
                    self._run_health_checks()
                    health_check_counter = 0
                
                # Check for performance issues
                self._check_performance_issues()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.error_tracker.record_error(
                    "monitoring_system", 
                    "monitoring_loop_error", 
                    str(e), 
                    "critical"
                )
                time.sleep(interval_seconds)
    
    def _run_health_checks(self):
        """Run health checks and record issues."""
        try:
            health_report = self.health_checker.run_all_health_checks()
            
            # Record errors for non-healthy checks
            for check in health_report['checks']:
                if check['status'] in ['warning', 'critical']:
                    self.error_tracker.record_error(
                        f"health_check_{check['name']}",
                        "health_check_failure",
                        check['message'],
                        check['status']
                    )
            
        except Exception as e:
            self.error_tracker.record_error(
                "monitoring_system",
                "health_check_error", 
                str(e),
                "critical"
            )
    
    def _check_performance_issues(self):
        """Check for performance issues and record them."""
        try:
            issues = self.performance_monitor.check_performance_thresholds()
            
            for issue in issues:
                self.error_tracker.record_error(
                    "performance_monitor",
                    "performance_threshold_exceeded",
                    f"{issue['operation']}: {issue['average_response_time_ms']}ms > {issue['threshold_ms']}ms",
                    issue['severity']
                )
                
        except Exception as e:
            self.error_tracker.record_error(
                "monitoring_system",
                "performance_check_error",
                str(e),
                "warning"
            )
    
    def record_prediction_batch(self, predictions_count: int, 
                               prediction_time_ms: float,
                               betting_opportunities: int,
                               model_type: str = "enhanced"):
        """Record prediction batch metrics."""
        # Record in system monitor
        prediction_metrics = PredictionMetrics(
            timestamp=datetime.now().isoformat(),
            predictions_generated=predictions_count,
            average_prediction_time_ms=prediction_time_ms,
            betting_opportunities_found=betting_opportunities,
            model_type=model_type,
            feature_count=0,  # Will be filled by actual system
            api_calls_made=1,  # Estimate
            cache_hit_rate=0.8  # Estimate
        )
        
        self.system_monitor.record_prediction_metrics(prediction_metrics)
        
        # Record throughput in performance monitor
        self.performance_monitor.record_throughput(
            "predictions_generated", 
            predictions_count, 
            60
        )
        
        print(f"📊 Recorded batch: {predictions_count} predictions, {betting_opportunities} opportunities")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get individual statuses
            system_health = self.system_monitor.get_system_health()
            error_summary = self.error_tracker.get_error_summary(hours=1)
            performance_summary = self.performance_monitor.get_performance_summary(hours=1)
            health_report = self.health_checker.run_all_health_checks()
            
            # Determine overall status
            health_status = health_report['overall_status']
            if system_health['overall_status'] == 'critical':
                overall_status = 'critical'
            elif error_summary['total_errors'] > 20:
                overall_status = 'degraded'
            elif health_status in ['warning', 'degraded']:
                overall_status = 'warning'
            elif health_status == 'healthy':
                overall_status = 'healthy'
            else:
                overall_status = 'unknown'
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': overall_status,
                'components': {
                    'system_health': system_health,
                    'error_summary': error_summary,
                    'performance_summary': performance_summary,
                    'health_checks': health_report
                },
                'monitoring_active': self.monitoring_active
            }
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'critical',
                'error': f"Failed to get status: {e}",
                'monitoring_active': self.monitoring_active
            }
    
    def export_all_data(self, output_dir: str = "logs"):
        """Export all monitoring data."""
        print(f"📁 Exporting all monitoring data to {output_dir}...")
        
        try:
            # Export from all monitors
            self.system_monitor.export_metrics(output_dir)
            self.error_tracker.export_errors(output_dir, hours=24)
            self.performance_monitor.export_performance_data(output_dir, hours=24)
            self.health_checker.export_health_data(output_dir)
            
            # Create summary export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = Path(output_dir) / f"monitoring_summary_{timestamp}.json"
            
            summary_data = {
                'export_timestamp': datetime.now().isoformat(),
                'comprehensive_status': self.get_comprehensive_status(),
                'monitoring_components': {
                    'system_monitor': 'active' if self.system_monitor.monitoring_active else 'inactive',
                    'error_tracker': f"{len(self.error_tracker.errors)} errors tracked",
                    'performance_monitor': f"{len(self.performance_monitor.metrics)} metrics",
                    'health_checker': f"{len(self.health_checker.health_history)} health checks"
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"✅ All monitoring data exported successfully")
            print(f"📄 Summary: {summary_file}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def print_dashboard(self):
        """Print monitoring dashboard to console."""
        status = self.get_comprehensive_status()
        
        # Clear screen and print dashboard
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("🏟️  BASEBALL HR PREDICTION SYSTEM - MONITORING DASHBOARD")
        print("=" * 80)
        print(f"Timestamp: {status['timestamp']}")
        print(f"Overall Status: {self._format_status(status['overall_status'])}")
        print(f"Monitoring: {'🟢 Active' if status['monitoring_active'] else '🔴 Inactive'}")
        
        if 'components' in status:
            components = status['components']
            
            # System Health
            print("\n📊 SYSTEM HEALTH:")
            sys_health = components.get('system_health', {})
            for component, details in sys_health.get('components', {}).items():
                print(f"  {self._format_status(details['status'])} {component}: {details['message']}")
            
            # Error Summary
            print("\n🚨 ERROR SUMMARY (Last Hour):")
            error_summary = components.get('error_summary', {})
            print(f"  Total Errors: {error_summary.get('total_errors', 0)}")
            for component, count in error_summary.get('by_component', {}).items():
                print(f"  - {component}: {count}")
            
            # Performance Summary
            print("\n⚡ PERFORMANCE (Last Hour):")
            perf_summary = components.get('performance_summary', {})
            print(f"  Total Operations: {perf_summary.get('total_operations', 0)}")
            for op, time_ms in perf_summary.get('average_response_times_ms', {}).items():
                print(f"  - {op}: {time_ms:.1f}ms avg")
            
            # Health Checks
            print("\n🔍 HEALTH CHECKS:")
            health_report = components.get('health_checks', {})
            status_counts = health_report.get('status_counts', {})
            print(f"  ✅ Healthy: {status_counts.get('healthy', 0)}")
            print(f"  ⚠️  Warning: {status_counts.get('warning', 0)}")
            print(f"  ❌ Critical: {status_counts.get('critical', 0)}")
        
        print("=" * 80)
        print("Press Ctrl+C to stop monitoring")
    
    def _format_status(self, status: str) -> str:
        """Format status with emoji."""
        status_emojis = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '❌',
            'degraded': '🟡',
            'unknown': '❓'
        }
        emoji = status_emojis.get(status, '❓')
        return f"{emoji} {status.upper()}"

def main():
    """Main monitoring dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Monitoring System')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval')
    parser.add_argument('--dashboard', action='store_true', help='Show live dashboard')
    parser.add_argument('--export', action='store_true', help='Export all data and exit')
    parser.add_argument('--status', action='store_true', help='Show status and exit')
    
    args = parser.parse_args()
    
    # Create monitoring system
    monitor = IntegratedMonitoringSystem()
    
    try:
        if args.export:
            monitor.export_all_data()
            return
        
        if args.status:
            status = monitor.get_comprehensive_status()
            print(json.dumps(status, indent=2))
            return
        
        # Start monitoring
        monitor.start_monitoring(args.interval)
        
        if args.dashboard:
            # Live dashboard mode
            while True:
                monitor.print_dashboard()
                time.sleep(args.interval)
        else:
            # Status monitoring mode
            print("🔄 Monitoring started. Press Ctrl+C to stop and export data.")
            while True:
                time.sleep(60)
                status = monitor.get_comprehensive_status()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status['overall_status']}")
    
    except KeyboardInterrupt:
        print("\n🛑 Monitoring interrupted")
    finally:
        monitor.stop_monitoring()
        if not args.status and not args.export:
            monitor.export_all_data()

if __name__ == "__main__":
    main()