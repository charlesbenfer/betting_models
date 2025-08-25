#!/usr/bin/env python3
"""
Error Tracking and Alerting System
==================================

Comprehensive error tracking, classification, and alerting for production systems.
"""

import os
import sys
import json
import time
import smtplib
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class ErrorEvent:
    """Individual error event."""
    timestamp: str
    error_id: str
    severity: str  # critical, warning, info
    component: str
    error_type: str
    message: str
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    resolved: bool = False
    
@dataclass
class AlertRule:
    """Alert configuration rule."""
    name: str
    condition: str  # error_rate, error_type, component
    threshold: float
    time_window_minutes: int
    severity: str
    enabled: bool = True

class ErrorTracker:
    """Production error tracking and alerting system."""
    
    def __init__(self, max_errors: int = 10000):
        self.max_errors = max_errors
        self.errors = deque(maxlen=max_errors)
        self.error_counts = defaultdict(int)
        self.alert_rules = []
        self.alert_handlers = []
        
        # Setup logging
        self.setup_error_logging()
        
        # Load default alert rules
        self.setup_default_alert_rules()
        
        # Alert state tracking
        self.alert_states = {}
        self.last_alert_times = {}
        
        self.logger.info("Error tracker initialized")
    
    def setup_error_logging(self):
        """Setup error-specific logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create error logger
        self.logger = logging.getLogger('error_tracker')
        self.logger.setLevel(logging.INFO)
        
        # File handler for error tracking logs
        error_log = log_dir / "error_tracking.log"
        handler = logging.FileHandler(error_log)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
    
    def setup_default_alert_rules(self):
        """Setup default alerting rules."""
        self.alert_rules = [
            AlertRule(
                name="High Error Rate",
                condition="error_rate",
                threshold=10.0,  # 10 errors per minute
                time_window_minutes=5,
                severity="critical"
            ),
            AlertRule(
                name="API Failure Rate",
                condition="api_errors",
                threshold=5.0,   # 5 API errors per minute
                time_window_minutes=5,
                severity="warning"
            ),
            AlertRule(
                name="Model Prediction Failures",
                condition="prediction_errors",
                threshold=3.0,   # 3 prediction errors per minute
                time_window_minutes=10,
                severity="critical"
            ),
            AlertRule(
                name="Critical System Errors",
                condition="critical_errors",
                threshold=1.0,   # Any critical error
                time_window_minutes=1,
                severity="critical"
            )
        ]
    
    def record_error(self, 
                    component: str,
                    error_type: str, 
                    message: str,
                    severity: str = "warning",
                    stack_trace: Optional[str] = None,
                    context: Optional[Dict[str, Any]] = None) -> str:
        """Record a new error event."""
        
        error_id = f"{component}_{error_type}_{int(time.time())}"
        
        error_event = ErrorEvent(
            timestamp=datetime.now().isoformat(),
            error_id=error_id,
            severity=severity,
            component=component,
            error_type=error_type,
            message=message,
            stack_trace=stack_trace,
            context=context or {}
        )
        
        # Store error
        self.errors.append(error_event)
        
        # Update counters
        self.error_counts['total'] += 1
        self.error_counts[f'by_component_{component}'] += 1
        self.error_counts[f'by_type_{error_type}'] += 1
        self.error_counts[f'by_severity_{severity}'] += 1
        
        # Log error
        self.logger.error(f"Error recorded: {component}.{error_type} - {message}")
        
        # Check alert rules
        self._check_alert_rules()
        
        return error_id
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [
            e for e in self.errors 
            if datetime.fromisoformat(e.timestamp) > cutoff_time
        ]
        
        # Group by various dimensions
        by_component = defaultdict(int)
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_hour = defaultdict(int)
        
        for error in recent_errors:
            by_component[error.component] += 1
            by_type[error.error_type] += 1
            by_severity[error.severity] += 1
            
            # Group by hour
            hour_key = datetime.fromisoformat(error.timestamp).strftime('%Y-%m-%d %H:00')
            by_hour[hour_key] += 1
        
        return {
            'time_period_hours': hours,
            'total_errors': len(recent_errors),
            'by_component': dict(by_component),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'by_hour': dict(by_hour),
            'error_rate_per_hour': len(recent_errors) / hours if hours > 0 else 0
        }
    
    def get_top_errors(self, limit: int = 10, hours: int = 24) -> List[Dict[str, Any]]:
        """Get most frequent errors in time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [
            e for e in self.errors 
            if datetime.fromisoformat(e.timestamp) > cutoff_time
        ]
        
        # Count error patterns
        error_patterns = defaultdict(lambda: {'count': 0, 'latest': None, 'severity': 'info'})
        
        for error in recent_errors:
            pattern_key = f"{error.component}::{error.error_type}"
            error_patterns[pattern_key]['count'] += 1
            error_patterns[pattern_key]['latest'] = error.timestamp
            error_patterns[pattern_key]['severity'] = error.severity
            error_patterns[pattern_key]['message'] = error.message
        
        # Sort by count and return top N
        sorted_patterns = sorted(
            error_patterns.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:limit]
        
        return [
            {
                'pattern': pattern,
                'count': data['count'],
                'latest_occurrence': data['latest'],
                'severity': data['severity'],
                'sample_message': data['message']
            }
            for pattern, data in sorted_patterns
        ]
    
    def _check_alert_rules(self):
        """Check all alert rules and trigger alerts if needed."""
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            try:
                should_alert = self._evaluate_alert_rule(rule, current_time)
                
                if should_alert:
                    # Check if we should suppress this alert (rate limiting)
                    if self._should_send_alert(rule.name, current_time):
                        self._trigger_alert(rule, current_time)
                        
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    def _evaluate_alert_rule(self, rule: AlertRule, current_time: datetime) -> bool:
        """Evaluate if an alert rule condition is met."""
        window_start = current_time - timedelta(minutes=rule.time_window_minutes)
        
        recent_errors = [
            e for e in self.errors
            if datetime.fromisoformat(e.timestamp) > window_start
        ]
        
        if rule.condition == "error_rate":
            error_rate = len(recent_errors) / rule.time_window_minutes
            return error_rate >= rule.threshold
            
        elif rule.condition == "api_errors":
            api_errors = [e for e in recent_errors if 'api' in e.component.lower()]
            api_error_rate = len(api_errors) / rule.time_window_minutes
            return api_error_rate >= rule.threshold
            
        elif rule.condition == "prediction_errors":
            pred_errors = [e for e in recent_errors if 'prediction' in e.component.lower() or 'model' in e.component.lower()]
            pred_error_rate = len(pred_errors) / rule.time_window_minutes
            return pred_error_rate >= rule.threshold
            
        elif rule.condition == "critical_errors":
            critical_errors = [e for e in recent_errors if e.severity == 'critical']
            critical_error_rate = len(critical_errors) / rule.time_window_minutes
            return critical_error_rate >= rule.threshold
        
        return False
    
    def _should_send_alert(self, rule_name: str, current_time: datetime) -> bool:
        """Check if we should send alert (rate limiting)."""
        # Don't send same alert more than once per hour
        last_alert = self.last_alert_times.get(rule_name)
        if last_alert:
            time_since_last = current_time - last_alert
            if time_since_last < timedelta(hours=1):
                return False
        
        return True
    
    def _trigger_alert(self, rule: AlertRule, current_time: datetime):
        """Trigger an alert for the given rule."""
        alert_data = {
            'rule_name': rule.name,
            'severity': rule.severity,
            'timestamp': current_time.isoformat(),
            'condition': rule.condition,
            'threshold': rule.threshold,
            'time_window_minutes': rule.time_window_minutes,
            'recent_errors': self.get_error_summary(hours=1)
        }
        
        # Record alert time
        self.last_alert_times[rule.name] = current_time
        
        # Send to all alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
        
        self.logger.warning(f"Alert triggered: {rule.name} - {rule.severity}")
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def export_errors(self, output_dir: str = "logs", hours: int = 24):
        """Export error data to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [
            asdict(e) for e in self.errors
            if datetime.fromisoformat(e.timestamp) > cutoff_time
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"errors_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_period_hours': hours,
            'total_errors': len(recent_errors),
            'summary': self.get_error_summary(hours),
            'top_errors': self.get_top_errors(hours=hours),
            'errors': recent_errors
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Errors exported to {filename}")

class EmailAlertHandler:
    """Email alert handler for critical errors."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, 
                 from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        
        self.logger = logging.getLogger('email_alerts')
    
    def __call__(self, alert_data: Dict[str, Any]):
        """Send email alert."""
        try:
            subject = f"🚨 Baseball HR System Alert: {alert_data['rule_name']}"
            
            body = f"""
Alert Triggered: {alert_data['rule_name']}
Severity: {alert_data['severity'].upper()}
Time: {alert_data['timestamp']}
Condition: {alert_data['condition']}
Threshold: {alert_data['threshold']}

Recent Error Summary:
- Total Errors (1 hour): {alert_data['recent_errors']['total_errors']}
- Error Rate: {alert_data['recent_errors']['error_rate_per_hour']:.1f}/hour

Top Error Types:
{self._format_error_types(alert_data['recent_errors']['by_type'])}

Please check the system immediately.
            """
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"Alert email sent for: {alert_data['rule_name']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def _format_error_types(self, error_types: Dict[str, int]) -> str:
        """Format error types for email body."""
        if not error_types:
            return "No errors"
        
        sorted_types = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        return '\n'.join([f"  - {error_type}: {count}" for error_type, count in sorted_types[:5]])

class ConsoleAlertHandler:
    """Console alert handler for development."""
    
    def __call__(self, alert_data: Dict[str, Any]):
        """Print alert to console."""
        severity_emoji = {
            'critical': '🔴',
            'warning': '🟡', 
            'info': '🔵'
        }
        
        emoji = severity_emoji.get(alert_data['severity'], '⚪')
        
        print(f"\n{emoji} ALERT: {alert_data['rule_name']}")
        print(f"Severity: {alert_data['severity'].upper()}")
        print(f"Time: {alert_data['timestamp']}")
        print(f"Condition: {alert_data['condition']} >= {alert_data['threshold']}")
        print(f"Recent errors: {alert_data['recent_errors']['total_errors']}")
        print("="*50)

def main():
    """Example usage of error tracking system."""
    # Create error tracker
    tracker = ErrorTracker()
    
    # Add console alert handler
    tracker.add_alert_handler(ConsoleAlertHandler())
    
    # Simulate some errors
    print("Simulating errors...")
    
    tracker.record_error("api_client", "connection_timeout", "Timeout connecting to The Odds API", "warning")
    tracker.record_error("model_system", "prediction_failed", "Model prediction failed for batch", "critical")
    tracker.record_error("api_client", "rate_limit", "API rate limit exceeded", "warning")
    
    # Get error summary
    summary = tracker.get_error_summary(hours=1)
    print(f"\nError Summary: {summary}")
    
    # Get top errors
    top_errors = tracker.get_top_errors(limit=5, hours=1)
    print(f"\nTop Errors: {top_errors}")
    
    # Export errors
    tracker.export_errors(hours=1)
    print("Errors exported to logs/")

if __name__ == "__main__":
    main()