# Baseball HR Prediction System - API Documentation

## Overview

This document provides comprehensive API documentation for the Baseball Home Run Prediction System. The system exposes both programmatic APIs and command-line interfaces for all major functionality.

## 🔗 Core APIs

### Machine Learning API

#### `modeling.py` - Enhanced Dual Model System

```python
from modeling import EnhancedDualModelSystem

# Initialize model system
model_system = EnhancedDualModelSystem(model_dir="saved_models")

# Load trained models
model_system.load_model("path/to/model.joblib")

# Train new model
training_result = model_system.train_model(
    training_data,
    target_column="home_runs",
    validation_split=0.2,
    optimize_hyperparameters=True
)

# Generate predictions
predictions = model_system.predict(features_df)

# Get model information
model_info = model_system.get_model_info()
```

**Methods:**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `__init__(model_dir, splitting_strategy)` | Initialize model system | `model_dir`: str, `splitting_strategy`: str | EnhancedDualModelSystem |
| `train_model(data, target_col, **kwargs)` | Train ML models | `data`: DataFrame, `target_col`: str | Dict with training results |
| `predict(features)` | Generate predictions | `features`: DataFrame | numpy.ndarray |
| `load_model(path)` | Load saved model | `path`: str | bool (success) |
| `save_model(path)` | Save current model | `path`: str | bool (success) |
| `get_model_info()` | Get model metadata | None | Dict with model info |
| `validate_model(test_data)` | Validate model performance | `test_data`: DataFrame | Dict with metrics |

**Example Response:**
```python
{
    "model_loaded": True,
    "model_type": "XGBoost",
    "features_count": 255,
    "training_date": "2025-08-22",
    "validation_accuracy": 0.821,
    "feature_importance": {
        "weather_temperature": 0.156,
        "batter_career_hr_rate": 0.142,
        "pitcher_era": 0.128
    }
}
```

### Live Prediction API

#### `live_prediction_system.py` - Real-time Predictions

```python
from live_prediction_system import LivePredictionSystem

# Initialize prediction system
live_system = LivePredictionSystem(api_key="your_api_key")

# Initialize models and data
live_system.initialize()

# Get today's predictions
predictions = live_system.get_todays_predictions()

# Find betting opportunities
opportunities = live_system.find_betting_opportunities(min_ev=0.05)

# Get system status
status = live_system.get_system_status()
```

**Methods:**

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `initialize()` | Initialize system components | None | bool (success) |
| `get_todays_predictions(date)` | Get predictions for date | `date`: str (optional) | DataFrame |
| `find_betting_opportunities(**kwargs)` | Find profitable bets | `min_ev`: float, `min_prob`: float | List[BettingOpportunity] |
| `get_system_status()` | Get system health status | None | Dict |
| `predict_game(game_data)` | Predict single game | `game_data`: Dict | Dict |

**Prediction Response:**
```python
{
    "game_id": "LAD_NYY_20250822",
    "home_team": "LAD",
    "away_team": "NYY", 
    "game_time": "2025-08-22T19:00:00Z",
    "predictions": {
        "home_runs_over_under": {
            "prediction": "over",
            "probability": 0.67,
            "confidence": 0.89,
            "line": 2.5
        },
        "total_home_runs": {
            "expected_value": 3.2,
            "range": [2.1, 4.8],
            "confidence_interval": [2.5, 3.9]
        }
    },
    "betting_recommendation": {
        "action": "bet",
        "bet_type": "over_2.5_hrs",
        "expected_value": 0.087,
        "recommended_stake": 0.02
    }
}
```

### Betting Analysis API

#### `betting_utils.py` - Betting Calculations

```python
from betting_utils import BettingAnalyzer, BettingOpportunity

# Initialize betting analyzer
analyzer = BettingAnalyzer(min_ev=0.05, min_probability=0.15)

# Analyze betting opportunity
opportunity = analyzer.analyze_bet(
    prediction_prob=0.65,
    bookmaker_odds=2.20,
    bet_type="home_run_over"
)

# Calculate expected value
ev = analyzer.calculate_expected_value(true_prob=0.60, odds=2.50)

# Get bankroll recommendation
stake = analyzer.calculate_optimal_stake(
    bankroll=1000,
    expected_value=0.08,
    confidence=0.85
)
```

**BettingOpportunity Object:**
```python
{
    "game_id": "game_123",
    "bet_type": "home_run_over_2.5",
    "bookmaker": "betmgm",
    "odds": 2.20,
    "implied_probability": 0.45,
    "predicted_probability": 0.67,
    "expected_value": 0.087,
    "confidence": 0.89,
    "recommended_stake_pct": 0.02,
    "risk_level": "moderate",
    "created_at": "2025-08-22T14:30:00Z"
}
```

### Data Integration API

#### `api_client.py` - External Data Sources

```python
from api_client import SafeAPIClient

# Initialize API client
client = SafeAPIClient(api_key="your_key")

# Fetch current odds
odds_data = client.fetch_current_odds(
    sport="baseball_mlb",
    markets=["totals", "h2h"]
)

# Get weather data
weather = client.get_weather_data(
    location="Los Angeles, CA",
    date="2025-08-22"
)

# Test API connectivity
health = client.test_api_health()
```

## 🖥️ Command Line Interfaces

### Production Management

#### Deployment Script
```bash
./scripts/deploy.sh [OPTIONS]

Options:
  --environment ENV     Target environment (development/staging/production)
  --backup             Create backup before deployment
  --validate           Run validation checks
  --rollback VERSION   Rollback to previous version
  --help               Show help message
```

#### Production Startup
```bash
python scripts/production_startup.py [OPTIONS]

Options:
  --environment ENV     Target environment
  --skip-validation    Skip environment validation
  --config-file PATH   Custom configuration file
  --log-level LEVEL    Logging level (DEBUG/INFO/WARNING/ERROR)
```

### Monitoring & Health

#### Health Checks
```bash
python monitoring/health_checks.py [OPTIONS]

Options:
  --comprehensive      Run all health checks
  --component NAME     Check specific component
  --export             Export results to file
  --threshold VALUE    Alert threshold
```

**Health Check Response:**
```json
{
  "timestamp": "2025-08-22T14:30:00Z",
  "overall_status": "healthy",
  "checks": [
    {
      "name": "database_connectivity",
      "status": "healthy",
      "response_time_ms": 15,
      "message": "Database connection successful"
    },
    {
      "name": "api_connectivity", 
      "status": "healthy",
      "response_time_ms": 234,
      "message": "All external APIs accessible"
    },
    {
      "name": "model_availability",
      "status": "healthy",
      "message": "2 models loaded successfully"
    }
  ]
}
```

#### System Monitor
```bash
python monitoring/system_monitor.py [OPTIONS]

Options:
  --interval SECONDS   Monitoring interval (default: 60)
  --dashboard          Show live dashboard
  --export PATH        Export metrics to file
  --alert-email EMAIL  Email for alerts
```

#### Performance Testing
```bash
python scripts/performance_tests.py [OPTIONS]

Options:
  --test TYPE          Run specific test (latency/throughput/memory)
  --quick              Run quick performance check only
  --export             Export detailed results
  --threshold FILE     Custom performance thresholds
```

### Security Operations

#### Security Validation
```bash
python security/security_validator.py [OPTIONS]

Options:
  --audit TYPE         Audit type (all/permissions/sensitive/deps)
  --directory PATH     Directory to audit
  --export             Export audit results
  --fix                Attempt automatic fixes
```

#### Security Hardening
```bash
python scripts/security_hardening.py [OPTIONS]

Options:
  --directory PATH     Directory to harden
  --dry-run           Show what would be done
  --export            Export hardening report
  --force             Force apply all fixes
```

### Backup & Recovery

#### Backup Manager
```bash
python backup/backup_manager.py [OPTIONS]

Commands:
  --create-backup      Create full system backup
  --restore PATH       Restore from backup file
  --cleanup            Clean up old backups
  --status             Show backup system status
  --backup-root DIR    Backup directory location
```

#### Disaster Recovery
```bash
python backup/disaster_recovery_plan.py [OPTIONS]

Commands:
  --assess SCENARIO        Assess disaster scenario
  --execute SCENARIO       Execute automated recovery
  --runbook SCENARIO       Generate recovery runbook
  --test                   Test disaster recovery procedures
  --list-scenarios         List available scenarios
```

**Available Disaster Scenarios:**
- `database_corruption` (RTO: 30 min, Automated: Yes)
- `configuration_loss` (RTO: 15 min, Automated: Yes)  
- `model_loss` (RTO: 45 min, Automated: Yes)
- `complete_system_loss` (RTO: 120 min, Automated: No)
- `data_center_outage` (RTO: 240 min, Automated: No)

### Configuration Management

#### Environment Configuration
```bash
python config/environment_config.py [OPTIONS]

Options:
  --environment ENV    Target environment
  --validate          Validate configuration
  --export PATH       Export configuration
  --import PATH       Import configuration
  --list-settings     List all settings
```

#### Secrets Manager
```bash
python config/secrets_manager.py [OPTIONS]

Commands:
  --set NAME VALUE DESC    Set a secret
  --get NAME               Get secret value
  --delete NAME            Delete a secret
  --list                   List all secrets
  --validate              Validate secrets
  --rotate-key            Rotate encryption key
  --export-template ENV    Export secrets template
```

## 🔧 Configuration API

### Environment Configuration

**Configuration Object Structure:**
```python
@dataclass
class EnvironmentConfig:
    environment: str                    # development/staging/production
    debug: bool                        # Debug mode enabled
    database: DatabaseConfig           # Database settings
    api: APIConfig                     # API configuration
    model: ModelConfig                 # Model settings
    monitoring: MonitoringConfig       # Monitoring settings
    logging: LoggingConfig             # Logging configuration
    security: SecurityConfig           # Security settings
```

**Database Configuration:**
```python
@dataclass
class DatabaseConfig:
    path: str                          # Database file path
    backup_path: Optional[str]         # Backup location
    max_size_mb: int                   # Maximum size limit
    connection_timeout: int            # Connection timeout
    enable_wal_mode: bool              # Write-Ahead Logging
```

### API Configuration Validation

```python
from config.config_validator import ConfigurationValidator

validator = ConfigurationValidator()

# Validate environment configuration
is_valid, issues = validator.validate_environment_config(config)

# Validate deployment readiness  
readiness = validator.get_deployment_readiness("production")
```

## 📊 Monitoring API

### System Metrics

```python
from monitoring.system_monitor import SystemMonitor

monitor = SystemMonitor()

# Get current system metrics
metrics = monitor.get_current_metrics()

# Get system health assessment
health = monitor.get_system_health()

# Record custom metrics
monitor.record_prediction_metrics(PredictionMetrics(...))
```

**Metrics Response:**
```python
{
    "timestamp": "2025-08-22T14:30:00Z",
    "system": {
        "cpu_percent": 12.5,
        "memory_percent": 34.2,
        "disk_usage_percent": 67.8,
        "load_average": [0.8, 0.9, 1.1]
    },
    "application": {
        "predictions_generated": 1247,
        "api_calls_made": 89,
        "cache_hit_rate": 0.87,
        "average_response_time_ms": 145
    },
    "database": {
        "connection_pool_size": 5,
        "active_connections": 2,
        "query_count": 456,
        "average_query_time_ms": 23
    }
}
```

### Error Tracking

```python
from monitoring.error_tracker import ErrorTracker

tracker = ErrorTracker()

# Record an error
error_id = tracker.record_error(
    component="api_client",
    error_type="timeout",
    message="Request timeout after 30 seconds",
    severity="warning"
)

# Get error summary
summary = tracker.get_error_summary(hours=24)

# Get top errors
top_errors = tracker.get_top_errors(limit=10)
```

## 🔐 Security API

### Input Validation

```python
from security.security_validator import InputValidator

# Validate team name
is_valid = InputValidator.validate_team_name("LAD")

# Validate date format
is_valid = InputValidator.validate_date("2025-08-22")

# Sanitize user input
clean_input = InputValidator.sanitize_string(user_input)

# Validate numeric input with range
is_valid = InputValidator.validate_numeric_input(value, min_val=0, max_val=100)
```

### Rate Limiting

```python
from security.security_validator import RateLimiter

# Initialize rate limiter (60 requests per minute)
limiter = RateLimiter(max_requests=60, time_window=60)

# Check if request is allowed
if limiter.is_allowed(user_id):
    # Process request
    pass
else:
    # Reject request - rate limit exceeded
    pass
```

## 🚀 Integration Examples

### Full System Integration

```python
# Complete workflow example
from modeling import EnhancedDualModelSystem
from live_prediction_system import LivePredictionSystem
from monitoring.health_checks import HealthChecker

# Initialize system
model_system = EnhancedDualModelSystem()
live_system = LivePredictionSystem()
health_checker = HealthChecker()

# Validate system health
health_report = health_checker.run_all_health_checks()
if health_report['overall_status'] != 'healthy':
    raise SystemError("System not ready for predictions")

# Initialize prediction system
live_system.initialize()

# Generate predictions
predictions = live_system.get_todays_predictions()

# Find betting opportunities
opportunities = live_system.find_betting_opportunities(min_ev=0.05)

# Process opportunities
for opp in opportunities:
    if opp.expected_value > 0.08:
        print(f"Strong opportunity: {opp.game_id} - EV: {opp.expected_value:.3f}")
```

### Automated Monitoring Setup

```python
# Setup automated monitoring
from monitoring.monitoring_integration import IntegratedMonitoringSystem

monitor = IntegratedMonitoringSystem()

# Start monitoring with 60-second intervals
monitor.start_monitoring(interval_seconds=60)

# Record prediction batch
monitor.record_prediction_batch(
    predictions_count=25,
    prediction_time_ms=1240,
    betting_opportunities=7
)

# Get comprehensive status
status = monitor.get_comprehensive_status()
print(f"System Status: {status['overall_status']}")
```

### Backup Automation

```python
# Automated backup workflow
from backup.backup_manager import BackupManager
from backup.disaster_recovery_plan import DisasterRecoveryPlan

backup_manager = BackupManager()
recovery_plan = DisasterRecoveryPlan()

# Create scheduled backup
backup_result = backup_manager.create_full_backup()

if backup_result['success']:
    print(f"Backup completed: {backup_result['total_size_mb']} MB")
    
    # Test disaster recovery
    test_results = recovery_plan.test_disaster_recovery()
    
    if test_results['overall_success']:
        print("✅ System backup and recovery validated")
    else:
        print("⚠️ Recovery testing found issues")
```

## 📈 Performance Considerations

### API Rate Limits

- **External APIs**: Respect provider rate limits (typically 500-1000 requests/hour)
- **Internal APIs**: No built-in limits but monitoring recommended
- **Database**: Connection pooling with max 10 concurrent connections

### Caching Strategy

- **Prediction Results**: 1-hour cache for game predictions
- **Odds Data**: 5-minute cache for live odds
- **Weather Data**: 30-minute cache for weather conditions
- **Model Metadata**: 24-hour cache for model information

### Memory Management

- **Model Loading**: Models loaded on-demand and cached
- **Data Processing**: Streaming processing for large datasets
- **Cleanup**: Automatic garbage collection for unused objects

## 🐛 Error Handling

### Standard Error Responses

```python
{
    "error": True,
    "error_type": "ValidationError",
    "message": "Invalid input parameters",
    "details": {
        "field": "team_name",
        "value": "INVALID",
        "expected": "Valid MLB team abbreviation"
    },
    "timestamp": "2025-08-22T14:30:00Z",
    "request_id": "req_123456789"
}
```

### Common Error Types

| Error Type | Description | HTTP Code | Resolution |
|------------|-------------|-----------|------------|
| `ValidationError` | Invalid input parameters | 400 | Check input format |
| `AuthenticationError` | Invalid API key | 401 | Verify API credentials |
| `RateLimitError` | Rate limit exceeded | 429 | Wait and retry |
| `ModelNotFoundError` | ML model not loaded | 500 | Load or retrain model |
| `DataSourceError` | External API failure | 503 | Check API status |

## 📚 Additional Resources

- **System Architecture**: See `docs/architecture.md`
- **Deployment Guide**: See `docs/deployment.md` 
- **Troubleshooting**: See `docs/troubleshooting.md`
- **Performance Tuning**: See `docs/performance.md`
- **Security Guide**: See `docs/security.md`

---

**API Version**: 1.0.0  
**Last Updated**: August 2025  
**Compatibility**: Python 3.8+