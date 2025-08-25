# Troubleshooting Guide

## 🔧 Overview

This comprehensive troubleshooting guide helps you diagnose and resolve common issues with the Baseball Home Run Prediction System. Issues are organized by category with step-by-step resolution procedures.

## 📋 Quick Diagnostic Commands

Before diving into specific issues, run these diagnostic commands:

```bash
# Quick system health check
python monitoring/health_checks.py --quick

# Performance validation  
python scripts/quick_performance_check.py

# Configuration validation
python config/config_validator.py --validate --environment production

# Recent error summary
python monitoring/error_tracker.py --recent --hours 24
```

## 🚨 Emergency Troubleshooting

### System Completely Down

#### Symptoms
- No predictions generated
- System won't start
- All commands failing

#### Immediate Actions
```bash
# 1. Check if process is running
ps aux | grep python | grep -E "(live_prediction|production_startup)"

# 2. Check system resources
free -h
df -h
top -b -n 1 | head -20

# 3. Check for obvious errors
tail -n 50 logs/application.log
tail -n 50 logs/error_tracking.log

# 4. Try basic startup
python scripts/production_startup.py --validate
```

#### Recovery Steps
1. **Stop all processes**: `pkill -f "baseball"` 
2. **Clear locks**: `rm -f data/*.lock`
3. **Restore from backup**: `python backup/backup_manager.py --restore latest`
4. **Restart system**: `python scripts/production_startup.py`

### Database Corruption

#### Symptoms
- Database errors in logs
- Unable to save/retrieve data
- SQLite integrity errors

#### Emergency Recovery
```bash
# 1. Stop all database connections
pkill -f "python.*live_prediction"

# 2. Check database integrity
sqlite3 data/production.db "PRAGMA integrity_check;"

# 3. If corrupted, restore from backup
python backup/backup_manager.py --restore latest_database_backup

# 4. Restart system
python scripts/production_startup.py
```

## 🔍 Installation & Setup Issues

### Python Environment Issues

#### Issue: ImportError or ModuleNotFoundError

**Error Examples**:
```
ImportError: No module named 'pandas'
ModuleNotFoundError: No module named 'sklearn'
```

**Diagnosis**:
```bash
# Check Python version
python --version

# Check virtual environment
which python
echo $VIRTUAL_ENV

# List installed packages
pip list | grep -E "(pandas|sklearn|xgboost)"
```

**Solutions**:
```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Force reinstall if needed
pip install -r requirements.txt --force-reinstall --no-cache-dir

# Check for conflicts
pip check
```

#### Issue: Permission Denied Errors

**Error Examples**:
```
PermissionError: [Errno 13] Permission denied
OSError: [Errno 13] Permission denied: 'data/production.db'
```

**Diagnosis**:
```bash
# Check file permissions
ls -la data/
ls -la logs/
ls -la config/

# Check ownership
whoami
ls -la . | head -5
```

**Solutions**:
```bash
# Fix ownership (Linux/Mac)
sudo chown -R $USER:$USER .

# Fix permissions
chmod 755 .
chmod 644 *.py
chmod 600 .env*
chmod 700 data/ logs/ backups/
chmod 600 data/*.db

# For systemd service
sudo chown -R baseball-hr:baseball-hr /opt/baseball-hr
```

### Configuration Issues

#### Issue: Invalid Configuration

**Error Examples**:
```
ConfigurationError: Invalid database path
ValueError: Invalid API key format
KeyError: 'THEODDS_API_KEY'
```

**Diagnosis**:
```bash
# Validate configuration
python config/config_validator.py --validate --environment development

# Check environment variables
env | grep BASEBALL
cat .env

# Check configuration loading
python -c "from config import config; print(vars(config))"
```

**Solutions**:
```bash
# Copy template if .env missing
cp .env.template .env

# Validate and fix configuration
python config/config_validator.py --validate --fix

# Reset to defaults
python config/environment_config.py --reset --environment development

# Test configuration loading
python config/environment_config.py --test-load
```

## 🤖 Model & Prediction Issues

### Model Loading Problems

#### Issue: Model Not Found

**Error Examples**:
```
ModelNotFoundError: No model file found at models/
FileNotFoundError: Model file 'enhanced_model.joblib' not found
```

**Diagnosis**:
```bash
# Check model files
ls -la models/
ls -la saved_models*/

# Check model metadata
python modeling.py --list-models

# Validate model files
python modeling.py --validate-models
```

**Solutions**:
```bash
# Download pre-trained models (if available)
python scripts/download_models.py

# Train new models
python modeling.py --train --data data/processed_games.csv

# Use basic model fallback
python modeling.py --train --quick --features basic

# Check model compatibility
python modeling.py --check-compatibility
```

#### Issue: Model Performance Degradation

**Error Examples**:
```
WARNING: Model accuracy dropped to 65%
ModelPerformanceWarning: Prediction confidence below threshold
```

**Diagnosis**:
```bash
# Check recent performance
python modeling.py --evaluate-recent --days 30

# Analyze feature drift
python dataset_builder.py --analyze-drift --days 90

# Check data quality
python data_utils.py --validate-recent --days 7
```

**Solutions**:
```bash
# Retrain with recent data
python modeling.py --retrain --days 365 --validate

# Update feature engineering
python dataset_builder.py --rebuild-features --optimize

# Rollback to previous model
python modeling.py --rollback --version previous

# Force retrain with expanded data
python modeling.py --retrain --days 1095 --cross-validate
```

### Prediction Generation Issues

#### Issue: No Predictions Generated

**Error Examples**:
```
No games found for today
No predictions generated - check game schedule
PredictionError: Unable to generate predictions
```

**Diagnosis**:
```bash
# Check game schedule
python live_prediction_system.py --debug --date today

# Check API connectivity
python api_client.py --test-connection

# Verify data availability
python data_utils.py --check-recent-data --days 1
```

**Solutions**:
```bash
# Force refresh game data
python live_prediction_system.py --refresh-data --date today

# Check different date
python live_prediction_system.py --date tomorrow

# Use cached data if API issues
python live_prediction_system.py --use-cache --fallback

# Manual game entry for testing
python live_prediction_system.py --test-mode --games LAD:NYY,BOS:HOU
```

#### Issue: Low Confidence Predictions

**Symptoms**: All predictions show confidence < 70%

**Diagnosis**:
```bash
# Check model performance metrics
python modeling.py --evaluate --verbose

# Analyze recent prediction accuracy
python live_prediction_system.py --analyze-recent --days 7

# Check feature quality
python dataset_builder.py --validate-features --recent
```

**Solutions**:
```bash
# Retrain with more data
python modeling.py --retrain --days 730

# Adjust confidence thresholds
python live_prediction_system.py --min-confidence 0.60

# Use ensemble voting
python modeling.py --enable-ensemble --voting-strategy soft

# Check and fix data quality issues
python data_utils.py --clean-outliers --validate
```

## 🌐 API & Data Issues

### API Connection Problems

#### Issue: API Authentication Failures

**Error Examples**:
```
AuthenticationError: Invalid API key
HTTP 401: Unauthorized access
API_KEY_INVALID: Please check your API key
```

**Diagnosis**:
```bash
# Test API key validity
python api_client.py --test-key

# Check API key format
python config/secrets_manager.py --get theodds_api_key --validate

# Test manual API call
curl -H "X-RapidAPI-Key: YOUR_KEY" "https://api.the-odds-api.com/v4/sports/"
```

**Solutions**:
```bash
# Update API key
python config/secrets_manager.py --set theodds_api_key "NEW_KEY" "Updated API key"

# Reset API client cache
python api_client.py --clear-cache

# Check API key with provider
# Visit: https://the-odds-api.com/account/

# Use fallback data if needed
python live_prediction_system.py --offline-mode
```

#### Issue: API Rate Limiting

**Error Examples**:
```
RateLimitError: Too many requests
HTTP 429: Rate limit exceeded
API quota exceeded - wait 3600 seconds
```

**Diagnosis**:
```bash
# Check API usage
python api_client.py --check-usage

# Review recent API calls
grep "API_CALL" logs/application.log | tail -20

# Check rate limiting configuration
python config/environment_config.py --show api
```

**Solutions**:
```bash
# Wait for rate limit reset
python api_client.py --wait-for-reset

# Increase delay between requests
python api_client.py --set-delay 5  # 5 seconds between calls

# Use cached data
python live_prediction_system.py --prefer-cache --max-age 3600

# Upgrade API plan if needed
echo "Consider upgrading API plan for higher limits"
```

### Data Quality Issues

#### Issue: Stale or Missing Data

**Error Examples**:
```
DataFreshnessError: Data is 6 hours old
NoDataError: No recent games found
Warning: Using data from 2 days ago
```

**Diagnosis**:
```bash
# Check data freshness
python data_utils.py --check-freshness --verbose

# Verify recent API calls
python api_client.py --check-last-update

# Check database contents
sqlite3 data/production.db "SELECT COUNT(*), MAX(created_at) FROM games WHERE game_date >= date('now');"
```

**Solutions**:
```bash
# Force data refresh
python live_prediction_system.py --force-refresh --no-cache

# Rebuild recent data
python data_utils.py --rebuild-recent --days 3

# Clear stale cache
python data_utils.py --clear-stale-cache --max-age 1800

# Fallback to historical patterns
python live_prediction_system.py --use-historical-fallback
```

## 🖥️ System Performance Issues

### High Memory Usage

#### Issue: Memory Consumption Too High

**Symptoms**: System using >2GB RAM, out of memory errors

**Diagnosis**:
```bash
# Check memory usage
python scripts/quick_performance_check.py

# Monitor memory by process
ps aux --sort=-%mem | head -10

# Check for memory leaks
python -c "
import psutil
import os
p = psutil.Process(os.getpid())
print(f'Memory: {p.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Solutions**:
```bash
# Clear model cache
python modeling.py --clear-cache

# Optimize feature processing
python dataset_builder.py --optimize-memory

# Reduce batch sizes
python live_prediction_system.py --batch-size 10

# Restart system to clear memory
python scripts/production_startup.py --restart
```

### Slow Performance

#### Issue: Predictions Taking Too Long

**Symptoms**: Predictions take >30 seconds, system feels sluggish

**Diagnosis**:
```bash
# Performance benchmarks
python scripts/performance_tests.py --test latency

# Check database performance
sqlite3 data/production.db "EXPLAIN QUERY PLAN SELECT * FROM games ORDER BY game_date DESC LIMIT 100;"

# Check system resources
iostat -x 1 3
top -b -n 1
```

**Solutions**:
```bash
# Optimize database
python matchup_database.py --optimize --vacuum --reindex

# Update indexes
sqlite3 data/production.db "
CREATE INDEX IF NOT EXISTS idx_games_date_teams ON games(game_date, home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_predictions_recent ON predictions(created_at DESC);
"

# Clear old data
python data_utils.py --cleanup-old --days 365

# Optimize model loading
python modeling.py --optimize-loading
```

### Database Issues

#### Issue: Database Locks

**Error Examples**:
```
sqlite3.OperationalError: database is locked
DatabaseError: Could not acquire exclusive lock
```

**Diagnosis**:
```bash
# Check for lock files
ls -la data/*.lock
ls -la data/*-journal

# Check active database connections
lsof data/production.db

# Check for zombie processes
ps aux | grep -E "(sqlite|python.*database)"
```

**Solutions**:
```bash
# Kill processes using database
pkill -f "python.*live_prediction"
sleep 5

# Remove lock files
rm -f data/*.lock
rm -f data/*-journal

# Check database integrity
sqlite3 data/production.db "PRAGMA integrity_check;"

# Restart with WAL mode
python matchup_database.py --enable-wal
```

## 🔐 Security & Access Issues

### Permission Problems

#### Issue: Access Denied Errors

**Error Examples**:
```
PermissionError: Access denied to 'logs/application.log'
OSError: Cannot create directory 'backups/'
```

**Diagnosis**:
```bash
# Check current permissions
ls -la
ls -la data/ logs/ backups/

# Check user context
whoami
groups

# Check SELinux/AppArmor (if applicable)
sestatus 2>/dev/null || echo "SELinux not active"
```

**Solutions**:
```bash
# Fix directory permissions
chmod 755 .
chmod 700 data/ logs/ backups/
chmod 644 *.py
chmod 600 .env*

# Fix ownership
sudo chown -R $USER:$USER .

# For systemd service
sudo chown -R baseball-hr:baseball-hr /opt/baseball-hr
```

### Security Validation Issues

#### Issue: Security Audit Failures

**Error Examples**:
```
SecurityError: Sensitive files world-readable
Warning: Hardcoded credentials detected
Critical: Encryption keys not secured
```

**Diagnosis**:
```bash
# Run security audit
python security/security_validator.py --audit all

# Check file permissions
find . -type f -perm /o+w -ls

# Check for sensitive data exposure
python security/security_validator.py --audit sensitive
```

**Solutions**:
```bash
# Apply security hardening
python scripts/security_hardening.py --fix-all

# Secure sensitive files
chmod 600 .env* config/secrets/*
chmod 700 config/secrets/

# Remove sensitive data from code
python security/security_validator.py --fix-sensitive --backup

# Update security headers
python security/security_validator.py --apply-headers
```

## 🚀 Deployment & Production Issues

### Service Management Problems

#### Issue: Service Won't Start

**Error Examples**:
```
systemctl: Failed to start baseball-hr.service
ExecStart process exited with failure
Service failed to start within timeout
```

**Diagnosis**:
```bash
# Check service status
sudo systemctl status baseball-hr

# Check service logs
sudo journalctl -u baseball-hr -n 50

# Test manual startup
python scripts/production_startup.py --validate --verbose

# Check service configuration
cat /etc/systemd/system/baseball-hr.service
```

**Solutions**:
```bash
# Fix service file
sudo nano /etc/systemd/system/baseball-hr.service

# Reload systemd configuration
sudo systemctl daemon-reload

# Check service dependencies
sudo systemctl list-dependencies baseball-hr

# Test with different user
sudo -u baseball-hr python scripts/production_startup.py --test
```

#### Issue: Service Keeps Crashing

**Symptoms**: Service starts then immediately stops

**Diagnosis**:
```bash
# Check crash logs
sudo journalctl -u baseball-hr --since "1 hour ago"

# Check for core dumps
ls -la /var/crash/ core*

# Monitor service startup
sudo systemctl start baseball-hr && sleep 10 && sudo systemctl status baseball-hr
```

**Solutions**:
```bash
# Increase startup timeout
sudo systemctl edit baseball-hr
# Add: [Service]
#      TimeoutStartSec=300

# Check resource limits
ulimit -a

# Add restart policy
# RestartSec=10
# Restart=always

# Debug startup sequence
python scripts/production_startup.py --debug --step-by-step
```

### Container Issues (Docker)

#### Issue: Container Won't Start

**Error Examples**:
```
docker: Error response from daemon: failed to create endpoint
Container exits immediately with code 1
OCI runtime create failed
```

**Diagnosis**:
```bash
# Check container logs
docker logs baseball-hr-prod

# Check container configuration
docker inspect baseball-hr-prod

# Test image directly
docker run -it --entrypoint=/bin/bash baseball-hr-predictor

# Check resource constraints
docker stats
```

**Solutions**:
```bash
# Rebuild image
docker build --no-cache -t baseball-hr-predictor .

# Check Dockerfile
cat Dockerfile

# Fix environment variables
docker run --env-file .env baseball-hr-predictor

# Increase resource limits
docker run -m 2g --cpus="2.0" baseball-hr-predictor
```

## 📊 Monitoring & Alerting Issues

### Monitoring System Problems

#### Issue: Monitoring Not Working

**Symptoms**: No metrics, alerts not firing, dashboard empty

**Diagnosis**:
```bash
# Check monitoring service
python monitoring/monitoring_integration.py --status

# Test metrics collection
python monitoring/system_monitor.py --test

# Check monitoring configuration
python config/environment_config.py --show monitoring
```

**Solutions**:
```bash
# Restart monitoring
python monitoring/monitoring_integration.py --restart

# Reset monitoring data
rm -rf logs/monitoring_*
python monitoring/monitoring_integration.py --initialize

# Fix monitoring permissions
chmod 755 logs/
chmod 644 logs/*.log
```

#### Issue: False Alerts

**Symptoms**: Too many alerts, alerts for normal conditions

**Diagnosis**:
```bash
# Check alert thresholds
python monitoring/error_tracker.py --show-rules

# Review recent alerts
python monitoring/error_tracker.py --recent --hours 6

# Analyze alert patterns
grep "ALERT" logs/monitoring.log | tail -20
```

**Solutions**:
```bash
# Adjust alert thresholds
python monitoring/error_tracker.py --update-threshold error_rate 15

# Disable noisy alerts temporarily
python monitoring/error_tracker.py --disable-rule "High Error Rate"

# Tune alert sensitivity
python monitoring/monitoring_integration.py --configure-alerts
```

## 🔄 Backup & Recovery Issues

### Backup Problems

#### Issue: Backups Failing

**Error Examples**:
```
BackupError: Insufficient disk space
PermissionError: Cannot write backup file
Backup verification failed
```

**Diagnosis**:
```bash
# Check backup status
python backup/backup_manager.py --status

# Check disk space
df -h
du -sh backups/

# Test backup process
python backup/backup_manager.py --create-backup --dry-run
```

**Solutions**:
```bash
# Clean old backups
python backup/backup_manager.py --cleanup

# Fix backup permissions
chmod 700 backups/
chown $USER:$USER backups/

# Change backup location
export BACKUP_ROOT=/opt/backups
python backup/backup_manager.py --create-backup
```

### Recovery Problems

#### Issue: Recovery Failing

**Error Examples**:
```
RestoreError: Backup file corrupted
Recovery verification failed
Cannot restore: incompatible backup version
```

**Diagnosis**:
```bash
# List available backups
ls -la backups/

# Test backup integrity
python backup/backup_manager.py --verify-backup backup_file.tar.gz

# Check backup compatibility
python backup/disaster_recovery_plan.py --assess database_corruption
```

**Solutions**:
```bash
# Try different backup
python backup/backup_manager.py --restore previous_backup.tar.gz

# Manual recovery
tar -tzf backup_file.tar.gz | head -10
tar -xzf backup_file.tar.gz -C /tmp/recovery/

# Partial recovery
python backup/backup_manager.py --restore-component database backup_file.tar.gz
```

## 🎯 Performance Optimization

### Optimization Strategies

#### Database Optimization
```bash
# Full database optimization
sqlite3 data/production.db "
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;
VACUUM;
REINDEX;
"

# Update table statistics
sqlite3 data/production.db "ANALYZE;"
```

#### Memory Optimization
```bash
# Clear Python caches
python -c "
import gc
import sys
gc.collect()
print(f'Memory cleared, objects: {len(gc.get_objects())}')
"

# Optimize model loading
python modeling.py --optimize-memory --lazy-loading

# Reduce feature cache size
export FEATURE_CACHE_SIZE=500
python live_prediction_system.py
```

#### Network Optimization
```bash
# Enable connection pooling
export API_CONNECTION_POOL=true

# Adjust timeouts
export API_TIMEOUT=60
export API_RETRY_DELAY=2

# Enable compression
export API_COMPRESSION=true
```

## 📞 Getting Additional Help

### Self-Diagnosis Tools

#### Comprehensive System Report
```bash
# Generate full diagnostic report
python scripts/generate_diagnostic_report.py --comprehensive

# This creates:
# - System configuration report
# - Performance benchmarks
# - Error log analysis
# - Security audit results
# - Database health check
```

#### Health Check Suite
```bash
# Full health validation
python monitoring/health_checks.py --comprehensive --verbose --export

# This checks:
# - System resources
# - Database connectivity
# - API availability
# - Model loading
# - Configuration validity
```

### Log Analysis

#### Important Log Files
```bash
# Application logs
tail -f logs/application.log

# Error logs
tail -f logs/error_tracking.log

# Performance logs
tail -f logs/performance.log

# Security logs
tail -f logs/security_audit.log

# System logs (if using systemd)
sudo journalctl -u baseball-hr -f
```

#### Log Analysis Commands
```bash
# Find recent errors
grep -i "error\|exception\|failed" logs/application.log | tail -10

# Analyze error patterns
grep "ERROR" logs/error_tracking.log | cut -d' ' -f4 | sort | uniq -c | sort -nr

# Check performance trends
grep "PERFORMANCE" logs/application.log | tail -20
```

### Emergency Contacts

#### Support Escalation Levels

1. **Self-Help** (Try first):
   - Run diagnostic tools
   - Check this troubleshooting guide
   - Review log files

2. **System Issues**:
   - Hardware/OS problems
   - Resource constraints
   - Network connectivity

3. **Application Issues**:
   - Model performance
   - Prediction accuracy
   - Configuration problems

4. **Data Issues**:
   - API connectivity
   - Data quality problems
   - External service outages

### Recovery Procedures

#### Emergency Recovery Checklist

1. **Stop all services**
2. **Create emergency backup** of current state
3. **Identify the problem** using diagnostic tools
4. **Try least destructive solution** first
5. **Test solution** in isolated environment if possible
6. **Apply fix** and monitor results
7. **Document resolution** for future reference

#### Disaster Recovery

For complete system failure:
```bash
# 1. Assess damage
python backup/disaster_recovery_plan.py --assess complete_system_loss

# 2. Execute recovery
python backup/disaster_recovery_plan.py --execute complete_system_loss

# 3. Verify recovery
python monitoring/health_checks.py --comprehensive

# 4. Resume operations
python scripts/production_startup.py --post-recovery
```

---

**Troubleshooting Guide Version**: 1.0.0  
**Last Updated**: August 2025  
**Coverage**: Common issues and resolutions for system version 1.0.0+

*If you encounter an issue not covered in this guide, please run the diagnostic report generator and check the system logs for additional clues.*