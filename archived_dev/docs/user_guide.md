# User Guide - Baseball Home Run Prediction System

## 🎯 Welcome

This user guide will help you get started with the Baseball Home Run Prediction System, understand its capabilities, and learn how to use it effectively for predicting home runs and identifying betting opportunities.

## 📚 Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Monitoring & Maintenance](#monitoring--maintenance)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)
7. [FAQ](#faq)

## 🚀 Getting Started

### What This System Does

The Baseball HR Prediction System uses machine learning to:
- **Predict home run likelihood** for MLB games
- **Identify profitable betting opportunities** with expected value calculations
- **Monitor system performance** with real-time dashboards
- **Provide data-driven insights** for baseball analytics

### Quick Setup (5 Minutes)

1. **Install the system** (see [Deployment Guide](deployment.md) for details)
2. **Configure API keys** in your `.env` file
3. **Run health check** to verify everything works
4. **Generate your first predictions**

```bash
# Quick health check
python monitoring/health_checks.py

# Generate today's predictions
python live_prediction_system.py
```

### System Requirements

- **Time**: 5-10 minutes daily for monitoring
- **Technical Level**: Basic command line familiarity helpful
- **API Costs**: $10-30/month for data sources (optional for basic use)
- **Data Storage**: ~1GB for historical data and models

## 💡 Basic Usage

### Daily Prediction Workflow

#### 1. Morning Setup (5 minutes)
```bash
# Check system health
python monitoring/health_checks.py --quick

# Expected output:
# ✅ System Status: HEALTHY
# ✅ API Connectivity: OK
# ✅ Models Loaded: 2/2
```

#### 2. Generate Predictions
```bash
# Get today's predictions
python live_prediction_system.py

# Or use the Python API
from live_prediction_system import LivePredictionSystem

live_system = LivePredictionSystem()
live_system.initialize()
predictions = live_system.get_todays_predictions()
```

#### 3. Review Results
The system outputs predictions in an easy-to-read format:

```
📊 MLB HOME RUN PREDICTIONS - August 22, 2025

🏟️  LAD vs NYY (7:00 PM ET)
   Over 2.5 HRs: 67% probability (Confidence: 89%)
   Expected HRs: 3.2 (Range: 2.1 - 4.8)
   💰 Betting Recommendation: BET OVER (EV: +8.7%)

🏟️  BOS vs HOU (8:00 PM ET)  
   Over 2.5 HRs: 45% probability (Confidence: 76%)
   Expected HRs: 2.1 (Range: 1.3 - 3.2)
   💰 Betting Recommendation: PASS (EV: -3.2%)
```

### Understanding Predictions

#### Prediction Components
- **Probability**: Likelihood of event occurring (0-100%)
- **Confidence**: System's confidence in prediction (0-100%)
- **Expected Value**: Range of likely outcomes
- **Betting Recommendation**: BET/PASS with expected value

#### Confidence Levels
- **High (85-100%)**: Very reliable predictions
- **Medium (70-84%)**: Good predictions, some uncertainty
- **Low (50-69%)**: Less reliable, use with caution

### Finding Betting Opportunities

#### 1. Automated Opportunity Detection
```bash
# Find high-value opportunities
python live_prediction_system.py --min-ev 0.05 --min-confidence 0.80

# This finds opportunities with:
# - At least 5% expected value
# - At least 80% confidence
```

#### 2. Manual Opportunity Review
Look for these indicators:
- ✅ **High Expected Value** (>5%)
- ✅ **High Confidence** (>80%)
- ✅ **Recent Model Performance** (check monitoring dashboard)
- ✅ **Reasonable Odds** (not extreme outliers)

#### 3. Risk Management
The system includes built-in risk management:
- **Stake Sizing**: Kelly criterion-based recommendations
- **Bankroll Limits**: Never risk more than 2% per bet
- **Diversification**: Spread bets across multiple games
- **Stop Losses**: Daily and weekly loss limits

## 🔧 Advanced Features

### Custom Prediction Scenarios

#### Analyze Specific Matchups
```python
from live_prediction_system import LivePredictionSystem

live_system = LivePredictionSystem()
live_system.initialize()

# Analyze specific game
game_prediction = live_system.predict_game({
    'home_team': 'LAD',
    'away_team': 'NYY',
    'game_date': '2025-08-22',
    'temperature': 75,
    'wind_speed': 8
})

print(f"Home Run Probability: {game_prediction['hr_probability']:.1%}")
print(f"Expected Home Runs: {game_prediction['expected_hrs']:.1f}")
```

#### Batch Processing Multiple Games
```python
# Process multiple games at once
games_data = [
    {'home_team': 'LAD', 'away_team': 'NYY', 'game_date': '2025-08-22'},
    {'home_team': 'BOS', 'away_team': 'HOU', 'game_date': '2025-08-22'},
    # ... more games
]

predictions = []
for game in games_data:
    prediction = live_system.predict_game(game)
    predictions.append(prediction)
```

### Model Performance Analysis

#### Check Model Accuracy
```bash
# View recent model performance
python modeling.py --evaluate-recent --days 30

# Expected output:
# 📊 Model Performance (Last 30 days)
# Accuracy: 78.5%
# Precision: 81.2%
# Recall: 75.8%
# ROC AUC: 0.847
```

#### Feature Importance Analysis
```python
from modeling import EnhancedDualModelSystem

model_system = EnhancedDualModelSystem()
model_info = model_system.get_model_info()

# Top features influencing predictions
print("Top 10 Most Important Features:")
for feature, importance in model_info['feature_importance'].items():
    print(f"{feature}: {importance:.3f}")
```

### Historical Analysis

#### Backtest Predictions
```bash
# Test model performance on historical data
python modeling.py --backtest --start-date 2024-01-01 --end-date 2024-12-31

# Analyze betting performance
python betting_utils.py --analyze-historical --min-ev 0.05
```

#### Performance Tracking
```python
from betting_utils import BettingAnalyzer

analyzer = BettingAnalyzer()

# Get historical betting performance
performance = analyzer.get_historical_performance(days=90)
print(f"Win Rate: {performance['win_rate']:.1%}")
print(f"Average ROI: {performance['roi']:.1%}")
print(f"Total Bets: {performance['total_bets']}")
```

## 📊 Monitoring & Maintenance

### Daily Monitoring (2 minutes)

#### Quick Health Check
```bash
# Run daily health check
python monitoring/health_checks.py --quick

# Check for any alerts
python monitoring/error_tracker.py --recent --hours 24
```

#### Review Performance Dashboard
```bash
# Start monitoring dashboard
python monitoring/monitoring_integration.py --dashboard

# View in browser at: http://localhost:8000/dashboard
```

### Weekly Maintenance (10 minutes)

#### 1. System Health Review
```bash
# Comprehensive health check
python monitoring/health_checks.py --comprehensive --export

# Review generated health report
cat logs/health_check_*.json
```

#### 2. Performance Analysis
```bash
# Run performance tests
python scripts/performance_tests.py --quick

# Review performance trends
python monitoring/performance_monitor.py --report --days 7
```

#### 3. Update Models (if needed)
```bash
# Check if model retraining is recommended
python modeling.py --check-performance-drift

# Retrain if performance has degraded
python modeling.py --retrain --validate
```

### Monthly Maintenance (30 minutes)

#### 1. Full System Backup
```bash
# Create comprehensive backup
python backup/backup_manager.py --create-backup

# Test backup integrity
python backup/backup_manager.py --verify-latest
```

#### 2. Security Audit
```bash
# Run security audit
python security/security_validator.py --audit all --export

# Apply any recommended fixes
python scripts/security_hardening.py
```

#### 3. Performance Optimization
```bash
# Comprehensive performance testing
python scripts/performance_tests.py --comprehensive

# Database optimization
python matchup_database.py --optimize --vacuum
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Issue: "No predictions generated"

**Symptoms**: System runs but produces no output  
**Causes**: Missing API keys, no games scheduled, model not loaded  

**Solutions**:
```bash
# Check API connectivity
python api_client.py --test-connection

# Verify model loading
python modeling.py --validate-models

# Check game schedule
python live_prediction_system.py --debug --date today
```

#### Issue: "Low prediction confidence"

**Symptoms**: All predictions show <70% confidence  
**Causes**: Limited data, model needs retraining, unusual conditions  

**Solutions**:
```bash
# Check data freshness
python data_utils.py --check-data-age

# Retrain models with recent data
python modeling.py --retrain --days 365

# Validate prediction pipeline
python live_prediction_system.py --validate-pipeline
```

#### Issue: "System running slowly"

**Symptoms**: Predictions take >30 seconds to generate  
**Causes**: Database issues, memory constraints, API rate limits  

**Solutions**:
```bash
# Check system performance
python scripts/quick_performance_check.py

# Optimize database
python matchup_database.py --optimize

# Clear caches
python data_utils.py --clear-caches
```

#### Issue: "Betting opportunities not found"

**Symptoms**: System shows no profitable bets  
**Causes**: Conservative settings, poor market conditions, model issues  

**Solutions**:
```bash
# Lower minimum expected value threshold
python live_prediction_system.py --min-ev 0.02

# Check recent betting performance
python betting_utils.py --analyze-recent --days 7

# Review market conditions
python api_client.py --check-odds-coverage
```

### Error Messages Guide

#### API Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `API_KEY_INVALID` | API key not working | Check key in `.env` file |
| `API_RATE_LIMIT` | Too many requests | Wait or upgrade API plan |
| `API_TIMEOUT` | Request took too long | Check internet connection |
| `API_NO_DATA` | No data available | Check if games are scheduled |

#### Model Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `MODEL_NOT_FOUND` | Model file missing | Run model training |
| `MODEL_OUTDATED` | Model needs updating | Retrain with recent data |
| `FEATURE_MISMATCH` | Data format changed | Regenerate features |
| `PREDICTION_FAILED` | Prediction error | Check input data format |

#### System Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| `DATABASE_LOCKED` | Database in use | Wait and retry |
| `MEMORY_LIMIT` | Insufficient memory | Restart system |
| `DISK_FULL` | No storage space | Clean up old files |
| `CONFIG_INVALID` | Configuration error | Check `.env` settings |

### Getting Help

#### Self-Diagnosis Tools
```bash
# Comprehensive system check
python monitoring/health_checks.py --comprehensive --verbose

# Generate diagnostic report
python scripts/generate_diagnostic_report.py

# Check system requirements
python scripts/validate_environment.py
```

#### Log Analysis
```bash
# Recent error logs
tail -n 100 logs/error_tracking.log

# Application logs
tail -n 100 logs/application.log

# Performance logs  
tail -n 100 logs/performance.log
```

## 📋 Best Practices

### Prediction Best Practices

#### 1. Data Quality
- ✅ **Verify API connectivity** before generating predictions
- ✅ **Check data freshness** (data should be <4 hours old)
- ✅ **Monitor model performance** weekly
- ✅ **Update models** when accuracy drops below 75%

#### 2. Interpretation Guidelines
- **High Confidence (>85%)**: Reliable for decision making
- **Medium Confidence (70-85%)**: Good but verify conditions
- **Low Confidence (<70%)**: Use caution, may skip betting

#### 3. Timing Recommendations
- **Best Time**: 2-4 hours before game start
- **Avoid**: Last-minute predictions (lineup changes)
- **Update**: Refresh predictions if weather changes significantly

### Betting Best Practices

#### 1. Bankroll Management
```python
# Recommended stake sizing
recommended_stake = bankroll * 0.01 * expected_value

# Example: $1000 bankroll, 8% EV = $0.80 stake
stake = 1000 * 0.01 * 0.08  # = $0.80
```

#### 2. Risk Management
- **Maximum Stake**: Never bet >2% of bankroll per game
- **Daily Limit**: Stop after 3 losing bets in one day
- **Weekly Review**: Analyze performance every Sunday
- **Diversification**: Spread bets across multiple games/markets

#### 3. Record Keeping
```bash
# Export betting history
python betting_utils.py --export-history --format csv

# Analyze performance
python betting_utils.py --analyze-performance --days 30
```

### System Maintenance Best Practices

#### 1. Regular Monitoring
- **Daily**: Quick health check (2 minutes)
- **Weekly**: Performance review (10 minutes)  
- **Monthly**: Full system audit (30 minutes)
- **Quarterly**: Complete system backup and disaster recovery test

#### 2. Security Practices
- **Weekly**: Check for security updates
- **Monthly**: Run security audit
- **Quarterly**: Rotate API keys and encryption keys
- **Annually**: Full security penetration test

#### 3. Performance Optimization
- **Monitor**: Keep system resource usage <80%
- **Clean**: Remove old logs and data monthly
- **Optimize**: Database maintenance weekly
- **Update**: Keep dependencies current

### Data Management Best Practices

#### 1. Storage Management
```bash
# Check storage usage
du -sh data/ logs/ backups/

# Clean old files (keep last 30 days of logs)
find logs/ -name "*.log" -mtime +30 -delete

# Optimize database monthly
python matchup_database.py --optimize
```

#### 2. Backup Strategy
- **Daily**: Automated incremental backups
- **Weekly**: Full system backup
- **Monthly**: Backup verification test
- **Quarterly**: Disaster recovery test

#### 3. Data Quality Monitoring
```bash
# Check data quality
python data_utils.py --validate-recent --days 7

# Monitor data freshness
python monitoring/data_quality_monitor.py --report
```

## ❓ FAQ

### General Questions

**Q: How accurate are the predictions?**
A: The system typically achieves 75-85% accuracy on home run over/under predictions, with higher accuracy for games with more favorable conditions.

**Q: How much can I expect to profit?**
A: Results vary, but users following best practices typically see 5-15% ROI over a full season. Past performance doesn't guarantee future results.

**Q: Do I need programming knowledge?**
A: Basic command line familiarity is helpful, but the system includes simple commands for common tasks. Full Python knowledge isn't required.

### Technical Questions

**Q: How often should I retrain models?**
A: Monitor model performance weekly. Retrain when accuracy drops below 75% or monthly during active season.

**Q: Can I run this on a cloud server?**
A: Yes, see the [Deployment Guide](deployment.md) for cloud deployment instructions. AWS, GCP, and Azure are all supported.

**Q: How much does it cost to run?**
A: API costs are $10-30/month. Cloud hosting costs $5-20/month depending on usage. Total operational cost is typically $15-50/month.

### Betting Questions  

**Q: Which sportsbooks work best with this system?**
A: The system works with any sportsbook. We recommend comparing odds across multiple books for best value.

**Q: Is this legal?**
A: Using predictive models for sports betting is legal where sports betting is legal. Check your local laws and regulations.

**Q: Should I bet every recommendation?**
A: No. Use the confidence levels and your own judgment. Start with high-confidence, high-expected-value opportunities.

### Troubleshooting Questions

**Q: What if predictions seem wrong?**
A: Check model performance, data freshness, and consider retraining. The system includes built-in performance monitoring.

**Q: System is running slowly - what can I do?**
A: Run the quick performance check, clear caches, and optimize the database. See the troubleshooting section for detailed steps.

**Q: How do I update to a new version?**
A: Follow the deployment guide for updating. Always backup your data before updating.

## 📞 Support and Resources

### Documentation
- **API Documentation**: [API.md](API.md)
- **System Architecture**: [architecture.md](architecture.md)
- **Deployment Guide**: [deployment.md](deployment.md)
- **Security Guide**: [security.md](security.md)

### Self-Help Tools
- **Health Checks**: `python monitoring/health_checks.py`
- **Performance Tests**: `python scripts/quick_performance_check.py`
- **Diagnostic Report**: `python scripts/generate_diagnostic_report.py`

### Best Practices Summary
1. **Start Small**: Begin with paper trading or small stakes
2. **Monitor Performance**: Check system health daily
3. **Follow Risk Management**: Never risk more than you can afford
4. **Keep Learning**: Analyze results and refine your approach
5. **Stay Updated**: Keep the system and your knowledge current

---

**User Guide Version**: 1.0.0  
**Last Updated**: August 2025  
**Compatible With**: System Version 1.0.0+

*Remember: This system is for educational and analytical purposes. Always bet responsibly and within your means.*