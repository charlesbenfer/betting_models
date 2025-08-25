# Baseball Home Run Prediction System

## Overview

A comprehensive machine learning system for predicting home run occurrences in MLB games with integrated betting analysis and real-time data processing capabilities.

## 🎯 Key Features

- **Advanced ML Models**: XGBoost and ensemble models with 255+ engineered features
- **Real-time Predictions**: Live game predictions with betting opportunity identification  
- **Production Monitoring**: Comprehensive system monitoring and alerting
- **Security Hardened**: Enterprise-grade security validation and hardening
- **Disaster Recovery**: Automated backup and recovery procedures
- **Performance Optimized**: Sub-second prediction latency with minimal resource usage

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Data Pipeline   │───▶│  ML Pipeline    │
│  • The Odds API │    │  • Data Fetching │    │  • Feature Eng. │
│  • Weather APIs │    │  • Validation    │    │  • Model Train  │
│  • Stats APIs   │    │  • Integration   │    │  • Prediction   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│  Live System     │───▶│  Betting Engine │
│  • Performance │    │  • Predictions   │    │  • Odds Analysis│
│  • Health       │    │  • Scheduling    │    │  • Opportunities│
│  • Alerts       │    │  • Management    │    │  • Risk Mgmt    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- SQLite 3
- 4GB RAM minimum, 8GB recommended
- API keys for data sources (optional for basic functionality)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd betting_models

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup configuration
cp .env.template .env
# Edit .env with your API keys and settings
```

### Basic Usage

```bash
# Run system health check
python monitoring/health_checks.py

# Generate predictions for today
python live_prediction_system.py

# Start monitoring dashboard
python monitoring/monitoring_integration.py --dashboard

# Create system backup
python backup/backup_manager.py --create-backup
```

## 📁 Project Structure

```
betting_models/
├── 📊 Core ML System
│   ├── modeling.py                    # ML models and training
│   ├── dataset_builder.py            # Feature engineering pipeline
│   ├── live_prediction_system.py     # Real-time prediction engine
│   └── betting_utils.py              # Betting analysis tools
│
├── 🔧 Configuration & Setup
│   ├── config/
│   │   ├── environment_config.py     # Environment-specific settings
│   │   ├── secrets_manager.py        # Secure secrets management
│   │   └── config_validator.py       # Configuration validation
│   └── config.py                     # Main configuration
│
├── 📡 Data & Integration  
│   ├── api_client.py                 # External API integration
│   ├── data_utils.py                 # Data processing utilities
│   └── matchup_database.py           # Database management
│
├── 🖥️ Production Systems
│   ├── scripts/
│   │   ├── production_startup.py     # Production deployment
│   │   ├── deploy.sh                 # Automated deployment
│   │   ├── performance_tests.py      # Performance validation
│   │   └── security_hardening.py     # Security automation
│   │
│   ├── monitoring/
│   │   ├── monitoring_integration.py # Integrated monitoring
│   │   ├── system_monitor.py         # System metrics
│   │   ├── error_tracker.py          # Error tracking & alerts
│   │   ├── performance_monitor.py    # Performance monitoring
│   │   └── health_checks.py          # Health validation
│   │
│   ├── security/
│   │   └── security_validator.py     # Security validation
│   │
│   └── backup/
│       ├── backup_manager.py         # Backup automation
│       └── disaster_recovery_plan.py # Disaster recovery
│
├── 📖 Documentation
│   ├── docs/                         # Comprehensive documentation
│   ├── README.md                     # This file
│   └── API.md                        # API documentation
│
└── 🗂️ Data & Models
    ├── data/                         # Training and processed data
    ├── models/                       # Trained ML models
    ├── logs/                         # System logs
    └── backups/                      # System backups
```

## 🎮 Core Components

### Machine Learning Pipeline

**Enhanced Dual Model System** (`modeling.py`)
- Primary: XGBoost classifier optimized for home run prediction
- Secondary: Ensemble model combining multiple algorithms
- 255+ engineered features including weather, player stats, ballpark factors
- Time-based validation with rolling window training

**Feature Engineering** (`dataset_builder.py`)
- **Player Features**: Career stats, recent form, matchup history
- **Environmental**: Weather conditions, ballpark dimensions, elevation
- **Game Context**: Inning, score situation, base runners
- **Advanced Metrics**: Expected stats, park-adjusted performance

### Live Prediction System

**Real-time Engine** (`live_prediction_system.py`)
- Fetches current game data and odds
- Generates predictions with confidence intervals
- Identifies profitable betting opportunities
- Caches results for performance optimization

**Betting Analysis** (`betting_utils.py`)
- Expected value calculations
- Risk assessment and bankroll management
- Opportunity ranking and filtering
- Historical performance tracking

### Production Infrastructure

**Monitoring & Observability**
- Real-time system health monitoring
- Performance metrics and alerting
- Error tracking with automatic classification
- Comprehensive logging and audit trails

**Security & Compliance**
- Input validation and sanitization
- API rate limiting and security headers
- Encrypted secrets management
- Vulnerability scanning and hardening

**Backup & Recovery**
- Automated full system backups
- Database-specific backup procedures
- Disaster recovery automation
- Configurable retention policies

## 🔧 Configuration

### Environment Variables

```bash
# Core Settings
BASEBALL_HR_ENV=development          # Environment: development/staging/production
DEBUG=true                           # Enable debug mode

# API Configuration
THEODDS_API_KEY=your_api_key        # Required: The Odds API
VISUALCROSSING_API_KEY=your_key     # Optional: Weather data

# Database
DATABASE_URL=sqlite:///data/baseball_hr.db
DATABASE_BACKUP_ENABLED=true

# Security
BASEBALL_HR_ENCRYPTION_KEY=auto     # Auto-generated encryption key
SECURE_HEADERS_ENABLED=true
RATE_LIMITING_ENABLED=true

# Monitoring
MONITORING_ENABLED=true
LOG_LEVEL=INFO
ALERT_EMAILS=admin@example.com
```

### Model Configuration

Models are configured in `config.py`:

```python
# Model settings
MODEL_DIR = "saved_models_pregame"
MIN_EV_THRESHOLD = 0.05      # 5% minimum expected value
MIN_PROB_THRESHOLD = 0.15    # 15% minimum probability
PREDICTION_CONFIDENCE = 0.85  # 85% confidence threshold
```

## 🚀 Deployment

### Development Environment

```bash
# Setup development environment
python scripts/validate_environment.py --environment development

# Start development server
python live_prediction_system.py --debug
```

### Production Deployment

```bash
# Automated production deployment
./scripts/deploy.sh

# Or manual deployment
python scripts/production_startup.py --environment production

# Verify deployment
python monitoring/health_checks.py --comprehensive
```

### Docker Deployment

```bash
# Build container
docker build -t baseball-hr-predictor .

# Run with environment file
docker run --env-file .env -p 8000:8000 baseball-hr-predictor

# Run monitoring dashboard
docker run -it baseball-hr-predictor python monitoring/monitoring_integration.py --dashboard
```

## 📊 Performance Metrics

### System Performance
- **Initialization Time**: < 0.1 seconds
- **Memory Usage**: ~240MB baseline
- **CPU Usage**: < 5% under normal load
- **Prediction Latency**: < 2 seconds per game

### Model Performance
- **Training Accuracy**: 78-82% on validation set
- **Live Performance**: Monitored continuously
- **Feature Importance**: Top features automatically tracked
- **Model Drift**: Detected and alerted automatically

### Betting Performance
- **Opportunity Detection**: ~15-25 opportunities per day
- **Expected Value**: 5-15% on recommended bets
- **Risk Management**: Configurable bankroll limits
- **Historical Tracking**: All decisions logged and analyzed

## 🔐 Security

### Security Features

- **Input Validation**: All inputs validated and sanitized
- **API Security**: Rate limiting and authentication
- **Data Encryption**: Sensitive data encrypted at rest
- **Security Headers**: OWASP recommended headers
- **Audit Logging**: All security events logged

### Security Scanning

```bash
# Run security audit
python security/security_validator.py --audit all

# Security hardening
python scripts/security_hardening.py

# Vulnerability scan
python security/security_validator.py --audit deps
```

## 🔄 Monitoring & Maintenance

### System Monitoring

```bash
# Real-time dashboard
python monitoring/monitoring_integration.py --dashboard

# Health check
python monitoring/health_checks.py

# Performance testing
python scripts/performance_tests.py
```

### Backup & Recovery

```bash
# Create backup
python backup/backup_manager.py --create-backup

# Restore from backup
python backup/backup_manager.py --restore backup_file.tar.gz

# Test disaster recovery
python backup/disaster_recovery_plan.py --test
```

### Log Management

Logs are automatically rotated and organized by component:

```
logs/
├── application.log      # Main application logs
├── error_tracking.log   # Error and exception tracking  
├── performance.log      # Performance metrics
├── security_audit.log   # Security events
└── monitoring/          # Detailed monitoring logs
```

## 🐛 Troubleshooting

### Common Issues

**API Key Errors**
```bash
# Verify API key configuration
python config/secrets_manager.py --validate

# Test API connectivity
python api_client.py --test-connection
```

**Model Loading Issues**
```bash
# Check model files
python modeling.py --validate-models

# Retrain if necessary  
python modeling.py --retrain
```

**Performance Issues**
```bash
# Performance diagnostics
python scripts/quick_performance_check.py

# Detailed performance analysis
python scripts/performance_tests.py
```

**Database Issues**
```bash
# Database integrity check
python matchup_database.py --verify

# Restore from backup
python backup/backup_manager.py --restore latest
```

### Getting Help

1. **Check Logs**: Review relevant log files for error details
2. **Health Check**: Run comprehensive health validation
3. **Performance Test**: Verify system performance benchmarks  
4. **Security Scan**: Ensure no security issues
5. **Documentation**: Consult API and component documentation

### Support Contacts

- **System Issues**: Check monitoring dashboard first
- **Security Concerns**: Run security audit and review results
- **Performance Problems**: Use performance testing suite
- **Data Issues**: Verify with health checks and backups

## 📈 Future Enhancements

### Planned Features

- **Multi-sport Support**: Extend to other sports beyond baseball
- **Advanced Analytics**: Additional statistical models and metrics
- **Web Interface**: Browser-based monitoring and control dashboard
- **Mobile Alerts**: Push notifications for critical events
- **Cloud Deployment**: Containerized deployment on cloud platforms

### Contribution Guidelines

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run full test suite (`python -m pytest tests/`)
4. Ensure security scan passes (`python security/security_validator.py --audit all`)
5. Update documentation as needed
6. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MLB data providers and APIs
- Open source machine learning libraries (scikit-learn, XGBoost)
- Weather data providers
- Sports betting data sources

---

**Version**: 1.0.0  
**Last Updated**: August 2025  
**Status**: Production Ready ✅