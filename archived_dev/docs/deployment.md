# Deployment Guide

## 🚀 Overview

This guide provides comprehensive instructions for deploying the Baseball Home Run Prediction System across different environments. The system supports development, staging, and production deployments with automated validation and monitoring.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **OS**: Linux (Ubuntu 18.04+), macOS 10.15+, Windows 10+
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4GB
- **Storage**: 20GB available space
- **Python**: 3.8 or higher
- **Network**: Broadband internet connection

#### Recommended Requirements (Production)
- **OS**: Ubuntu 20.04 LTS or CentOS 8
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 16GB
- **Storage**: 100GB SSD with backup storage
- **Network**: 1 Gbps connection
- **Monitoring**: Dedicated monitoring instance

#### Software Dependencies
```bash
# Core dependencies
Python 3.8+
pip 21.0+
virtualenv or conda
SQLite 3.31+
Git 2.25+

# Optional dependencies
nginx (for production web serving)
systemd (for service management)
cron (for scheduled tasks)
logrotate (for log management)
```

### External Dependencies

#### Required API Keys
- **The Odds API**: Sports betting odds data (required for live predictions)
  - Get key at: https://the-odds-api.com/
  - Free tier: 500 requests/month
  - Paid tier: Up to 10,000 requests/month

#### Optional API Keys
- **Visual Crossing Weather API**: Enhanced weather data
  - Get key at: https://www.visualcrossing.com/
  - Free tier: 1,000 requests/day
- **Additional sports APIs**: For expanded data sources

## 🛠️ Installation Methods

### Method 1: Automated Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/betting_models.git
cd betting_models

# Run automated setup
./scripts/deploy.sh --environment development

# Follow interactive prompts for configuration
```

### Method 2: Manual Installation

#### Step 1: Environment Setup

```bash
# Create project directory
mkdir -p /opt/baseball-hr
cd /opt/baseball-hr

# Clone repository
git clone https://github.com/yourusername/betting_models.git .

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 2: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# For development with additional tools
pip install -r requirements-dev.txt

# Verify installation
python -c "import sklearn, xgboost, pandas; print('Dependencies installed successfully')"
```

#### Step 3: Configuration Setup

```bash
# Create configuration from template
cp .env.template .env

# Edit configuration file
nano .env

# Set required environment variables
export BASEBALL_HR_ENV=development
export THEODDS_API_KEY=your_api_key_here

# Validate configuration
python config/config_validator.py --validate --environment development
```

#### Step 4: Database Initialization

```bash
# Create data directory
mkdir -p data logs backups

# Initialize database
python matchup_database.py --initialize

# Test database connection
python matchup_database.py --test-connection
```

#### Step 5: System Validation

```bash
# Run comprehensive health check
python monitoring/health_checks.py --comprehensive

# Run quick performance test
python scripts/quick_performance_check.py

# Test API connectivity
python api_client.py --test-connection
```

### Method 3: Docker Installation

```bash
# Build Docker image
docker build -t baseball-hr-predictor .

# Run with environment file
docker run -d \
  --name baseball-hr-prod \
  --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  baseball-hr-predictor

# Verify container is running
docker ps
docker logs baseball-hr-prod
```

## 🌍 Environment-Specific Deployments

### Development Environment

**Purpose**: Local development and testing
**Characteristics**:
- Debug mode enabled
- Verbose logging
- Mock data sources allowed
- Fast iteration cycle

```bash
# Setup development environment
export BASEBALL_HR_ENV=development

# Install development dependencies
pip install -r requirements-dev.txt

# Start development server with hot reload
python live_prediction_system.py --debug --reload

# Run development tests
python -m pytest tests/ -v --cov
```

**Development Configuration**:
```bash
# .env.development
BASEBALL_HR_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///data/dev.db
CACHE_TTL=60
MONITORING_INTERVAL=30
API_RATE_LIMIT=1000
```

### Staging Environment

**Purpose**: Pre-production testing and validation
**Characteristics**:
- Production-like configuration
- Limited external API usage
- Comprehensive monitoring
- Performance testing

```bash
# Setup staging environment
export BASEBALL_HR_ENV=staging

# Deploy to staging
./scripts/deploy.sh --environment staging --validate

# Run staging validation tests
python scripts/performance_tests.py --comprehensive
python backup/disaster_recovery_plan.py --test
```

**Staging Configuration**:
```bash
# .env.staging
BASEBALL_HR_ENV=staging
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///data/staging.db
CACHE_TTL=300
MONITORING_INTERVAL=60
BACKUP_ENABLED=true
ALERT_EMAILS=staging-alerts@company.com
```

### Production Environment

**Purpose**: Live production deployment
**Characteristics**:
- Optimized performance settings
- Enhanced security measures
- Comprehensive monitoring and alerting
- Automated backup and recovery

```bash
# Setup production environment
export BASEBALL_HR_ENV=production

# Create production user
sudo useradd -r -s /bin/false baseball-hr
sudo mkdir -p /opt/baseball-hr
sudo chown baseball-hr:baseball-hr /opt/baseball-hr

# Deploy to production
sudo -u baseball-hr ./scripts/deploy.sh --environment production --backup

# Verify production deployment
python monitoring/health_checks.py --comprehensive --export
python scripts/production_startup.py --validate
```

**Production Configuration**:
```bash
# .env.production
BASEBALL_HR_ENV=production
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=sqlite:///data/production.db
CACHE_TTL=3600
MONITORING_INTERVAL=60
BACKUP_ENABLED=true
BACKUP_RETENTION_DAYS=30
ALERT_EMAILS=production-alerts@company.com
SECURITY_HARDENING=true
RATE_LIMITING_ENABLED=true
SSL_ENABLED=true
```

## 🔧 Configuration Management

### Environment Variables

**Core Settings**:
```bash
# Environment identification
BASEBALL_HR_ENV=production              # Environment type
DEBUG=false                            # Debug mode

# Database configuration
DATABASE_URL=sqlite:///data/production.db
DATABASE_BACKUP_ENABLED=true
DATABASE_MAX_SIZE_MB=1000

# API configuration
THEODDS_API_KEY=your_production_key
VISUALCROSSING_API_KEY=your_weather_key
API_TIMEOUT_SECONDS=30
API_MAX_RETRIES=3
API_RATE_LIMIT_PER_MINUTE=60

# Security settings
BASEBALL_HR_ENCRYPTION_KEY=auto_generated_key
SECURE_HEADERS_ENABLED=true
RATE_LIMITING_ENABLED=true
INPUT_VALIDATION_STRICT=true

# Monitoring and logging
LOG_LEVEL=WARNING
LOG_MAX_FILE_SIZE_MB=100
LOG_BACKUP_COUNT=5
MONITORING_ENABLED=true
MONITORING_INTERVAL_SECONDS=60
ALERT_EMAILS=admin@company.com

# Performance settings
PREDICTION_CACHE_TTL=300
MODEL_CACHE_SIZE=100
FEATURE_CACHE_SIZE=1000
MAX_CONCURRENT_PREDICTIONS=10

# Backup settings
BACKUP_ENABLED=true
BACKUP_COMPRESSION=true
BACKUP_RETENTION_DAYS=30
BACKUP_VERIFICATION=true
```

### Configuration Validation

```bash
# Validate configuration for specific environment
python config/config_validator.py --validate --environment production

# Check deployment readiness
python config/config_validator.py --readiness production

# Export configuration for review
python config/environment_config.py --export /tmp/production_config.json
```

### Secrets Management

```bash
# Initialize secrets manager
python config/secrets_manager.py --environment production

# Set production secrets
python config/secrets_manager.py --set theodds_api_key "your_api_key" "Production API key"
python config/secrets_manager.py --set encryption_key "auto_generated" "System encryption key"

# Validate all secrets are configured
python config/secrets_manager.py --validate

# Export secrets template for new environment
python config/secrets_manager.py --export-template production
```

## 🚀 Deployment Process

### Automated Deployment

The automated deployment script handles the complete deployment process:

```bash
# Full automated deployment
./scripts/deploy.sh --environment production --backup --validate

# Deployment with custom options
./scripts/deploy.sh \
  --environment production \
  --config-file custom_config.py \
  --backup-before-deploy \
  --run-tests \
  --validate-performance \
  --notify-completion
```

**Deployment Script Features**:
- Pre-deployment validation
- Automatic backup creation
- Dependency installation
- Configuration validation
- Database migration
- Service restart
- Post-deployment testing
- Rollback on failure

### Manual Deployment Steps

#### 1. Pre-Deployment Preparation

```bash
# Create backup of current system
python backup/backup_manager.py --create-backup

# Validate current system health
python monitoring/health_checks.py --comprehensive

# Check for configuration changes
python config/config_validator.py --validate --environment production
```

#### 2. Code Deployment

```bash
# Stop current services (if running)
sudo systemctl stop baseball-hr

# Update code from repository
git fetch origin
git checkout main
git pull origin main

# Install/update dependencies
source .venv/bin/activate
pip install -r requirements.txt --upgrade
```

#### 3. Configuration Updates

```bash
# Update configuration files
cp config/production.py config.py

# Update environment variables
source .env.production

# Validate new configuration
python config/config_validator.py --validate --environment production
```

#### 4. Database Migration

```bash
# Check for database schema changes
python matchup_database.py --check-migrations

# Apply database migrations
python matchup_database.py --migrate

# Verify database integrity
python matchup_database.py --verify
```

#### 5. Service Restart

```bash
# Restart application services
sudo systemctl start baseball-hr
sudo systemctl enable baseball-hr

# Verify services are running
sudo systemctl status baseball-hr
```

#### 6. Post-Deployment Validation

```bash
# Run health checks
python monitoring/health_checks.py --comprehensive

# Test core functionality
python live_prediction_system.py --test

# Verify performance benchmarks
python scripts/quick_performance_check.py

# Check monitoring and alerting
python monitoring/monitoring_integration.py --test
```

## 🐳 Docker Deployment

### Production Docker Setup

#### Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -r -s /bin/false appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python monitoring/health_checks.py --quick || exit 1

# Default command
CMD ["python", "scripts/production_startup.py"]
```

#### Docker Compose for Production

```yaml
version: '3.8'

services:
  baseball-hr:
    build: .
    container_name: baseball-hr-prod
    restart: unless-stopped
    environment:
      - BASEBALL_HR_ENV=production
    env_file:
      - .env.production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./backups:/app/backups
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python", "monitoring/health_checks.py", "--quick"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  monitoring:
    build: .
    container_name: baseball-hr-monitoring
    restart: unless-stopped
    command: ["python", "monitoring/monitoring_integration.py", "--dashboard"]
    environment:
      - BASEBALL_HR_ENV=production
    env_file:
      - .env.production
    volumes:
      - ./logs:/app/logs
    ports:
      - "8001:8001"
    depends_on:
      - baseball-hr
```

#### Docker Deployment Commands

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f baseball-hr

# Scale application instances
docker-compose up -d --scale baseball-hr=3

# Update deployment
docker-compose pull
docker-compose up -d

# Backup data volumes
docker run --rm -v baseball-hr-data:/data -v $(pwd)/backup:/backup \
  alpine tar czf /backup/data-backup-$(date +%Y%m%d).tar.gz /data
```

## 🔄 Service Management

### Systemd Service (Linux)

#### Service Configuration

Create `/etc/systemd/system/baseball-hr.service`:

```ini
[Unit]
Description=Baseball HR Prediction Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=baseball-hr
Group=baseball-hr
WorkingDirectory=/opt/baseball-hr
Environment=PATH=/opt/baseball-hr/.venv/bin
Environment=BASEBALL_HR_ENV=production
EnvironmentFile=/opt/baseball-hr/.env.production
ExecStart=/opt/baseball-hr/.venv/bin/python scripts/production_startup.py
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=baseball-hr

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/baseball-hr/data /opt/baseball-hr/logs /opt/baseball-hr/backups

[Install]
WantedBy=multi-user.target
```

#### Service Management Commands

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable baseball-hr
sudo systemctl start baseball-hr

# Service status and control
sudo systemctl status baseball-hr
sudo systemctl stop baseball-hr
sudo systemctl restart baseball-hr

# View service logs
sudo journalctl -u baseball-hr -f
sudo journalctl -u baseball-hr --since "1 hour ago"
```

### Process Management (Alternative)

#### Using Supervisor

Install and configure supervisor:

```bash
# Install supervisor
sudo apt-get install supervisor

# Create configuration
sudo tee /etc/supervisor/conf.d/baseball-hr.conf << EOF
[program:baseball-hr]
command=/opt/baseball-hr/.venv/bin/python scripts/production_startup.py
directory=/opt/baseball-hr
user=baseball-hr
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/baseball-hr.err.log
stdout_logfile=/var/log/supervisor/baseball-hr.out.log
environment=BASEBALL_HR_ENV="production"
EOF

# Start service
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start baseball-hr
```

## 📊 Monitoring Setup

### Production Monitoring

#### Monitoring Service Configuration

```bash
# Start integrated monitoring
python monitoring/monitoring_integration.py --dashboard --interval 60

# Setup monitoring as systemd service
sudo tee /etc/systemd/system/baseball-hr-monitoring.service << EOF
[Unit]
Description=Baseball HR Monitoring Service
After=network.target baseball-hr.service

[Service]
Type=simple
User=baseball-hr
Group=baseball-hr
WorkingDirectory=/opt/baseball-hr
Environment=PATH=/opt/baseball-hr/.venv/bin
EnvironmentFile=/opt/baseball-hr/.env.production
ExecStart=/opt/baseball-hr/.venv/bin/python monitoring/monitoring_integration.py --dashboard
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable baseball-hr-monitoring
sudo systemctl start baseball-hr-monitoring
```

#### Log Rotation Setup

```bash
# Configure logrotate
sudo tee /etc/logrotate.d/baseball-hr << EOF
/opt/baseball-hr/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 baseball-hr baseball-hr
    postrotate
        systemctl reload baseball-hr
    endscript
}
EOF
```

#### Automated Backup Setup

```bash
# Create backup cron job
crontab -e

# Add daily backup at 2:00 AM
0 2 * * * /opt/baseball-hr/.venv/bin/python /opt/baseball-hr/backup/backup_manager.py --create-backup

# Add weekly cleanup at 3:00 AM Sunday
0 3 * * 0 /opt/baseball-hr/.venv/bin/python /opt/baseball-hr/backup/backup_manager.py --cleanup
```

## 🔐 Security Hardening

### Production Security Setup

```bash
# Run security hardening
python scripts/security_hardening.py --directory /opt/baseball-hr

# Apply recommended security settings
python security/security_validator.py --audit all --fix

# Setup firewall rules
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8000/tcp  # Application port
sudo ufw allow 8001/tcp  # Monitoring port (if needed)
```

### SSL/TLS Configuration

```bash
# Install certbot for Let's Encrypt
sudo apt-get install certbot

# Obtain SSL certificate
sudo certbot certonly --standalone -d your-domain.com

# Configure nginx with SSL (if using web server)
sudo tee /etc/nginx/sites-available/baseball-hr << EOF
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://\$server_name\$request_uri;
}
EOF
```

## 🔧 Troubleshooting Deployment

### Common Issues and Solutions

#### Issue: Module Import Errors
```bash
# Solution: Check Python path and virtual environment
source .venv/bin/activate
python -c "import sys; print('\n'.join(sys.path))"
pip list | grep -E "(pandas|sklearn|xgboost)"
```

#### Issue: Database Connection Errors
```bash
# Solution: Check database file permissions and path
ls -la data/
sqlite3 data/production.db ".tables"
python matchup_database.py --test-connection
```

#### Issue: API Authentication Errors
```bash
# Solution: Verify API keys and network connectivity
python config/secrets_manager.py --get theodds_api_key
python api_client.py --test-connection
curl -I https://api.the-odds-api.com/
```

#### Issue: Performance Issues
```bash
# Solution: Run performance diagnostics
python scripts/quick_performance_check.py
python scripts/performance_tests.py --test memory
top -p $(pgrep -f python)
```

#### Issue: Service Won't Start
```bash
# Solution: Check service configuration and logs
sudo systemctl status baseball-hr
sudo journalctl -u baseball-hr -n 50
python scripts/production_startup.py --validate
```

### Health Check Validation

```bash
# Comprehensive deployment validation
python monitoring/health_checks.py --comprehensive --export

# Expected output:
# ✅ Database connectivity: healthy
# ✅ API connectivity: healthy  
# ✅ Model availability: healthy
# ✅ System resources: healthy
# ✅ Configuration: valid
# Overall Status: HEALTHY
```

### Performance Validation

```bash
# Quick performance check
python scripts/quick_performance_check.py

# Expected benchmarks:
# ✅ Initialization: < 10 seconds
# ✅ Memory usage: < 1000 MB
# ✅ Prediction speed: < 3000 ms
# ✅ CPU usage: < 80%
# Overall Status: GOOD
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] System requirements verified
- [ ] Dependencies installed
- [ ] Configuration files prepared
- [ ] API keys configured
- [ ] Database initialized
- [ ] Backup created
- [ ] Security scan completed

### Deployment
- [ ] Code deployed successfully
- [ ] Configuration applied
- [ ] Database migrated
- [ ] Services restarted
- [ ] Health checks passing
- [ ] Monitoring active
- [ ] Logs accessible

### Post-Deployment
- [ ] Functionality testing completed
- [ ] Performance benchmarks met
- [ ] Security validation passed
- [ ] Backup system verified
- [ ] Monitoring alerts configured
- [ ] Documentation updated
- [ ] Team notifications sent

## 📚 Additional Resources

- **System Architecture**: `docs/architecture.md`
- **API Documentation**: `docs/API.md`
- **Security Guide**: `docs/security.md`
- **Performance Tuning**: `docs/performance.md`
- **Troubleshooting**: `docs/troubleshooting.md`

---

**Deployment Version**: 1.0.0  
**Last Updated**: August 2025  
**Tested Environments**: Ubuntu 20.04, CentOS 8, macOS 12+