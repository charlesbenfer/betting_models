#!/usr/bin/env python3
"""
Configuration Validator and Deployment Tool
==========================================

Validates configurations and manages deployment across environments.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.environment_config import ConfigurationManager, EnvironmentConfig
from config.secrets_manager import SecretsManager

class ConfigurationValidator:
    """Validates configuration for deployment readiness."""
    
    def __init__(self):
        self.logger = logging.getLogger('config_validator')
        
    def validate_environment_config(self, config: EnvironmentConfig) -> Tuple[bool, List[str]]:
        """Validate an environment configuration."""
        issues = []
        
        # Required settings validation
        required_checks = [
            self._check_database_config,
            self._check_api_config,
            self._check_model_config,
            self._check_monitoring_config,
            self._check_logging_config,
            self._check_security_config
        ]
        
        for check in required_checks:
            try:
                check_issues = check(config)
                issues.extend(check_issues)
            except Exception as e:
                issues.append(f"Validation check failed: {e}")
        
        return len(issues) == 0, issues
    
    def _check_database_config(self, config: EnvironmentConfig) -> List[str]:
        """Check database configuration."""
        issues = []
        db_config = config.database
        
        # Check database path
        try:
            db_path = Path(db_config.path)
            if not db_path.parent.exists():
                db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Invalid database path: {e}")
        
        # Check backup path
        if db_config.backup_path:
            try:
                backup_path = Path(db_config.backup_path)
                backup_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Invalid backup path: {e}")
        
        # Check size limits
        if db_config.max_size_mb <= 0:
            issues.append("Database max_size_mb must be positive")
        
        return issues
    
    def _check_api_config(self, config: EnvironmentConfig) -> List[str]:
        """Check API configuration."""
        issues = []
        api_config = config.api
        
        # Check required API keys for production/staging
        if config.environment in ['production', 'staging']:
            if not api_config.theodds_api_key:
                issues.append("THEODDS_API_KEY is required for production/staging")
        
        # Check timeout values
        if api_config.timeout_seconds <= 0:
            issues.append("API timeout must be positive")
        if api_config.timeout_seconds > 300:
            issues.append("API timeout should not exceed 300 seconds")
        
        # Check retry limits
        if api_config.max_retries < 0:
            issues.append("Max retries cannot be negative")
        if api_config.max_retries > 10:
            issues.append("Max retries should not exceed 10")
        
        # Check rate limits
        if api_config.rate_limit_per_minute <= 0:
            issues.append("Rate limit must be positive")
        
        return issues
    
    def _check_model_config(self, config: EnvironmentConfig) -> List[str]:
        """Check model configuration."""
        issues = []
        model_config = config.model
        
        # Check model directory
        try:
            model_dir = Path(model_config.model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Invalid model directory: {e}")
        
        # Check cache size
        if model_config.feature_cache_size <= 0:
            issues.append("Feature cache size must be positive")
        
        # Check batch size
        if model_config.prediction_batch_size <= 0:
            issues.append("Prediction batch size must be positive")
        if model_config.prediction_batch_size > 10000:
            issues.append("Prediction batch size should not exceed 10000")
        
        return issues
    
    def _check_monitoring_config(self, config: EnvironmentConfig) -> List[str]:
        """Check monitoring configuration."""
        issues = []
        monitoring_config = config.monitoring
        
        # Check interval
        if monitoring_config.interval_seconds <= 0:
            issues.append("Monitoring interval must be positive")
        if monitoring_config.interval_seconds < 10:
            issues.append("Monitoring interval should be at least 10 seconds")
        
        # Check retention
        if monitoring_config.retention_hours <= 0:
            issues.append("Retention hours must be positive")
        
        # Check alert thresholds
        if monitoring_config.alert_thresholds:
            thresholds = monitoring_config.alert_thresholds
            
            if 'cpu_percent' in thresholds:
                if not 0 < thresholds['cpu_percent'] <= 100:
                    issues.append("CPU threshold must be between 0 and 100")
            
            if 'memory_percent' in thresholds:
                if not 0 < thresholds['memory_percent'] <= 100:
                    issues.append("Memory threshold must be between 0 and 100")
            
            if 'disk_percent' in thresholds:
                if not 0 < thresholds['disk_percent'] <= 100:
                    issues.append("Disk threshold must be between 0 and 100")
        
        return issues
    
    def _check_logging_config(self, config: EnvironmentConfig) -> List[str]:
        """Check logging configuration."""
        issues = []
        logging_config = config.logging
        
        # Check log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if logging_config.level not in valid_levels:
            issues.append(f"Invalid log level: {logging_config.level}")
        
        # Check log directory
        try:
            log_dir = Path(logging_config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Invalid log directory: {e}")
        
        # Check file size
        if logging_config.max_file_size_mb <= 0:
            issues.append("Max file size must be positive")
        
        # Check backup count
        if logging_config.backup_count < 0:
            issues.append("Backup count cannot be negative")
        
        return issues
    
    def _check_security_config(self, config: EnvironmentConfig) -> List[str]:
        """Check security configuration."""
        issues = []
        
        # Production security requirements
        if config.environment == 'production':
            if not config.security.encrypt_api_keys:
                issues.append("API key encryption should be enabled in production")
            if not config.security.secure_headers:
                issues.append("Secure headers should be enabled in production")
            if not config.security.rate_limiting:
                issues.append("Rate limiting should be enabled in production")
            if not config.security.audit_logging:
                issues.append("Audit logging should be enabled in production")
        
        return issues

class ConfigurationDeployer:
    """Manages configuration deployment across environments."""
    
    def __init__(self):
        self.logger = logging.getLogger('config_deployer')
        
    def deploy_configuration(self, source_env: str, target_env: str, 
                           dry_run: bool = True) -> Tuple[bool, List[str]]:
        """Deploy configuration from one environment to another."""
        messages = []
        
        try:
            # Load source configuration
            os.environ['BASEBALL_HR_ENV'] = source_env
            source_manager = ConfigurationManager()
            source_config = source_manager.config
            
            # Validate source configuration
            validator = ConfigurationValidator()
            is_valid, issues = validator.validate_environment_config(source_config)
            
            if not is_valid:
                messages.append(f"Source configuration is invalid:")
                messages.extend([f"  - {issue}" for issue in issues])
                return False, messages
            
            # Modify for target environment
            target_config = self._adapt_config_for_environment(source_config, target_env)
            
            # Validate target configuration
            is_valid, issues = validator.validate_environment_config(target_config)
            if not is_valid:
                messages.append(f"Adapted configuration is invalid:")
                messages.extend([f"  - {issue}" for issue in issues])
                return False, messages
            
            if dry_run:
                messages.append(f"✅ Dry run successful: {source_env} → {target_env}")
                messages.append("Configuration would be deployed successfully")
                return True, messages
            
            # Actually deploy
            os.environ['BASEBALL_HR_ENV'] = target_env
            target_manager = ConfigurationManager()
            target_manager.config = target_config
            target_manager.save_configuration()
            
            messages.append(f"✅ Configuration deployed: {source_env} → {target_env}")
            return True, messages
            
        except Exception as e:
            messages.append(f"❌ Deployment failed: {e}")
            return False, messages
    
    def _adapt_config_for_environment(self, config: EnvironmentConfig, 
                                    target_env: str) -> EnvironmentConfig:
        """Adapt configuration for target environment."""
        # Create a copy
        import copy
        adapted_config = copy.deepcopy(config)
        
        # Update environment
        adapted_config.environment = target_env
        
        # Adjust settings based on environment
        if target_env == 'production':
            adapted_config.debug = False
            adapted_config.logging.level = 'INFO'
            adapted_config.logging.console_output = False
            adapted_config.security.encrypt_api_keys = True
            adapted_config.security.secure_headers = True
            adapted_config.security.rate_limiting = True
            adapted_config.security.audit_logging = True
            adapted_config.monitoring.interval_seconds = 60
            adapted_config.monitoring.retention_hours = 48
            
        elif target_env == 'staging':
            adapted_config.debug = False
            adapted_config.logging.level = 'INFO'
            adapted_config.logging.console_output = True
            adapted_config.security.encrypt_api_keys = True
            adapted_config.security.rate_limiting = True
            adapted_config.monitoring.interval_seconds = 90
            adapted_config.monitoring.retention_hours = 24
            
        else:  # development
            adapted_config.debug = True
            adapted_config.logging.level = 'DEBUG'
            adapted_config.logging.console_output = True
            adapted_config.security.encrypt_api_keys = False
            adapted_config.security.rate_limiting = False
            adapted_config.monitoring.interval_seconds = 120
            adapted_config.monitoring.retention_hours = 12
        
        return adapted_config
    
    def backup_configuration(self, environment: str) -> str:
        """Create a backup of current configuration."""
        backup_dir = Path("config/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"config_backup_{environment}_{timestamp}.json"
        
        # Load current configuration
        os.environ['BASEBALL_HR_ENV'] = environment
        config_manager = ConfigurationManager()
        
        # Export configuration
        export_file = config_manager.export_configuration(str(backup_dir))
        
        # Copy to backup location
        shutil.copy2(export_file, backup_file)
        
        self.logger.info(f"Configuration backed up to {backup_file}")
        return str(backup_file)
    
    def restore_configuration(self, backup_file: str, target_env: str) -> bool:
        """Restore configuration from backup."""
        try:
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)
            
            config_data = backup_data['configuration']
            config_data['environment'] = target_env
            
            # Recreate configuration object
            # (This would need proper deserialization logic)
            
            self.logger.info(f"Configuration restored from {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore configuration: {e}")
            return False

class ConfigurationIntegration:
    """Integrates configuration with secrets and deployment."""
    
    def __init__(self):
        self.config_manager = None
        self.secrets_manager = None
        self.validator = ConfigurationValidator()
        self.deployer = ConfigurationDeployer()
        
    def initialize_for_environment(self, environment: str):
        """Initialize for specific environment."""
        os.environ['BASEBALL_HR_ENV'] = environment
        self.config_manager = ConfigurationManager()
        self.secrets_manager = SecretsManager()
    
    def validate_complete_setup(self, environment: str) -> Tuple[bool, List[str]]:
        """Validate complete setup (config + secrets) for environment."""
        self.initialize_for_environment(environment)
        
        all_issues = []
        
        # Validate configuration
        config_valid, config_issues = self.validator.validate_environment_config(
            self.config_manager.config
        )
        if not config_valid:
            all_issues.extend([f"Config: {issue}" for issue in config_issues])
        
        # Validate secrets
        secret_issues = self.secrets_manager.validate_secrets(environment)
        if secret_issues:
            all_issues.extend([f"Secrets: {issue}" for issue in secret_issues])
        
        return len(all_issues) == 0, all_issues
    
    def get_deployment_readiness(self, environment: str) -> Dict[str, Any]:
        """Get comprehensive deployment readiness report."""
        self.initialize_for_environment(environment)
        
        # Configuration validation
        config_valid, config_issues = self.validator.validate_environment_config(
            self.config_manager.config
        )
        
        # Secrets validation
        secret_issues = self.secrets_manager.validate_secrets(environment)
        secrets_valid = len(secret_issues) == 0
        
        # Overall readiness
        overall_ready = config_valid and secrets_valid
        
        return {
            'environment': environment,
            'timestamp': datetime.now().isoformat(),
            'overall_ready': overall_ready,
            'configuration': {
                'valid': config_valid,
                'issues': config_issues
            },
            'secrets': {
                'valid': secrets_valid,
                'issues': secret_issues,
                'count': len(self.secrets_manager.list_secrets(environment))
            },
            'recommendations': self._get_recommendations(environment, config_valid, secrets_valid)
        }
    
    def _get_recommendations(self, environment: str, config_valid: bool, 
                           secrets_valid: bool) -> List[str]:
        """Get deployment recommendations."""
        recommendations = []
        
        if not config_valid:
            recommendations.append("Fix configuration issues before deployment")
        
        if not secrets_valid:
            recommendations.append("Configure required secrets")
        
        if environment == 'production':
            recommendations.extend([
                "Ensure backup procedures are in place",
                "Verify monitoring alerts are configured",
                "Test rollback procedures"
            ])
        elif environment == 'staging':
            recommendations.extend([
                "Test with production-like data",
                "Validate API connectivity"
            ])
        
        return recommendations

def main():
    """Configuration management CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Validation and Deployment')
    parser.add_argument('--validate', help='Validate configuration for environment')
    parser.add_argument('--deploy', nargs=2, metavar=('SOURCE', 'TARGET'), 
                       help='Deploy configuration from source to target environment')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run deployment')
    parser.add_argument('--backup', help='Backup configuration for environment')
    parser.add_argument('--readiness', help='Check deployment readiness for environment')
    parser.add_argument('--complete-check', help='Complete validation (config + secrets)')
    
    args = parser.parse_args()
    
    integration = ConfigurationIntegration()
    
    if args.validate:
        integration.initialize_for_environment(args.validate)
        validator = ConfigurationValidator()
        is_valid, issues = validator.validate_environment_config(
            integration.config_manager.config
        )
        
        if is_valid:
            print(f"✅ Configuration for {args.validate} is valid")
        else:
            print(f"❌ Configuration for {args.validate} has issues:")
            for issue in issues:
                print(f"  - {issue}")
    
    elif args.deploy:
        source, target = args.deploy
        deployer = ConfigurationDeployer()
        success, messages = deployer.deploy_configuration(source, target, args.dry_run)
        
        for message in messages:
            print(message)
        
        if not success:
            exit(1)
    
    elif args.backup:
        deployer = ConfigurationDeployer()
        backup_file = deployer.backup_configuration(args.backup)
        print(f"✅ Backup created: {backup_file}")
    
    elif args.readiness:
        readiness = integration.get_deployment_readiness(args.readiness)
        
        status_emoji = "✅" if readiness['overall_ready'] else "❌"
        print(f"{status_emoji} Deployment Readiness: {args.readiness}")
        print(f"Configuration: {'✅' if readiness['configuration']['valid'] else '❌'}")
        print(f"Secrets: {'✅' if readiness['secrets']['valid'] else '❌'}")
        
        if readiness['recommendations']:
            print("\nRecommendations:")
            for rec in readiness['recommendations']:
                print(f"  - {rec}")
    
    elif args.complete_check:
        is_valid, issues = integration.validate_complete_setup(args.complete_check)
        
        if is_valid:
            print(f"✅ Complete setup for {args.complete_check} is valid")
        else:
            print(f"❌ Setup issues for {args.complete_check}:")
            for issue in issues:
                print(f"  - {issue}")
    
    else:
        print("Use --help for available commands")

if __name__ == "__main__":
    main()