#!/usr/bin/env python3
"""
Environment-Specific Configuration System
========================================

Manages configuration for different environments (development, staging, production).
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str
    backup_path: Optional[str] = None
    max_size_mb: int = 100
    auto_backup: bool = True

@dataclass
class APIConfig:
    """API configuration."""
    theodds_api_key: Optional[str] = None
    visualcrossing_api_key: Optional[str] = None
    base_url: str = "https://api.the-odds-api.com/v4"
    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 50

@dataclass
class ModelConfig:
    """Model configuration."""
    model_dir: str = "saved_models_pregame"
    auto_load: bool = True
    feature_cache_size: int = 10000
    prediction_batch_size: int = 1000
    model_validation: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enabled: bool = True
    interval_seconds: int = 60
    retention_hours: int = 24
    alert_thresholds: Dict[str, float] = None
    export_metrics: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "logs"
    max_file_size_mb: int = 10
    backup_count: int = 5
    console_output: bool = True

@dataclass
class SecurityConfig:
    """Security configuration."""
    encrypt_api_keys: bool = True
    secure_headers: bool = True
    rate_limiting: bool = True
    audit_logging: bool = True

@dataclass
class EnvironmentConfig:
    """Complete environment configuration."""
    environment: str
    debug: bool
    database: DatabaseConfig
    api: APIConfig
    model: ModelConfig
    monitoring: MonitoringConfig
    logging: LoggingConfig
    security: SecurityConfig
    custom_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_settings is None:
            self.custom_settings = {}
        if self.monitoring.alert_thresholds is None:
            self.monitoring.alert_thresholds = self._get_default_alert_thresholds()
    
    def _get_default_alert_thresholds(self) -> Dict[str, float]:
        """Get default alert thresholds based on environment."""
        if self.environment == "production":
            return {
                "cpu_percent": 80.0,
                "memory_percent": 85.0,
                "disk_percent": 90.0,
                "error_rate_per_minute": 5.0,
                "api_response_time_ms": 3000.0
            }
        elif self.environment == "staging":
            return {
                "cpu_percent": 85.0,
                "memory_percent": 90.0,
                "disk_percent": 95.0,
                "error_rate_per_minute": 10.0,
                "api_response_time_ms": 5000.0
            }
        else:  # development
            return {
                "cpu_percent": 95.0,
                "memory_percent": 95.0,
                "disk_percent": 98.0,
                "error_rate_per_minute": 50.0,
                "api_response_time_ms": 10000.0
            }

class ConfigurationManager:
    """Manages environment-specific configurations."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Environment detection
        self.current_environment = self._detect_environment()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(f"Configuration manager initialized for environment: {self.current_environment}")
    
    def _setup_logging(self):
        """Setup configuration-specific logging."""
        self.logger = logging.getLogger('config_manager')
        self.logger.setLevel(getattr(logging, self.config.logging.level))
        
        # Create handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _detect_environment(self) -> str:
        """Detect current environment."""
        # Check environment variable first
        env = os.getenv('BASEBALL_HR_ENV', '').lower()
        if env in ['development', 'staging', 'production']:
            return env
        
        # Check for environment indicators
        if os.getenv('PROD', '').lower() in ['true', '1', 'yes']:
            return 'production'
        elif os.getenv('STAGING', '').lower() in ['true', '1', 'yes']:
            return 'staging'
        
        # Default to development
        return 'development'
    
    def _load_configuration(self) -> EnvironmentConfig:
        """Load configuration for current environment."""
        config_file = self.config_dir / f"{self.current_environment}.json"
        
        if config_file.exists():
            return self._load_from_file(config_file)
        else:
            return self._create_default_configuration()
    
    def _load_from_file(self, config_file: Path) -> EnvironmentConfig:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Convert nested dictionaries to dataclasses
            return EnvironmentConfig(
                environment=config_data['environment'],
                debug=config_data['debug'],
                database=DatabaseConfig(**config_data['database']),
                api=APIConfig(**config_data['api']),
                model=ModelConfig(**config_data['model']),
                monitoring=MonitoringConfig(**config_data['monitoring']),
                logging=LoggingConfig(**config_data['logging']),
                security=SecurityConfig(**config_data['security']),
                custom_settings=config_data.get('custom_settings', {})
            )
            
        except Exception as e:
            print(f"Error loading config from {config_file}: {e}")
            return self._create_default_configuration()
    
    def _create_default_configuration(self) -> EnvironmentConfig:
        """Create default configuration for current environment."""
        if self.current_environment == "production":
            return self._create_production_config()
        elif self.current_environment == "staging":
            return self._create_staging_config()
        else:
            return self._create_development_config()
    
    def _create_development_config(self) -> EnvironmentConfig:
        """Create development configuration."""
        return EnvironmentConfig(
            environment="development",
            debug=True,
            database=DatabaseConfig(
                path="data/matchup_database.db",
                backup_path="data/backups",
                max_size_mb=50,
                auto_backup=False
            ),
            api=APIConfig(
                theodds_api_key=os.getenv("THEODDS_API_KEY"),
                visualcrossing_api_key=os.getenv("VISUALCROSSING_API_KEY"),
                timeout_seconds=60,
                max_retries=5,
                rate_limit_per_minute=30
            ),
            model=ModelConfig(
                model_dir="saved_models_pregame",
                auto_load=True,
                feature_cache_size=5000,
                prediction_batch_size=500,
                model_validation=False
            ),
            monitoring=MonitoringConfig(
                enabled=True,
                interval_seconds=120,
                retention_hours=12,
                export_metrics=False
            ),
            logging=LoggingConfig(
                level="DEBUG",
                log_dir="logs",
                max_file_size_mb=5,
                backup_count=3,
                console_output=True
            ),
            security=SecurityConfig(
                encrypt_api_keys=False,
                secure_headers=False,
                rate_limiting=False,
                audit_logging=False
            )
        )
    
    def _create_staging_config(self) -> EnvironmentConfig:
        """Create staging configuration."""
        return EnvironmentConfig(
            environment="staging",
            debug=False,
            database=DatabaseConfig(
                path="data/matchup_database.db",
                backup_path="data/backups",
                max_size_mb=75,
                auto_backup=True
            ),
            api=APIConfig(
                theodds_api_key=os.getenv("THEODDS_API_KEY"),
                visualcrossing_api_key=os.getenv("VISUALCROSSING_API_KEY"),
                timeout_seconds=45,
                max_retries=4,
                rate_limit_per_minute=40
            ),
            model=ModelConfig(
                model_dir="saved_models_pregame",
                auto_load=True,
                feature_cache_size=8000,
                prediction_batch_size=750,
                model_validation=True
            ),
            monitoring=MonitoringConfig(
                enabled=True,
                interval_seconds=90,
                retention_hours=24,
                export_metrics=True
            ),
            logging=LoggingConfig(
                level="INFO",
                log_dir="logs",
                max_file_size_mb=8,
                backup_count=4,
                console_output=True
            ),
            security=SecurityConfig(
                encrypt_api_keys=True,
                secure_headers=True,
                rate_limiting=True,
                audit_logging=True
            )
        )
    
    def _create_production_config(self) -> EnvironmentConfig:
        """Create production configuration."""
        return EnvironmentConfig(
            environment="production",
            debug=False,
            database=DatabaseConfig(
                path="data/matchup_database.db",
                backup_path="data/backups",
                max_size_mb=100,
                auto_backup=True
            ),
            api=APIConfig(
                theodds_api_key=os.getenv("THEODDS_API_KEY"),
                visualcrossing_api_key=os.getenv("VISUALCROSSING_API_KEY"),
                timeout_seconds=30,
                max_retries=3,
                rate_limit_per_minute=50
            ),
            model=ModelConfig(
                model_dir="saved_models_pregame",
                auto_load=True,
                feature_cache_size=10000,
                prediction_batch_size=1000,
                model_validation=True
            ),
            monitoring=MonitoringConfig(
                enabled=True,
                interval_seconds=60,
                retention_hours=48,
                export_metrics=True
            ),
            logging=LoggingConfig(
                level="INFO",
                log_dir="logs",
                max_file_size_mb=10,
                backup_count=5,
                console_output=False
            ),
            security=SecurityConfig(
                encrypt_api_keys=True,
                secure_headers=True,
                rate_limiting=True,
                audit_logging=True
            )
        )
    
    def save_configuration(self, config: Optional[EnvironmentConfig] = None):
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        config_file = self.config_dir / f"{config.environment}.json"
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Add metadata
        config_dict['_metadata'] = {
            'created_at': datetime.now().isoformat(),
            'created_by': 'ConfigurationManager',
            'version': '1.0'
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def validate_configuration(self, config: Optional[EnvironmentConfig] = None) -> List[str]:
        """Validate configuration and return list of issues."""
        if config is None:
            config = self.config
        
        issues = []
        
        # Validate API keys for production/staging
        if config.environment in ['production', 'staging']:
            if not config.api.theodds_api_key:
                issues.append("Missing required THEODDS_API_KEY for production/staging")
        
        # Validate paths
        try:
            Path(config.database.path).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Invalid database path: {e}")
        
        try:
            Path(config.model.model_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Invalid model directory: {e}")
        
        try:
            Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Invalid log directory: {e}")
        
        # Validate thresholds
        thresholds = config.monitoring.alert_thresholds
        if thresholds:
            if thresholds.get('cpu_percent', 0) > 100:
                issues.append("CPU threshold cannot exceed 100%")
            if thresholds.get('memory_percent', 0) > 100:
                issues.append("Memory threshold cannot exceed 100%")
        
        # Validate logging level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level not in valid_levels:
            issues.append(f"Invalid logging level: {config.logging.level}")
        
        return issues
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting using dot notation."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except Exception:
            return default
    
    def update_setting(self, key: str, value: Any):
        """Update a configuration setting using dot notation."""
        keys = key.split('.')
        config_obj = self.config
        
        # Navigate to the parent object
        for k in keys[:-1]:
            if hasattr(config_obj, k):
                config_obj = getattr(config_obj, k)
            else:
                raise ValueError(f"Invalid configuration path: {key}")
        
        # Set the final value
        final_key = keys[-1]
        if hasattr(config_obj, final_key):
            setattr(config_obj, final_key, value)
            self.logger.info(f"Updated configuration: {key} = {value}")
        else:
            raise ValueError(f"Invalid configuration key: {final_key}")
    
    def export_configuration(self, output_dir: str = "config/exports") -> str:
        """Export current configuration for backup/deployment."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = output_path / f"config_export_{self.current_environment}_{timestamp}.json"
        
        export_data = {
            'export_metadata': {
                'timestamp': datetime.now().isoformat(),
                'environment': self.current_environment,
                'exported_by': 'ConfigurationManager'
            },
            'configuration': asdict(self.config),
            'validation_results': self.validate_configuration()
        }
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Configuration exported to {export_file}")
        return str(export_file)
    
    def create_environment_template(self, environment: str) -> str:
        """Create configuration template for new environment."""
        if environment == "production":
            template_config = self._create_production_config()
        elif environment == "staging":
            template_config = self._create_staging_config()
        else:
            template_config = self._create_development_config()
        
        template_config.environment = environment
        
        template_file = self.config_dir / f"{environment}_template.json"
        
        with open(template_file, 'w') as f:
            json.dump(asdict(template_config), f, indent=2)
        
        self.logger.info(f"Template created: {template_file}")
        return str(template_file)

def main():
    """Configuration management CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Management')
    parser.add_argument('--environment', choices=['development', 'staging', 'production'],
                       help='Target environment')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    parser.add_argument('--save', action='store_true', help='Save current configuration')
    parser.add_argument('--export', action='store_true', help='Export configuration')
    parser.add_argument('--create-template', help='Create template for environment')
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--get', help='Get specific setting (dot notation)')
    parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), 
                       help='Set configuration value')
    
    args = parser.parse_args()
    
    # Set environment if specified
    if args.environment:
        os.environ['BASEBALL_HR_ENV'] = args.environment
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    if args.validate:
        issues = config_manager.validate_configuration()
        if issues:
            print("❌ Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Configuration is valid")
    
    elif args.save:
        config_manager.save_configuration()
        print(f"✅ Configuration saved for {config_manager.current_environment}")
    
    elif args.export:
        export_file = config_manager.export_configuration()
        print(f"✅ Configuration exported to {export_file}")
    
    elif args.create_template:
        template_file = config_manager.create_environment_template(args.create_template)
        print(f"✅ Template created: {template_file}")
    
    elif args.show:
        config_dict = asdict(config_manager.config)
        print(json.dumps(config_dict, indent=2))
    
    elif args.get:
        value = config_manager.get_setting(args.get)
        print(f"{args.get}: {value}")
    
    elif args.set:
        key, value = args.set
        try:
            # Try to parse as JSON for complex types
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass  # Keep as string
            
            config_manager.update_setting(key, value)
            print(f"✅ Updated {key} = {value}")
        except Exception as e:
            print(f"❌ Error updating setting: {e}")
    
    else:
        print(f"Current Environment: {config_manager.current_environment}")
        print(f"Configuration valid: {'✅' if not config_manager.validate_configuration() else '❌'}")

if __name__ == "__main__":
    main()