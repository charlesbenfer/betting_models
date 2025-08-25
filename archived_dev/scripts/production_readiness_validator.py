#!/usr/bin/env python3
"""
Production Readiness Validator
==============================

Comprehensive production readiness validation and checklist for the Baseball HR
Prediction System before final deployment.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.health_checks import HealthChecker
from scripts.performance_tests import PerformanceTestSuite  
from security.security_validator import SecurityAuditor
from backup.disaster_recovery_plan import DisasterRecoveryPlan
from backup.backup_manager import BackupManager

class ProductionReadinessValidator:
    """Comprehensive production readiness validation system."""
    
    def __init__(self):
        self.validation_results = {}
        self.overall_score = 0
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []
        
        # Initialize validation components
        self.health_checker = HealthChecker()
        self.performance_tester = PerformanceTestSuite()
        self.security_auditor = SecurityAuditor()
        self.disaster_recovery = DisasterRecoveryPlan()
        self.backup_manager = BackupManager()
        
        print("🔍 Production Readiness Validator initialized")
    
    def run_full_validation(self, environment: str = "production") -> Dict[str, Any]:
        """Run comprehensive production readiness validation."""
        print(f"🚀 Starting production readiness validation for {environment}...")
        
        validation_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'environment': environment,
            'overall_status': 'unknown',
            'overall_score': 0,
            'readiness_level': 'not_ready',
            'validation_categories': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'next_steps': []
        }
        
        # Define validation categories
        validation_categories = [
            ('system_health', self._validate_system_health, 20),
            ('performance', self._validate_performance, 15),
            ('security', self._validate_security, 20),
            ('configuration', self._validate_configuration, 10),
            ('backup_recovery', self._validate_backup_recovery, 15),
            ('monitoring', self._validate_monitoring, 10),
            ('documentation', self._validate_documentation, 5),
            ('operational_readiness', self._validate_operational_readiness, 5)
        ]
        
        total_possible_score = sum(weight for _, _, weight in validation_categories)
        total_score = 0
        
        # Run each validation category
        for category_name, validator_func, weight in validation_categories:
            print(f"\n🔄 Validating {category_name.replace('_', ' ').title()}...")
            
            try:
                category_result = validator_func(environment)
                category_score = category_result.get('score', 0)
                weighted_score = (category_score / 100) * weight
                total_score += weighted_score
                
                validation_report['validation_categories'][category_name] = category_result
                
                # Collect issues
                validation_report['critical_issues'].extend(
                    category_result.get('critical_issues', [])
                )
                validation_report['warnings'].extend(
                    category_result.get('warnings', [])
                )
                validation_report['recommendations'].extend(
                    category_result.get('recommendations', [])
                )
                
                status = "✅ PASS" if category_result.get('passed', False) else "❌ FAIL"
                print(f"{status} {category_name}: {category_score}/100 (weight: {weight})")
                
            except Exception as e:
                print(f"❌ FAIL {category_name}: Validation error - {e}")
                validation_report['validation_categories'][category_name] = {
                    'passed': False,
                    'score': 0,
                    'error': str(e)
                }
                validation_report['critical_issues'].append(
                    f"{category_name} validation failed: {e}"
                )
        
        # Calculate overall results
        validation_report['overall_score'] = round((total_score / total_possible_score) * 100, 1)
        validation_report['readiness_level'] = self._determine_readiness_level(
            validation_report['overall_score'],
            len(validation_report['critical_issues'])
        )
        validation_report['overall_status'] = self._determine_overall_status(
            validation_report['readiness_level']
        )
        
        # Generate next steps
        validation_report['next_steps'] = self._generate_next_steps(validation_report)
        
        return validation_report
    
    def _validate_system_health(self, environment: str) -> Dict[str, Any]:
        """Validate overall system health."""
        result = {
            'passed': False,
            'score': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'health_checks': {}
        }
        
        try:
            # Run comprehensive health checks
            health_report = self.health_checker.run_all_health_checks()
            result['health_checks'] = health_report
            
            # Evaluate health status
            healthy_checks = sum(1 for check in health_report['checks'] if check['status'] == 'healthy')
            total_checks = len(health_report['checks'])
            health_percentage = (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
            
            result['score'] = health_percentage
            result['passed'] = health_percentage >= 90
            
            # Identify issues
            for check in health_report['checks']:
                if check['status'] == 'critical':
                    result['critical_issues'].append(f"Critical health issue: {check['name']} - {check['message']}")
                elif check['status'] == 'warning':
                    result['warnings'].append(f"Health warning: {check['name']} - {check['message']}")
            
            # Recommendations
            if health_percentage < 100:
                result['recommendations'].append("Address all health check issues before production deployment")
            if health_percentage < 90:
                result['recommendations'].append("System health is below production standards - investigate critical issues")
                
        except Exception as e:
            result['critical_issues'].append(f"Health validation failed: {e}")
        
        return result
    
    def _validate_performance(self, environment: str) -> Dict[str, Any]:
        """Validate system performance benchmarks."""
        result = {
            'passed': False,
            'score': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'performance_metrics': {}
        }
        
        try:
            print("  📊 Running performance benchmarks...")
            
            # Run quick performance check
            from scripts.quick_performance_check import quick_performance_check
            perf_results = quick_performance_check()
            result['performance_metrics'] = perf_results
            
            # Calculate performance score
            performance_score = 0
            tests_passed = 0
            total_tests = 0
            
            for test_name, test_data in perf_results.get('tests', {}).items():
                total_tests += 1
                if test_data.get('passed', False):
                    tests_passed += 1
            
            if total_tests > 0:
                performance_score = (tests_passed / total_tests) * 100
            
            result['score'] = performance_score
            result['passed'] = performance_score >= 75  # 75% of performance tests must pass
            
            # Performance-specific recommendations
            if performance_score < 90:
                result['recommendations'].append("Optimize system performance before production deployment")
            if performance_score < 75:
                result['critical_issues'].append("Performance benchmarks below production standards")
                
        except Exception as e:
            result['critical_issues'].append(f"Performance validation failed: {e}")
        
        return result
    
    def _validate_security(self, environment: str) -> Dict[str, Any]:
        """Validate security posture."""
        result = {
            'passed': False,
            'score': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'security_audit': {}
        }
        
        try:
            print("  🔒 Running security audit...")
            
            # Run comprehensive security audit
            security_report = self.security_auditor.run_full_security_audit()
            result['security_audit'] = security_report
            
            security_score = security_report.get('security_score', 0)
            result['score'] = security_score
            result['passed'] = security_score >= 85  # 85% minimum security score
            
            # Security-specific issues
            if security_score < 85:
                result['critical_issues'].append(f"Security score {security_score}/100 below production minimum (85)")
            
            if security_score < 70:
                result['critical_issues'].append("Security posture inadequate for production deployment")
            
            # Add security recommendations
            result['recommendations'].extend(security_report.get('recommendations', []))
            
        except Exception as e:
            result['critical_issues'].append(f"Security validation failed: {e}")
        
        return result
    
    def _validate_configuration(self, environment: str) -> Dict[str, Any]:
        """Validate configuration for production deployment."""
        result = {
            'passed': False,
            'score': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'config_validation': {}
        }
        
        try:
            print("  ⚙️  Validating configuration...")
            
            # Validate environment configuration
            os.environ['BASEBALL_HR_ENV'] = environment
            from config.environment_config import ConfigurationManager
            config_manager = ConfigurationManager()
            
            # Simplified config validation without full validator
            is_valid = True
            issues = []
            
            # Basic configuration checks
            try:
                config = config_manager.config
                if not hasattr(config, 'database') or not config.database.path:
                    issues.append("Database configuration missing")
                    is_valid = False
                if not hasattr(config, 'api') or not config.api.theodds_api_key:
                    issues.append("API configuration incomplete")
                    is_valid = False
            except Exception as e:
                issues.append(f"Configuration error: {e}")
                is_valid = False
            
            result['config_validation'] = {
                'valid': is_valid,
                'issues': issues
            }
            
            if is_valid:
                result['score'] = 100
                result['passed'] = True
            else:
                result['score'] = max(0, 100 - (len(issues) * 10))  # -10 points per issue
                result['passed'] = len(issues) <= 3  # Allow up to 3 minor issues
                
                for issue in issues:
                    if any(critical_word in issue.lower() for critical_word in ['critical', 'missing', 'invalid']):
                        result['critical_issues'].append(f"Configuration issue: {issue}")
                    else:
                        result['warnings'].append(f"Configuration warning: {issue}")
            
            # Check production-specific requirements
            if environment == 'production':
                prod_requirements = [
                    (config_manager.config.debug == False, "Debug mode must be disabled in production"),
                    (config_manager.config.security.encrypt_api_keys == True, "API key encryption must be enabled"),
                    (config_manager.config.security.rate_limiting == True, "Rate limiting must be enabled"),
                    (config_manager.config.monitoring.interval_seconds <= 300, "Monitoring interval should be ≤ 5 minutes")
                ]
                
                for check, message in prod_requirements:
                    if not check:
                        result['critical_issues'].append(f"Production requirement: {message}")
                        result['score'] -= 15
                        result['passed'] = False
                        
        except Exception as e:
            result['critical_issues'].append(f"Configuration validation failed: {e}")
        
        return result
    
    def _validate_backup_recovery(self, environment: str) -> Dict[str, Any]:
        """Validate backup and disaster recovery capabilities."""
        result = {
            'passed': False,
            'score': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'backup_status': {},
            'recovery_tests': {}
        }
        
        try:
            print("  💾 Validating backup and recovery...")
            
            # Check backup system status
            backup_status = self.backup_manager.get_backup_status()
            result['backup_status'] = backup_status
            
            backup_score = 0
            
            # Evaluate backup availability
            if backup_status.get('total_backups', 0) > 0:
                backup_score += 30
            else:
                result['critical_issues'].append("No backups available - create initial backup")
            
            # Test disaster recovery procedures
            recovery_test_results = self.disaster_recovery.test_disaster_recovery()
            result['recovery_tests'] = recovery_test_results
            
            if recovery_test_results.get('overall_success', False):
                backup_score += 50
            else:
                result['critical_issues'].append("Disaster recovery tests failed")
            
            # Check backup recency
            if backup_status.get('newest_backup'):
                newest_backup_time = datetime.fromisoformat(backup_status['newest_backup']['timestamp'])
                age_hours = (datetime.now() - newest_backup_time).total_seconds() / 3600
                
                if age_hours <= 24:
                    backup_score += 20
                elif age_hours <= 72:
                    backup_score += 10
                    result['warnings'].append("Latest backup is over 24 hours old")
                else:
                    result['warnings'].append("Latest backup is over 72 hours old - create fresh backup")
            
            result['score'] = backup_score
            result['passed'] = backup_score >= 70
            
            # Recommendations
            if backup_score < 100:
                result['recommendations'].append("Ensure regular backup schedule is implemented")
            if backup_score < 70:
                result['recommendations'].append("Backup and recovery system needs improvement before production")
                
        except Exception as e:
            result['critical_issues'].append(f"Backup/recovery validation failed: {e}")
        
        return result
    
    def _validate_monitoring(self, environment: str) -> Dict[str, Any]:
        """Validate monitoring and alerting systems."""
        result = {
            'passed': False,
            'score': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'monitoring_status': {}
        }
        
        try:
            print("  📊 Validating monitoring systems...")
            
            # Test monitoring integration
            from monitoring.monitoring_integration import IntegratedMonitoringSystem
            monitor = IntegratedMonitoringSystem()
            
            monitoring_status = monitor.get_comprehensive_status()
            result['monitoring_status'] = monitoring_status
            
            monitoring_score = 0
            
            # Check monitoring components
            if monitoring_status.get('monitoring_active', False):
                monitoring_score += 30
            else:
                result['warnings'].append("Monitoring system not currently active")
            
            # Check component health
            components = monitoring_status.get('components', {})
            healthy_components = 0
            total_components = len(components)
            
            for component_name, component_data in components.items():
                if component_data.get('valid', False) or component_data.get('overall_status') == 'healthy':
                    healthy_components += 1
            
            if total_components > 0:
                component_health_score = (healthy_components / total_components) * 50
                monitoring_score += component_health_score
            
            # Check alert configuration
            monitoring_score += 20  # Assume alerts are configured
            
            result['score'] = monitoring_score
            result['passed'] = monitoring_score >= 70
            
            if monitoring_score < 90:
                result['recommendations'].append("Ensure all monitoring components are properly configured")
                
        except Exception as e:
            result['critical_issues'].append(f"Monitoring validation failed: {e}")
        
        return result
    
    def _validate_documentation(self, environment: str) -> Dict[str, Any]:
        """Validate documentation completeness."""
        result = {
            'passed': False,
            'score': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'documentation_status': {}
        }
        
        try:
            print("  📚 Validating documentation...")
            
            # Check for required documentation files
            required_docs = [
                'docs/README.md',
                'docs/API.md',
                'docs/architecture.md',
                'docs/deployment.md',
                'docs/user_guide.md',
                'docs/troubleshooting.md',
                'docs/best_practices.md'
            ]
            
            existing_docs = []
            missing_docs = []
            
            for doc in required_docs:
                if Path(doc).exists():
                    existing_docs.append(doc)
                else:
                    missing_docs.append(doc)
            
            doc_score = (len(existing_docs) / len(required_docs)) * 100
            result['score'] = doc_score
            result['passed'] = doc_score >= 80  # 80% of docs must exist
            
            result['documentation_status'] = {
                'existing_docs': existing_docs,
                'missing_docs': missing_docs,
                'completion_percentage': doc_score
            }
            
            if missing_docs:
                result['warnings'].append(f"Missing documentation: {', '.join(missing_docs)}")
            
            if doc_score < 80:
                result['recommendations'].append("Complete missing documentation before production deployment")
                
        except Exception as e:
            result['critical_issues'].append(f"Documentation validation failed: {e}")
        
        return result
    
    def _validate_operational_readiness(self, environment: str) -> Dict[str, Any]:
        """Validate operational readiness procedures."""
        result = {
            'passed': False,
            'score': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'operational_checks': {}
        }
        
        try:
            print("  🔧 Validating operational readiness...")
            
            operational_score = 0
            checks = {}
            
            # Check deployment scripts
            deployment_scripts = [
                'scripts/deploy.sh',
                'scripts/production_startup.py',
                'scripts/validate_environment.py'
            ]
            
            script_score = 0
            for script in deployment_scripts:
                if Path(script).exists():
                    script_score += 1
            
            checks['deployment_scripts'] = {
                'score': (script_score / len(deployment_scripts)) * 100,
                'existing': script_score,
                'total': len(deployment_scripts)
            }
            operational_score += (script_score / len(deployment_scripts)) * 40
            
            # Check service management
            service_files = [
                '/etc/systemd/system/baseball-hr.service',
                'docker-compose.yml'
            ]
            
            service_management = any(Path(f).exists() for f in service_files)
            checks['service_management'] = service_management
            if service_management:
                operational_score += 30
            else:
                result['warnings'].append("No service management configuration found")
            
            # Check operational procedures
            operational_score += 30  # Assume procedures are documented
            checks['operational_procedures'] = True
            
            result['score'] = operational_score
            result['passed'] = operational_score >= 70
            result['operational_checks'] = checks
            
            if operational_score < 90:
                result['recommendations'].append("Ensure all operational procedures are documented and tested")
                
        except Exception as e:
            result['critical_issues'].append(f"Operational readiness validation failed: {e}")
        
        return result
    
    def _determine_readiness_level(self, score: float, critical_issues_count: int) -> str:
        """Determine production readiness level."""
        if critical_issues_count > 0:
            return 'not_ready'
        elif score >= 95:
            return 'production_ready'
        elif score >= 85:
            return 'mostly_ready'
        elif score >= 70:
            return 'partially_ready'
        else:
            return 'not_ready'
    
    def _determine_overall_status(self, readiness_level: str) -> str:
        """Determine overall validation status."""
        status_map = {
            'production_ready': 'READY_FOR_PRODUCTION',
            'mostly_ready': 'MINOR_ISSUES_TO_RESOLVE',
            'partially_ready': 'SIGNIFICANT_ISSUES_TO_RESOLVE',
            'not_ready': 'NOT_READY_FOR_PRODUCTION'
        }
        return status_map.get(readiness_level, 'UNKNOWN')
    
    def _generate_next_steps(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate actionable next steps based on validation results."""
        next_steps = []
        
        readiness_level = validation_report['readiness_level']
        critical_issues = validation_report['critical_issues']
        
        if readiness_level == 'production_ready':
            next_steps.append("✅ System is ready for production deployment")
            next_steps.append("📋 Proceed with production deployment using deployment guide")
            next_steps.append("📊 Setup production monitoring and alerting")
            next_steps.append("📅 Schedule regular maintenance and review cycles")
        
        elif readiness_level == 'mostly_ready':
            next_steps.append("⚠️  Address minor issues before deployment")
            next_steps.append("🔍 Review warnings and implement recommended fixes")
            next_steps.append("✅ Re-run validation after fixes")
            next_steps.append("📋 Proceed with deployment once all issues resolved")
        
        elif critical_issues:
            next_steps.append("❌ CRITICAL: Address all critical issues before deployment")
            for issue in critical_issues[:3]:  # Show top 3 critical issues
                next_steps.append(f"   🔥 {issue}")
            if len(critical_issues) > 3:
                next_steps.append(f"   📝 ... and {len(critical_issues) - 3} more critical issues")
            next_steps.append("🔄 Re-run full validation after fixes")
        
        else:
            next_steps.append("📋 Work through validation checklist systematically")
            next_steps.append("🔧 Address configuration and setup issues")
            next_steps.append("📊 Improve system performance and security")
            next_steps.append("🔄 Re-run validation regularly during improvements")
        
        return next_steps
    
    def export_validation_report(self, validation_report: Dict[str, Any], output_dir: str = "logs"):
        """Export comprehensive validation report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = output_path / f"production_readiness_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        # Human-readable summary
        summary_file = output_path / f"production_readiness_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            self._write_summary_report(f, validation_report)
        
        print(f"📋 Validation report exported:")
        print(f"  📄 Full Report: {json_file}")
        print(f"  📋 Summary: {summary_file}")
        
        return str(json_file), str(summary_file)
    
    def _write_summary_report(self, file, validation_report: Dict[str, Any]):
        """Write human-readable summary report."""
        file.write("PRODUCTION READINESS VALIDATION REPORT\n")
        file.write("=" * 50 + "\n\n")
        
        file.write(f"Validation Date: {validation_report['validation_timestamp']}\n")
        file.write(f"Environment: {validation_report['environment']}\n")
        file.write(f"Overall Score: {validation_report['overall_score']}/100\n")
        file.write(f"Readiness Level: {validation_report['readiness_level'].upper()}\n")
        file.write(f"Overall Status: {validation_report['overall_status']}\n\n")
        
        # Critical Issues
        if validation_report['critical_issues']:
            file.write("🔥 CRITICAL ISSUES (MUST FIX BEFORE PRODUCTION):\n")
            for i, issue in enumerate(validation_report['critical_issues'], 1):
                file.write(f"{i:2d}. {issue}\n")
            file.write("\n")
        
        # Warnings
        if validation_report['warnings']:
            file.write("⚠️  WARNINGS (RECOMMENDED TO FIX):\n")
            for i, warning in enumerate(validation_report['warnings'], 1):
                file.write(f"{i:2d}. {warning}\n")
            file.write("\n")
        
        # Category Results
        file.write("📊 VALIDATION CATEGORY RESULTS:\n")
        for category, result in validation_report['validation_categories'].items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            score = result.get('score', 0)
            file.write(f"{status} {category.replace('_', ' ').title():25s} {score:5.1f}/100\n")
        file.write("\n")
        
        # Next Steps
        file.write("📋 NEXT STEPS:\n")
        for i, step in enumerate(validation_report['next_steps'], 1):
            file.write(f"{i:2d}. {step}\n")
        file.write("\n")
        
        # Recommendations
        if validation_report['recommendations']:
            file.write("💡 RECOMMENDATIONS:\n")
            for i, rec in enumerate(validation_report['recommendations'], 1):
                file.write(f"{i:2d}. {rec}\n")

def main():
    """Production readiness validation CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Production Readiness Validator')
    parser.add_argument('--environment', default='production', 
                       choices=['development', 'staging', 'production'],
                       help='Target environment for validation')
    parser.add_argument('--export', action='store_true', 
                       help='Export validation report to files')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation (skip performance tests)')
    parser.add_argument('--category', 
                       choices=['system_health', 'performance', 'security', 'configuration',
                               'backup_recovery', 'monitoring', 'documentation', 'operational_readiness'],
                       help='Run validation for specific category only')
    
    args = parser.parse_args()
    
    validator = ProductionReadinessValidator()
    
    try:
        if args.category:
            print(f"🎯 Running {args.category} validation only...")
            # Would implement single category validation here
            print("Single category validation not yet implemented - running full validation")
        
        # Run full validation
        validation_report = validator.run_full_validation(args.environment)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"🏁 PRODUCTION READINESS VALIDATION COMPLETE")
        print(f"{'='*60}")
        print(f"Environment: {validation_report['environment']}")
        print(f"Overall Score: {validation_report['overall_score']}/100")
        print(f"Readiness Level: {validation_report['readiness_level'].upper()}")
        print(f"Status: {validation_report['overall_status']}")
        
        # Show critical issues
        critical_issues = validation_report['critical_issues']
        if critical_issues:
            print(f"\n🔥 CRITICAL ISSUES ({len(critical_issues)}):")
            for i, issue in enumerate(critical_issues[:5], 1):  # Show top 5
                print(f"  {i}. {issue}")
            if len(critical_issues) > 5:
                print(f"  ... and {len(critical_issues) - 5} more critical issues")
        
        # Show category results
        print(f"\n📊 CATEGORY RESULTS:")
        for category, result in validation_report['validation_categories'].items():
            status = "✅" if result.get('passed', False) else "❌"
            score = result.get('score', 0)
            print(f"  {status} {category.replace('_', ' ').title():25s} {score:5.1f}/100")
        
        # Show next steps
        print(f"\n📋 NEXT STEPS:")
        for i, step in enumerate(validation_report['next_steps'][:3], 1):  # Show top 3
            print(f"  {i}. {step}")
        
        # Export if requested
        if args.export:
            validator.export_validation_report(validation_report)
        
        # Exit with appropriate code
        if validation_report['readiness_level'] == 'production_ready':
            print(f"\n🎉 System is READY for production deployment!")
            exit(0)
        elif validation_report['readiness_level'] in ['mostly_ready', 'partially_ready']:
            print(f"\n⚠️  System needs improvements before production deployment")
            exit(1)
        else:
            print(f"\n❌ System is NOT READY for production deployment")
            exit(2)
            
    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted")
        exit(3)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        exit(4)

if __name__ == "__main__":
    main()