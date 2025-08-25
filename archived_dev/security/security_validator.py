#!/usr/bin/env python3
"""
Security Validation and Hardening
==================================

Comprehensive security validation, input sanitization, and vulnerability scanning.
"""

import os
import re
import json
import hashlib
import secrets
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class InputValidator:
    """Secure input validation and sanitization."""
    
    # Regex patterns for common inputs
    PATTERNS = {
        'team_name': re.compile(r'^[A-Z]{2,4}$'),  # 2-4 uppercase letters
        'date': re.compile(r'^\d{4}-\d{2}-\d{2}$'),  # YYYY-MM-DD
        'player_name': re.compile(r'^[a-zA-Z\s\'\-\.]{1,50}$'),  # Names with common chars
        'numeric': re.compile(r'^-?\d*\.?\d+$'),  # Numbers (int/float)
        'alphanumeric': re.compile(r'^[a-zA-Z0-9_\-]{1,100}$'),  # Safe identifiers
        'api_key': re.compile(r'^[a-zA-Z0-9]{16,64}$'),  # API keys
        'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    }
    
    @classmethod
    def validate_team_name(cls, team_name: str) -> bool:
        """Validate MLB team abbreviation."""
        if not isinstance(team_name, str):
            return False
        
        valid_teams = {
            'LAA', 'HOU', 'OAK', 'SEA', 'TEX',  # AL West
            'CWS', 'CLE', 'DET', 'KC', 'MIN',   # AL Central  
            'BAL', 'BOS', 'NYY', 'TB', 'TOR',   # AL East
            'ATL', 'MIA', 'NYM', 'PHI', 'WSN',  # NL East
            'CHC', 'CIN', 'MIL', 'PIT', 'STL',  # NL Central
            'ARI', 'COL', 'LAD', 'SD', 'SF'     # NL West
        }
        
        return team_name.upper() in valid_teams
    
    @classmethod
    def validate_date(cls, date_str: str) -> bool:
        """Validate date string format and range."""
        if not isinstance(date_str, str) or not cls.PATTERNS['date'].match(date_str):
            return False
        
        try:
            from datetime import datetime
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Check reasonable date range (2020-2030)
            min_date = datetime(2020, 1, 1)
            max_date = datetime(2030, 12, 31)
            
            return min_date <= date_obj <= max_date
        except ValueError:
            return False
    
    @classmethod
    def validate_numeric_input(cls, value: Union[str, int, float], 
                             min_val: float = None, max_val: float = None) -> bool:
        """Validate numeric input with optional range checking."""
        try:
            if isinstance(value, str):
                if not cls.PATTERNS['numeric'].match(value):
                    return False
                num_val = float(value)
            else:
                num_val = float(value)
            
            if min_val is not None and num_val < min_val:
                return False
            if max_val is not None and num_val > max_val:
                return False
                
            return True
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def sanitize_string(cls, input_str: str, max_length: int = 100) -> str:
        """Sanitize string input to prevent injection attacks."""
        if not isinstance(input_str, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';\\&|`${}()]', '', input_str)
        
        # Limit length
        sanitized = sanitized[:max_length]
        
        # Remove excessive whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    @classmethod
    def validate_api_key(cls, api_key: str) -> bool:
        """Validate API key format."""
        if not isinstance(api_key, str):
            return False
        return cls.PATTERNS['api_key'].match(api_key) is not None
    
    @classmethod
    def validate_file_path(cls, file_path: str) -> bool:
        """Validate file path to prevent directory traversal."""
        if not isinstance(file_path, str):
            return False
        
        # Check for directory traversal attempts
        if '..' in file_path or file_path.startswith('/'):
            return False
        
        # Only allow certain file extensions
        allowed_extensions = {'.csv', '.json', '.txt', '.log', '.joblib', '.pkl'}
        path_obj = Path(file_path)
        
        if path_obj.suffix.lower() not in allowed_extensions:
            return False
        
        # Check path length
        if len(file_path) > 255:
            return False
        
        return True

class SecurityScanner:
    """Security vulnerability scanner."""
    
    def __init__(self):
        self.scan_results = {}
        
    def scan_file_permissions(self, directory: str = ".") -> Dict[str, Any]:
        """Scan file permissions for security issues."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'directory': directory,
            'issues': [],
            'secure_files': 0,
            'total_files': 0
        }
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    results['total_files'] += 1
                    
                    try:
                        stat_info = os.stat(file_path)
                        mode = stat_info.st_mode & 0o777
                        
                        # Check for overly permissive files
                        if mode & 0o002:  # World writable
                            results['issues'].append({
                                'type': 'world_writable',
                                'file': file_path,
                                'permissions': oct(mode)
                            })
                        elif mode & 0o004:  # World readable (sensitive files)
                            sensitive_patterns = ['.key', '.secret', '.env', 'config']
                            if any(pattern in file_path.lower() for pattern in sensitive_patterns):
                                results['issues'].append({
                                    'type': 'sensitive_world_readable',
                                    'file': file_path,
                                    'permissions': oct(mode)
                                })
                        else:
                            results['secure_files'] += 1
                            
                    except (OSError, PermissionError):
                        pass  # Skip files we can't read
                        
        except Exception as e:
            results['error'] = str(e)
        
        results['security_score'] = (results['secure_files'] / results['total_files'] * 100) if results['total_files'] > 0 else 100
        
        return results
    
    def scan_sensitive_data(self, directory: str = ".") -> Dict[str, Any]:
        """Scan for exposed sensitive data in files."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'directory': directory,
            'findings': [],
            'files_scanned': 0
        }
        
        # Patterns for sensitive data
        sensitive_patterns = {
            'api_key': re.compile(r'(?i)api[_\-]?key[\'"\s]*[:=][\'"\s]*[a-zA-Z0-9]{16,}'),
            'password': re.compile(r'(?i)password[\'"\s]*[:=][\'"\s]*[^\s\'"]{8,}'),
            'secret': re.compile(r'(?i)secret[\'"\s]*[:=][\'"\s]*[a-zA-Z0-9]{16,}'),
            'token': re.compile(r'(?i)token[\'"\s]*[:=][\'"\s]*[a-zA-Z0-9]{20,}'),
            'aws_key': re.compile(r'AKIA[0-9A-Z]{16}'),
            'private_key': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----')
        }
        
        try:
            for root, dirs, files in os.walk(directory):
                # Skip certain directories
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv'}]
                
                for file in files:
                    # Only scan text files
                    if not file.endswith(('.py', '.txt', '.json', '.yaml', '.yml', '.env', '.conf', '.cfg')):
                        continue
                    
                    file_path = os.path.join(root, file)
                    results['files_scanned'] += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        for pattern_name, pattern in sensitive_patterns.items():
                            matches = pattern.findall(content)
                            if matches:
                                results['findings'].append({
                                    'type': pattern_name,
                                    'file': file_path,
                                    'matches': len(matches),
                                    'sample': matches[0][:50] + '...' if matches[0] else ''
                                })
                    
                    except (UnicodeDecodeError, PermissionError, FileNotFoundError):
                        pass  # Skip files we can't read
                        
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan Python dependencies for known vulnerabilities."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'vulnerabilities': [],
            'safe_packages': 0,
            'total_packages': 0
        }
        
        try:
            # Try to use safety (if available)
            result = subprocess.run(['safety', 'check', '--json'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                try:
                    safety_output = json.loads(result.stdout)
                    results['vulnerabilities'] = safety_output
                    results['total_packages'] = len(safety_output)
                except json.JSONDecodeError:
                    results['error'] = 'Could not parse safety output'
            else:
                results['error'] = 'Safety check failed or not installed'
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback: manual check of common packages
            results.update(self._manual_dependency_check())
        
        return results
    
    def _manual_dependency_check(self) -> Dict[str, Any]:
        """Manual dependency vulnerability check."""
        try:
            import pkg_resources
            
            # Known vulnerable versions (example - would be updated from CVE databases)
            known_vulnerabilities = {
                'requests': ['2.25.0', '2.25.1'],  # Example vulnerable versions
                'urllib3': ['1.26.0', '1.26.1'],
                'jinja2': ['2.10.0', '2.10.1']
            }
            
            vulnerabilities = []
            safe_packages = 0
            total_packages = 0
            
            for package in pkg_resources.working_set:
                total_packages += 1
                package_name = package.project_name.lower()
                package_version = package.version
                
                if package_name in known_vulnerabilities:
                    if package_version in known_vulnerabilities[package_name]:
                        vulnerabilities.append({
                            'package': package_name,
                            'version': package_version,
                            'vulnerability': 'Known security issue'
                        })
                    else:
                        safe_packages += 1
                else:
                    safe_packages += 1
            
            return {
                'vulnerabilities': vulnerabilities,
                'safe_packages': safe_packages,
                'total_packages': total_packages
            }
            
        except Exception as e:
            return {'error': f'Manual dependency check failed: {e}'}

class RateLimiter:
    """Rate limiting for API endpoints and user actions."""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.request_history = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.now()
        
        # Initialize if first request from this identifier
        if identifier not in self.request_history:
            self.request_history[identifier] = []
        
        # Clean old requests outside time window
        cutoff_time = now - timedelta(seconds=self.time_window)
        self.request_history[identifier] = [
            timestamp for timestamp in self.request_history[identifier]
            if timestamp > cutoff_time
        ]
        
        # Check if under limit
        if len(self.request_history[identifier]) < self.max_requests:
            self.request_history[identifier].append(now)
            return True
        
        return False
    
    def get_reset_time(self, identifier: str) -> Optional[datetime]:
        """Get time when rate limit resets for identifier."""
        if identifier not in self.request_history or not self.request_history[identifier]:
            return None
        
        oldest_request = min(self.request_history[identifier])
        return oldest_request + timedelta(seconds=self.time_window)

class SecurityHeaders:
    """Security headers for web responses."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }

class SecurityAuditor:
    """Comprehensive security auditing system."""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.security_scanner = SecurityScanner()
        
    def run_full_security_audit(self, directory: str = ".") -> Dict[str, Any]:
        """Run comprehensive security audit."""
        print("🔒 Running comprehensive security audit...")
        
        audit_results = {
            'timestamp': datetime.now().isoformat(),
            'directory': directory,
            'overall_status': 'unknown',
            'security_score': 0,
            'audits': {}
        }
        
        # Run individual audits
        print("📁 Scanning file permissions...")
        audit_results['audits']['file_permissions'] = self.security_scanner.scan_file_permissions(directory)
        
        print("🔍 Scanning for sensitive data exposure...")
        audit_results['audits']['sensitive_data'] = self.security_scanner.scan_sensitive_data(directory)
        
        print("📦 Scanning dependencies for vulnerabilities...")
        audit_results['audits']['dependencies'] = self.security_scanner.scan_dependencies()
        
        print("⚙️  Validating configuration security...")
        audit_results['audits']['configuration'] = self._audit_configuration()
        
        # Calculate overall security score
        scores = []
        
        # File permissions score
        if 'security_score' in audit_results['audits']['file_permissions']:
            scores.append(audit_results['audits']['file_permissions']['security_score'])
        
        # Sensitive data score (penalize findings)
        sensitive_findings = len(audit_results['audits']['sensitive_data'].get('findings', []))
        sensitive_score = max(0, 100 - (sensitive_findings * 20))  # -20 points per finding
        scores.append(sensitive_score)
        
        # Dependencies score
        dep_vulns = len(audit_results['audits']['dependencies'].get('vulnerabilities', []))
        dep_score = max(0, 100 - (dep_vulns * 25))  # -25 points per vulnerability
        scores.append(dep_score)
        
        # Overall score
        audit_results['security_score'] = sum(scores) / len(scores) if scores else 0
        
        # Determine overall status
        if audit_results['security_score'] >= 90:
            audit_results['overall_status'] = 'excellent'
        elif audit_results['security_score'] >= 80:
            audit_results['overall_status'] = 'good'
        elif audit_results['security_score'] >= 70:
            audit_results['overall_status'] = 'acceptable'
        else:
            audit_results['overall_status'] = 'needs_improvement'
        
        # Generate recommendations
        audit_results['recommendations'] = self._generate_security_recommendations(audit_results)
        
        return audit_results
    
    def _audit_configuration(self) -> Dict[str, Any]:
        """Audit configuration security."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'issues': [],
            'secure_configs': 0,
            'total_configs': 0
        }
        
        config_files = ['config.py', 'config/environment_config.py', 'config/secrets_manager.py']
        
        for config_file in config_files:
            if os.path.exists(config_file):
                results['total_configs'] += 1
                
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                    
                    # Check for hardcoded secrets
                    if re.search(r'(?i)(password|secret|key)\s*=\s*[\'"][^\'"\s]{8,}[\'"]', content):
                        results['issues'].append({
                            'type': 'hardcoded_secret',
                            'file': config_file,
                            'description': 'Possible hardcoded secret found'
                        })
                    
                    # Check for debug mode in production configs
                    if 'production' in config_file.lower() and re.search(r'debug\s*=\s*true', content, re.IGNORECASE):
                        results['issues'].append({
                            'type': 'debug_in_production',
                            'file': config_file,
                            'description': 'Debug mode enabled in production config'
                        })
                    
                    if not results['issues']:
                        results['secure_configs'] += 1
                        
                except Exception as e:
                    results['issues'].append({
                        'type': 'read_error',
                        'file': config_file,
                        'description': f'Could not read file: {e}'
                    })
        
        return results
    
    def _generate_security_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate security improvement recommendations."""
        recommendations = []
        
        # File permission issues
        file_issues = audit_results['audits']['file_permissions'].get('issues', [])
        if file_issues:
            recommendations.append("Fix file permission issues - some files are overly permissive")
        
        # Sensitive data exposure
        sensitive_findings = audit_results['audits']['sensitive_data'].get('findings', [])
        if sensitive_findings:
            recommendations.append("Remove or secure exposed sensitive data in source files")
        
        # Dependency vulnerabilities
        dep_vulns = audit_results['audits']['dependencies'].get('vulnerabilities', [])
        if dep_vulns:
            recommendations.append("Update vulnerable dependencies to secure versions")
        
        # Configuration issues
        config_issues = audit_results['audits']['configuration'].get('issues', [])
        if config_issues:
            recommendations.append("Review configuration files for security issues")
        
        # General recommendations
        if audit_results['security_score'] < 90:
            recommendations.append("Implement additional security measures for production deployment")
        
        if not recommendations:
            recommendations.append("Security posture is good - maintain current security practices")
        
        return recommendations
    
    def export_audit_results(self, audit_results: Dict[str, Any], output_dir: str = "logs"):
        """Export security audit results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = output_path / f"security_audit_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(audit_results, f, indent=2)
        
        # Summary report
        summary_file = output_path / f"security_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Security Audit Summary - {audit_results['timestamp']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Overall Status: {audit_results['overall_status'].upper()}\n")
            f.write(f"Security Score: {audit_results['security_score']:.1f}/100\n\n")
            
            f.write("Key Findings:\n")
            
            # File permission issues
            file_issues = audit_results['audits']['file_permissions'].get('issues', [])
            f.write(f"  File Permission Issues: {len(file_issues)}\n")
            
            # Sensitive data findings
            sensitive_findings = audit_results['audits']['sensitive_data'].get('findings', [])
            f.write(f"  Sensitive Data Exposures: {len(sensitive_findings)}\n")
            
            # Dependency vulnerabilities
            dep_vulns = audit_results['audits']['dependencies'].get('vulnerabilities', [])
            f.write(f"  Dependency Vulnerabilities: {len(dep_vulns)}\n")
            
            f.write("\nRecommendations:\n")
            for rec in audit_results['recommendations']:
                f.write(f"  - {rec}\n")
        
        print(f"🔒 Security audit results exported:")
        print(f"  📄 Full Report: {json_file}")
        print(f"  📋 Summary: {summary_file}")

def main():
    """Run security audit."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Security Validation and Hardening')
    parser.add_argument('--directory', default='.', help='Directory to audit')
    parser.add_argument('--audit', choices=['all', 'permissions', 'sensitive', 'deps'], 
                       default='all', help='Audit type to run')
    parser.add_argument('--export', action='store_true', help='Export results to files')
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor()
    
    if args.audit == 'all':
        results = auditor.run_full_security_audit(args.directory)
        
        # Print summary
        print(f"\n🔒 SECURITY AUDIT SUMMARY")
        print("=" * 50)
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Security Score: {results['security_score']:.1f}/100")
        
        if results['recommendations']:
            print(f"\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  💡 {rec}")
        
        if args.export:
            auditor.export_audit_results(results)
        
        # Exit code based on security score
        if results['security_score'] >= 90:
            exit(0)
        elif results['security_score'] >= 70:
            exit(1)
        else:
            exit(2)
    
    else:
        # Run specific audit type
        scanner = SecurityScanner()
        
        if args.audit == 'permissions':
            results = scanner.scan_file_permissions(args.directory)
        elif args.audit == 'sensitive':
            results = scanner.scan_sensitive_data(args.directory)
        elif args.audit == 'deps':
            results = scanner.scan_dependencies()
        
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()