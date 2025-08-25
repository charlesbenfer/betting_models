#!/usr/bin/env python3
"""
Security Hardening Script
=========================

Automated security hardening for production deployment.
"""

import os
import sys
import stat
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from security.security_validator import SecurityAuditor, InputValidator

class SecurityHardener:
    """Automated security hardening system."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.hardening_log = []
        
    def run_security_hardening(self, directory: str = ".") -> Dict[str, Any]:
        """Run comprehensive security hardening."""
        print("🔒 Starting security hardening process...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'directory': directory,
            'dry_run': self.dry_run,
            'actions_taken': [],
            'issues_fixed': 0,
            'issues_remaining': 0
        }
        
        # Step 1: Fix file permissions
        print("📁 Hardening file permissions...")
        perm_results = self._harden_file_permissions(directory)
        results['actions_taken'].extend(perm_results)
        
        # Step 2: Secure configuration files
        print("⚙️  Securing configuration files...")
        config_results = self._secure_configuration_files(directory)
        results['actions_taken'].extend(config_results)
        
        # Step 3: Create security policy files
        print("📋 Creating security policy files...")
        policy_results = self._create_security_policies(directory)
        results['actions_taken'].extend(policy_results)
        
        # Step 4: Setup environment file template
        print("🔐 Setting up secure environment template...")
        env_results = self._setup_secure_environment_template(directory)
        results['actions_taken'].extend(env_results)
        
        # Count results
        results['issues_fixed'] = len([a for a in results['actions_taken'] if a['status'] == 'fixed'])
        results['issues_remaining'] = len([a for a in results['actions_taken'] if a['status'] == 'remaining'])
        
        return results
    
    def _harden_file_permissions(self, directory: str) -> List[Dict[str, Any]]:
        """Fix file permission issues."""
        actions = []
        
        # Files that should have restricted permissions
        sensitive_files = {
            '.env': 0o600,           # Owner read/write only
            'config.py': 0o600,      # Owner read/write only  
            '*.key': 0o600,          # Private keys
            '*.secret': 0o600,       # Secret files
            'secrets.json': 0o600,   # Secrets storage
        }
        
        # Directories that should be restricted
        sensitive_dirs = {
            'config/secrets': 0o700,  # Owner access only
            'logs': 0o750,           # Owner + group
            'backups': 0o750,        # Owner + group
        }
        
        try:
            # Fix sensitive files
            for pattern, target_mode in sensitive_files.items():
                if '*' in pattern:
                    # Handle wildcard patterns
                    for path in Path(directory).rglob(pattern):
                        self._fix_file_permission(str(path), target_mode, actions)
                else:
                    # Handle specific files
                    file_path = Path(directory) / pattern
                    if file_path.exists():
                        self._fix_file_permission(str(file_path), target_mode, actions)
            
            # Fix sensitive directories
            for dir_path, target_mode in sensitive_dirs.items():
                full_path = Path(directory) / dir_path
                if full_path.exists():
                    self._fix_file_permission(str(full_path), target_mode, actions)
            
            # Fix other overly permissive files in project root
            for item in Path(directory).iterdir():
                if item.is_file() and not str(item).startswith('.venv'):
                    current_mode = item.stat().st_mode & 0o777
                    
                    # Check if world writable (major security issue)
                    if current_mode & 0o002:
                        target_mode = current_mode & ~0o002  # Remove world write
                        self._fix_file_permission(str(item), target_mode, actions)
                    
                    # Check sensitive files that are world readable
                    if any(sensitive in item.name.lower() for sensitive in ['key', 'secret', 'password', 'config']):
                        if current_mode & 0o044:  # World or group readable
                            target_mode = 0o600  # Owner only
                            self._fix_file_permission(str(item), target_mode, actions)
        
        except Exception as e:
            actions.append({
                'type': 'file_permissions',
                'action': 'error',
                'description': f'Error fixing permissions: {e}',
                'status': 'error'
            })
        
        return actions
    
    def _fix_file_permission(self, file_path: str, target_mode: int, actions: List[Dict[str, Any]]):
        """Fix individual file permission."""
        try:
            current_mode = os.stat(file_path).st_mode & 0o777
            
            if current_mode != target_mode:
                action = {
                    'type': 'file_permissions',
                    'file': file_path,
                    'current_mode': oct(current_mode),
                    'target_mode': oct(target_mode),
                    'action': 'chmod',
                    'status': 'pending'
                }
                
                if not self.dry_run:
                    os.chmod(file_path, target_mode)
                    action['status'] = 'fixed'
                    print(f"  ✅ Fixed permissions: {file_path} -> {oct(target_mode)}")
                else:
                    action['status'] = 'would_fix'
                    print(f"  🔄 Would fix permissions: {file_path} -> {oct(target_mode)}")
                
                actions.append(action)
        
        except (OSError, PermissionError) as e:
            actions.append({
                'type': 'file_permissions',
                'file': file_path,
                'action': 'error',
                'description': str(e),
                'status': 'error'
            })
    
    def _secure_configuration_files(self, directory: str) -> List[Dict[str, Any]]:
        """Secure configuration files."""
        actions = []
        
        config_files = ['config.py', 'config/environment_config.py']
        
        for config_file in config_files:
            file_path = Path(directory) / config_file
            
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for hardcoded secrets
                    import re
                    secret_patterns = [
                        (r'api_key\s*=\s*[\'"][^\'"\s]{16,}[\'"]', 'API key'),
                        (r'password\s*=\s*[\'"][^\'"\s]{8,}[\'"]', 'Password'),
                        (r'secret\s*=\s*[\'"][^\'"\s]{16,}[\'"]', 'Secret'),
                    ]
                    
                    for pattern, secret_type in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            actions.append({
                                'type': 'configuration_security',
                                'file': str(file_path),
                                'issue': f'Hardcoded {secret_type.lower()} detected',
                                'recommendation': f'Move {secret_type.lower()} to environment variable or secrets manager',
                                'status': 'remaining'
                            })
                
                except Exception as e:
                    actions.append({
                        'type': 'configuration_security',
                        'file': str(file_path),
                        'action': 'error',
                        'description': str(e),
                        'status': 'error'
                    })
        
        return actions
    
    def _create_security_policies(self, directory: str) -> List[Dict[str, Any]]:
        """Create security policy files."""
        actions = []
        
        # .gitignore security entries
        gitignore_entries = [
            "# Security - Sensitive files",
            "*.key",
            "*.pem", 
            "*.p12",
            "*.secret",
            ".env.production",
            ".env.staging",
            "secrets.json",
            "config/secrets/",
            "backups/*.env",
            "*.log",
            "__pycache__/",
            "*.pyc"
        ]
        
        gitignore_path = Path(directory) / '.gitignore'
        
        try:
            # Read existing .gitignore
            existing_content = ""
            if gitignore_path.exists():
                with open(gitignore_path, 'r') as f:
                    existing_content = f.read()
            
            # Add missing security entries
            missing_entries = []
            for entry in gitignore_entries:
                if entry not in existing_content and not entry.startswith('#'):
                    missing_entries.append(entry)
            
            if missing_entries:
                action = {
                    'type': 'security_policy',
                    'file': str(gitignore_path),
                    'action': 'update_gitignore',
                    'entries_added': missing_entries,
                    'status': 'pending'
                }
                
                if not self.dry_run:
                    with open(gitignore_path, 'a') as f:
                        f.write('\n\n' + '\n'.join(missing_entries))
                    action['status'] = 'fixed'
                    print(f"  ✅ Updated .gitignore with security entries")
                else:
                    action['status'] = 'would_fix'
                    print(f"  🔄 Would update .gitignore with {len(missing_entries)} security entries")
                
                actions.append(action)
        
        except Exception as e:
            actions.append({
                'type': 'security_policy',
                'file': str(gitignore_path),
                'action': 'error',
                'description': str(e),
                'status': 'error'
            })
        
        # Security headers file
        security_headers_content = '''"""
Security Headers Configuration
==============================

Recommended security headers for production deployment.
"""

SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
}

def apply_security_headers(response):
    """Apply security headers to HTTP response."""
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    return response
'''
        
        headers_path = Path(directory) / 'security' / 'headers.py'
        
        if not headers_path.exists():
            action = {
                'type': 'security_policy',
                'file': str(headers_path),
                'action': 'create_security_headers',
                'status': 'pending'
            }
            
            if not self.dry_run:
                headers_path.parent.mkdir(exist_ok=True)
                with open(headers_path, 'w') as f:
                    f.write(security_headers_content)
                action['status'] = 'fixed'
                print(f"  ✅ Created security headers configuration")
            else:
                action['status'] = 'would_fix'
                print(f"  🔄 Would create security headers configuration")
            
            actions.append(action)
        
        return actions
    
    def _setup_secure_environment_template(self, directory: str) -> List[Dict[str, Any]]:
        """Setup secure environment file template."""
        actions = []
        
        env_template_content = '''# Baseball HR Prediction System - Environment Configuration
# =========================================================
# 
# SECURITY NOTICE: This is a template file. 
# Copy to .env and fill in actual values. Never commit .env to version control.

# Environment
BASEBALL_HR_ENV=development

# API Keys (Get from respective providers)
THEODDS_API_KEY=your_theodds_api_key_here
VISUALCROSSING_API_KEY=your_visualcrossing_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///data/baseball_hr.db
DATABASE_BACKUP_ENABLED=true

# Security
BASEBALL_HR_ENCRYPTION_KEY=generate_with_secrets_manager
SECURE_HEADERS_ENABLED=true
RATE_LIMITING_ENABLED=true

# Monitoring
MONITORING_ENABLED=true
LOG_LEVEL=INFO

# Email Alerts (Optional)
SMTP_SERVER=
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=
ALERT_EMAILS=

# Production Settings (Only for production environment)
# DEBUG=false
# SSL_ENABLED=true
# AUDIT_LOGGING_ENABLED=true
'''
        
        template_path = Path(directory) / '.env.template'
        
        if not template_path.exists():
            action = {
                'type': 'environment_security',
                'file': str(template_path),
                'action': 'create_env_template',
                'status': 'pending'
            }
            
            if not self.dry_run:
                with open(template_path, 'w') as f:
                    f.write(env_template_content)
                
                # Set secure permissions
                os.chmod(template_path, 0o644)  # Readable by all (it's a template)
                
                action['status'] = 'fixed'
                print(f"  ✅ Created secure environment template")
            else:
                action['status'] = 'would_fix'
                print(f"  🔄 Would create secure environment template")
            
            actions.append(action)
        
        # Check if .env exists and secure it
        env_path = Path(directory) / '.env'
        if env_path.exists():
            current_mode = env_path.stat().st_mode & 0o777
            if current_mode != 0o600:
                action = {
                    'type': 'environment_security',
                    'file': str(env_path),
                    'action': 'secure_env_file',
                    'current_mode': oct(current_mode),
                    'target_mode': '0o600',
                    'status': 'pending'
                }
                
                if not self.dry_run:
                    os.chmod(env_path, 0o600)
                    action['status'] = 'fixed'
                    print(f"  ✅ Secured .env file permissions")
                else:
                    action['status'] = 'would_fix'
                    print(f"  🔄 Would secure .env file permissions")
                
                actions.append(action)
        
        return actions
    
    def export_hardening_report(self, results: Dict[str, Any], output_dir: str = "logs"):
        """Export security hardening report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_file = output_path / f"security_hardening_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Summary report
        summary_file = output_path / f"security_hardening_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Security Hardening Report - {results['timestamp']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Directory: {results['directory']}\n")
            f.write(f"Dry Run: {results['dry_run']}\n")
            f.write(f"Issues Fixed: {results['issues_fixed']}\n")
            f.write(f"Issues Remaining: {results['issues_remaining']}\n\n")
            
            f.write("Actions Taken:\n")
            for action in results['actions_taken']:
                status_emoji = {
                    'fixed': '✅',
                    'would_fix': '🔄', 
                    'remaining': '⚠️',
                    'error': '❌'
                }.get(action['status'], '❓')
                
                f.write(f"  {status_emoji} {action['type']}: {action.get('description', action.get('action', 'Unknown'))}\n")
                if 'file' in action:
                    f.write(f"      File: {action['file']}\n")
        
        print(f"🔒 Security hardening report exported:")
        print(f"  📄 Full Report: {json_file}")
        print(f"  📋 Summary: {summary_file}")

def main():
    """Run security hardening."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Security Hardening Script')
    parser.add_argument('--directory', default='.', help='Directory to harden')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--export', action='store_true', help='Export results to files')
    
    args = parser.parse_args()
    
    hardener = SecurityHardener(dry_run=args.dry_run)
    results = hardener.run_security_hardening(args.directory)
    
    # Print summary
    print(f"\n🔒 SECURITY HARDENING SUMMARY")
    print("=" * 50)
    print(f"Issues Fixed: {results['issues_fixed']}")
    print(f"Issues Remaining: {results['issues_remaining']}")
    print(f"Dry Run: {results['dry_run']}")
    
    if results['issues_remaining'] > 0:
        print(f"\n⚠️  {results['issues_remaining']} issues require manual attention")
    
    if args.export:
        hardener.export_hardening_report(results)
    
    # Exit code based on results
    if results['issues_remaining'] > 0:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()