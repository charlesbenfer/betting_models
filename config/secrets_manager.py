#!/usr/bin/env python3
"""
Secrets Management System
========================

Secure handling of API keys, database credentials, and other sensitive data.
"""

import os
import json
import base64
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SecretEntry:
    """Individual secret entry."""
    name: str
    value: str
    environment: str
    encrypted: bool
    created_at: str
    last_accessed: Optional[str] = None
    description: Optional[str] = None

class SecretsManager:
    """Secure secrets management with encryption."""
    
    def __init__(self, secrets_dir: str = "config/secrets", encryption_key: Optional[str] = None):
        self.secrets_dir = Path(secrets_dir)
        self.secrets_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup encryption
        self.encryption_key = encryption_key or self._get_or_create_encryption_key()
        self.cipher = self._create_cipher()
        
        # Load secrets
        self.secrets = self._load_secrets()
        
        # Setup logging
        self.logger = logging.getLogger('secrets_manager')
        self.logger.info("Secrets manager initialized")
    
    def _get_or_create_encryption_key(self) -> str:
        """Get or create encryption key."""
        # Try environment variable first
        key = os.getenv('BASEBALL_HR_ENCRYPTION_KEY')
        if key:
            return key
        
        # Try key file
        key_file = self.secrets_dir / ".encryption_key"
        if key_file.exists():
            try:
                with open(key_file, 'r') as f:
                    return f.read().strip()
            except Exception:
                pass
        
        # Generate new key
        key = Fernet.generate_key().decode()
        
        # Save to file (with restricted permissions)
        try:
            key_file.touch(mode=0o600)
            with open(key_file, 'w') as f:
                f.write(key)
            
            print(f"⚠️  New encryption key generated and saved to {key_file}")
            print("🔑 Please backup this key securely - losing it means losing access to encrypted secrets")
            
        except Exception as e:
            print(f"⚠️  Could not save encryption key to file: {e}")
            print("🔑 Please set BASEBALL_HR_ENCRYPTION_KEY environment variable")
        
        return key
    
    def _create_cipher(self) -> Fernet:
        """Create encryption cipher."""
        try:
            # If key is already base64 encoded Fernet key
            return Fernet(self.encryption_key.encode())
        except Exception:
            # Derive key from password
            password = self.encryption_key.encode()
            salt = b'baseball_hr_salt'  # In production, use random salt per secret
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            return Fernet(key)
    
    def _load_secrets(self) -> Dict[str, SecretEntry]:
        """Load secrets from storage."""
        secrets = {}
        
        # Load from environment variables (highest priority)
        env_secrets = self._load_from_environment()
        secrets.update(env_secrets)
        
        # Load from encrypted file
        secrets_file = self.secrets_dir / "secrets.json"
        if secrets_file.exists():
            file_secrets = self._load_from_file(secrets_file)
            # Environment variables override file secrets
            for name, secret in file_secrets.items():
                if name not in secrets:
                    secrets[name] = secret
        
        return secrets
    
    def _load_from_environment(self) -> Dict[str, SecretEntry]:
        """Load secrets from environment variables."""
        secrets = {}
        
        # Known secret environment variables
        env_secrets = {
            'THEODDS_API_KEY': 'The Odds API key for betting odds',
            'VISUALCROSSING_API_KEY': 'Visual Crossing API key for weather data',
            'SMTP_PASSWORD': 'SMTP password for email alerts',
            'DATABASE_PASSWORD': 'Database password (if applicable)',
            'WEBHOOK_SECRET': 'Webhook secret for external integrations'
        }
        
        current_env = os.getenv('BASEBALL_HR_ENV', 'development')
        
        for env_var, description in env_secrets.items():
            value = os.getenv(env_var)
            if value:
                secrets[env_var.lower()] = SecretEntry(
                    name=env_var.lower(),
                    value=value,
                    environment=current_env,
                    encrypted=False,  # Environment variables are not encrypted in memory
                    created_at=datetime.now().isoformat(),
                    description=description
                )
        
        return secrets
    
    def _load_from_file(self, secrets_file: Path) -> Dict[str, SecretEntry]:
        """Load secrets from encrypted file."""
        try:
            with open(secrets_file, 'r') as f:
                encrypted_data = json.load(f)
            
            secrets = {}
            for name, entry_data in encrypted_data.items():
                try:
                    # Decrypt the value if it's encrypted
                    if entry_data['encrypted']:
                        encrypted_value = entry_data['value'].encode()
                        decrypted_value = self.cipher.decrypt(encrypted_value).decode()
                    else:
                        decrypted_value = entry_data['value']
                    
                    secrets[name] = SecretEntry(
                        name=entry_data['name'],
                        value=decrypted_value,
                        environment=entry_data['environment'],
                        encrypted=entry_data['encrypted'],
                        created_at=entry_data['created_at'],
                        last_accessed=entry_data.get('last_accessed'),
                        description=entry_data.get('description')
                    )
                except Exception as e:
                    self.logger.error(f"Failed to decrypt secret {name}: {e}")
                    continue
            
            return secrets
            
        except Exception as e:
            self.logger.error(f"Failed to load secrets file: {e}")
            return {}
    
    def get_secret(self, name: str, environment: Optional[str] = None) -> Optional[str]:
        """Get a secret value."""
        current_env = environment or os.getenv('BASEBALL_HR_ENV', 'development')
        
        # Try exact name first
        if name in self.secrets:
            secret = self.secrets[name]
            if secret.environment == current_env or environment is None:
                # Update last accessed time
                secret.last_accessed = datetime.now().isoformat()
                return secret.value
        
        # Try with environment suffix
        env_name = f"{name}_{current_env}"
        if env_name in self.secrets:
            secret = self.secrets[env_name]
            secret.last_accessed = datetime.now().isoformat()
            return secret.value
        
        return None
    
    def set_secret(self, name: str, value: str, environment: Optional[str] = None, 
                   description: Optional[str] = None, encrypt: bool = True):
        """Set a secret value."""
        current_env = environment or os.getenv('BASEBALL_HR_ENV', 'development')
        
        secret_entry = SecretEntry(
            name=name,
            value=value,
            environment=current_env,
            encrypted=encrypt,
            created_at=datetime.now().isoformat(),
            description=description
        )
        
        self.secrets[name] = secret_entry
        self.logger.info(f"Secret set: {name} for environment {current_env}")
    
    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        if name in self.secrets:
            del self.secrets[name]
            self.logger.info(f"Secret deleted: {name}")
            return True
        return False
    
    def list_secrets(self, environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all secrets (without values)."""
        current_env = environment or os.getenv('BASEBALL_HR_ENV', 'development')
        
        secrets_list = []
        for name, secret in self.secrets.items():
            if environment is None or secret.environment == current_env:
                secrets_list.append({
                    'name': secret.name,
                    'environment': secret.environment,
                    'encrypted': secret.encrypted,
                    'created_at': secret.created_at,
                    'last_accessed': secret.last_accessed,
                    'description': secret.description,
                    'value_length': len(secret.value) if secret.value else 0,
                    'value_preview': secret.value[:4] + '...' if secret.value and len(secret.value) > 4 else ''
                })
        
        return sorted(secrets_list, key=lambda x: x['name'])
    
    def save_secrets(self):
        """Save secrets to encrypted file."""
        secrets_file = self.secrets_dir / "secrets.json"
        
        # Filter out environment variable secrets (don't save to file)
        file_secrets = {}
        
        for name, secret in self.secrets.items():
            # Skip secrets that come from environment variables
            env_var_name = name.upper()
            if os.getenv(env_var_name) == secret.value:
                continue
            
            # Encrypt the value if requested
            if secret.encrypted:
                encrypted_value = self.cipher.encrypt(secret.value.encode()).decode()
            else:
                encrypted_value = secret.value
            
            file_secrets[name] = {
                'name': secret.name,
                'value': encrypted_value,
                'environment': secret.environment,
                'encrypted': secret.encrypted,
                'created_at': secret.created_at,
                'last_accessed': secret.last_accessed,
                'description': secret.description
            }
        
        try:
            # Set restrictive permissions
            secrets_file.touch(mode=0o600)
            
            with open(secrets_file, 'w') as f:
                json.dump(file_secrets, f, indent=2)
            
            self.logger.info(f"Secrets saved to {secrets_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save secrets: {e}")
    
    def validate_secrets(self, environment: Optional[str] = None) -> List[str]:
        """Validate that required secrets are available."""
        current_env = environment or os.getenv('BASEBALL_HR_ENV', 'development')
        issues = []
        
        # Required secrets by environment
        required_secrets = {
            'development': ['theodds_api_key'],
            'staging': ['theodds_api_key', 'visualcrossing_api_key'],
            'production': ['theodds_api_key', 'visualcrossing_api_key']
        }
        
        required = required_secrets.get(current_env, [])
        
        for secret_name in required:
            if not self.get_secret(secret_name, current_env):
                issues.append(f"Missing required secret: {secret_name}")
        
        # Check for weak secrets
        for name, secret in self.secrets.items():
            if secret.environment == current_env:
                if len(secret.value) < 8:
                    issues.append(f"Secret {name} is too short (< 8 characters)")
                
                # Check for common weak patterns
                if secret.value.lower() in ['password', 'secret', 'key', '12345']:
                    issues.append(f"Secret {name} uses a weak value")
        
        return issues
    
    def rotate_encryption_key(self, new_key: Optional[str] = None):
        """Rotate the encryption key and re-encrypt all secrets."""
        if new_key is None:
            new_key = Fernet.generate_key().decode()
        
        # Create new cipher
        old_cipher = self.cipher
        self.encryption_key = new_key
        self.cipher = self._create_cipher()
        
        # Re-encrypt all secrets
        for secret in self.secrets.values():
            if secret.encrypted:
                # Decrypt with old key, encrypt with new key
                try:
                    # The value is already decrypted in memory, so just mark for re-encryption
                    pass
                except Exception as e:
                    self.logger.error(f"Failed to rotate key for secret {secret.name}: {e}")
        
        # Save with new encryption
        self.save_secrets()
        
        # Update key file
        key_file = self.secrets_dir / ".encryption_key"
        try:
            with open(key_file, 'w') as f:
                f.write(new_key)
            self.logger.info("Encryption key rotated successfully")
        except Exception as e:
            self.logger.error(f"Failed to save new encryption key: {e}")
    
    def export_secrets_template(self, environment: str) -> str:
        """Export a template for secrets in an environment."""
        template_file = self.secrets_dir / f"secrets_template_{environment}.json"
        
        required_secrets = {
            'development': {
                'theodds_api_key': 'Your The Odds API key from theoddsapi.com',
                'visualcrossing_api_key': 'Optional: Visual Crossing weather API key'
            },
            'staging': {
                'theodds_api_key': 'The Odds API key for staging environment',
                'visualcrossing_api_key': 'Visual Crossing API key for staging',
                'smtp_password': 'SMTP password for email alerts'
            },
            'production': {
                'theodds_api_key': 'Production The Odds API key',
                'visualcrossing_api_key': 'Production Visual Crossing API key',
                'smtp_password': 'Production SMTP password for alerts',
                'webhook_secret': 'Secret for webhook authentication'
            }
        }
        
        template = {}
        for name, description in required_secrets.get(environment, {}).items():
            template[name] = {
                'value': 'REPLACE_WITH_ACTUAL_VALUE',
                'description': description,
                'encrypted': True,
                'environment': environment
            }
        
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        return str(template_file)

def main():
    """Secrets management CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Secrets Management')
    parser.add_argument('--set', nargs=3, metavar=('NAME', 'VALUE', 'DESCRIPTION'),
                       help='Set a secret')
    parser.add_argument('--get', help='Get a secret value')
    parser.add_argument('--delete', help='Delete a secret')
    parser.add_argument('--list', action='store_true', help='List all secrets')
    parser.add_argument('--validate', action='store_true', help='Validate secrets')
    parser.add_argument('--save', action='store_true', help='Save secrets to file')
    parser.add_argument('--rotate-key', action='store_true', help='Rotate encryption key')
    parser.add_argument('--export-template', help='Export secrets template for environment')
    parser.add_argument('--environment', help='Target environment')
    parser.add_argument('--no-encrypt', action='store_true', help='Do not encrypt secret')
    
    args = parser.parse_args()
    
    # Set environment if specified
    if args.environment:
        os.environ['BASEBALL_HR_ENV'] = args.environment
    
    # Create secrets manager
    try:
        secrets_manager = SecretsManager()
    except Exception as e:
        print(f"❌ Failed to initialize secrets manager: {e}")
        return
    
    if args.set:
        name, value, description = args.set
        encrypt = not args.no_encrypt
        secrets_manager.set_secret(name, value, description=description, encrypt=encrypt)
        print(f"✅ Secret set: {name}")
    
    elif args.get:
        value = secrets_manager.get_secret(args.get)
        if value:
            print(value)
        else:
            print(f"❌ Secret not found: {args.get}")
    
    elif args.delete:
        if secrets_manager.delete_secret(args.delete):
            print(f"✅ Secret deleted: {args.delete}")
        else:
            print(f"❌ Secret not found: {args.delete}")
    
    elif args.list:
        secrets_list = secrets_manager.list_secrets()
        if secrets_list:
            print("Secrets:")
            for secret in secrets_list:
                encrypted_flag = "🔒" if secret['encrypted'] else "🔓"
                print(f"  {encrypted_flag} {secret['name']} ({secret['environment']}) - {secret['description'] or 'No description'}")
        else:
            print("No secrets found")
    
    elif args.validate:
        issues = secrets_manager.validate_secrets()
        if issues:
            print("❌ Secret validation issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ All secrets are valid")
    
    elif args.save:
        secrets_manager.save_secrets()
        print("✅ Secrets saved")
    
    elif args.rotate_key:
        secrets_manager.rotate_encryption_key()
        print("✅ Encryption key rotated")
    
    elif args.export_template:
        template_file = secrets_manager.export_secrets_template(args.export_template)
        print(f"✅ Template exported: {template_file}")
    
    else:
        # Show status
        current_env = os.getenv('BASEBALL_HR_ENV', 'development')
        secrets_list = secrets_manager.list_secrets()
        issues = secrets_manager.validate_secrets()
        
        print(f"Environment: {current_env}")
        print(f"Secrets count: {len(secrets_list)}")
        print(f"Validation: {'✅ Valid' if not issues else '❌ Issues found'}")

if __name__ == "__main__":
    main()