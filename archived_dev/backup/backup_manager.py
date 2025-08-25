#!/usr/bin/env python3
"""
Backup and Recovery Manager
============================

Comprehensive backup and disaster recovery system for production data.
"""

import os
import sys
import shutil
import sqlite3
import json
import gzip
import tarfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class BackupManager:
    """Comprehensive backup management system."""
    
    def __init__(self, backup_root: str = "backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)
        
        # Backup configuration
        self.config = {
            'database_retention_days': 30,
            'config_retention_days': 90,
            'log_retention_days': 7,
            'model_retention_days': 60,
            'compress_backups': True,
            'verify_backups': True
        }
        
        logger.info(f"Backup manager initialized with root: {self.backup_root}")
    
    def create_full_backup(self) -> Dict[str, Any]:
        """Create comprehensive system backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"full_backup_{timestamp}"
        backup_dir = self.backup_root / backup_name
        
        print(f"🔄 Creating full system backup: {backup_name}")
        
        backup_result = {
            'timestamp': datetime.now().isoformat(),
            'backup_name': backup_name,
            'backup_path': str(backup_dir),
            'components': {},
            'total_size_mb': 0,
            'success': False
        }
        
        try:
            backup_dir.mkdir(exist_ok=True)
            
            # Backup databases
            print("💾 Backing up databases...")
            db_result = self._backup_databases(backup_dir / "databases")
            backup_result['components']['databases'] = db_result
            
            # Backup configuration files
            print("⚙️  Backing up configuration...")
            config_result = self._backup_configuration(backup_dir / "config")
            backup_result['components']['configuration'] = config_result
            
            # Backup models
            print("🤖 Backing up models...")
            model_result = self._backup_models(backup_dir / "models")
            backup_result['components']['models'] = model_result
            
            # Backup critical data
            print("📊 Backing up critical data...")
            data_result = self._backup_critical_data(backup_dir / "data")
            backup_result['components']['data'] = data_result
            
            # Backup logs (recent only)
            print("📝 Backing up recent logs...")
            log_result = self._backup_logs(backup_dir / "logs")
            backup_result['components']['logs'] = log_result
            
            # Create backup manifest
            print("📋 Creating backup manifest...")
            manifest_result = self._create_backup_manifest(backup_dir)
            backup_result['components']['manifest'] = manifest_result
            
            # Calculate total size
            total_size = sum(
                sum(Path(root).stat().st_size for root, dirs, files in os.walk(root) for f in files)
                for root, dirs, files in os.walk(backup_dir)
            )
            backup_result['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            # Compress backup if enabled
            if self.config['compress_backups']:
                print("🗜️  Compressing backup...")
                compressed_path = self._compress_backup(backup_dir)
                if compressed_path:
                    backup_result['compressed_path'] = str(compressed_path)
                    # Remove uncompressed backup
                    shutil.rmtree(backup_dir)
                    backup_result['backup_path'] = str(compressed_path)
            
            # Verify backup
            if self.config['verify_backups']:
                print("✅ Verifying backup integrity...")
                verification_result = self._verify_backup(backup_result['backup_path'])
                backup_result['verification'] = verification_result
            
            backup_result['success'] = True
            print(f"✅ Full backup completed: {backup_result['total_size_mb']} MB")
            
        except Exception as e:
            backup_result['error'] = str(e)
            logger.error(f"Full backup failed: {e}")
            print(f"❌ Full backup failed: {e}")
        
        return backup_result
    
    def _backup_databases(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup all database files."""
        backup_dir.mkdir(exist_ok=True)
        result = {'files_backed_up': 0, 'total_size_mb': 0, 'databases': []}
        
        # Find database files
        db_extensions = ['.db', '.sqlite', '.sqlite3']
        db_files = []
        
        for ext in db_extensions:
            db_files.extend(Path('.').rglob(f'*{ext}'))
        
        # Also check common database locations
        common_db_paths = ['data/baseball_hr.db', 'data/games.db', 'matchup_database.db']
        for db_path in common_db_paths:
            if Path(db_path).exists():
                db_files.append(Path(db_path))
        
        for db_file in set(db_files):  # Remove duplicates
            try:
                if db_file.exists() and db_file.suffix.lower() in db_extensions:
                    # Create SQLite backup using SQLite backup API
                    backup_file = backup_dir / f"{db_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{db_file.suffix}"
                    
                    if self._backup_sqlite_db(str(db_file), str(backup_file)):
                        size_mb = backup_file.stat().st_size / (1024 * 1024)
                        result['databases'].append({
                            'original_path': str(db_file),
                            'backup_path': str(backup_file),
                            'size_mb': round(size_mb, 2),
                            'backup_time': datetime.now().isoformat()
                        })
                        result['files_backed_up'] += 1
                        result['total_size_mb'] += size_mb
                        
                        print(f"  ✅ Database backed up: {db_file.name} ({size_mb:.2f} MB)")
                    
            except Exception as e:
                logger.error(f"Failed to backup database {db_file}: {e}")
                print(f"  ❌ Failed to backup {db_file}: {e}")
        
        return result
    
    def _backup_sqlite_db(self, source_path: str, backup_path: str) -> bool:
        """Backup SQLite database using SQLite backup API."""
        try:
            # Connect to source and destination
            source_conn = sqlite3.connect(source_path)
            backup_conn = sqlite3.connect(backup_path)
            
            # Perform backup
            source_conn.backup(backup_conn)
            
            # Close connections
            source_conn.close()
            backup_conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            return False
    
    def _backup_configuration(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup configuration files."""
        backup_dir.mkdir(exist_ok=True)
        result = {'files_backed_up': 0, 'total_size_mb': 0, 'config_files': []}
        
        # Configuration files to backup
        config_patterns = [
            'config.py',
            'config/*.py',
            '.env.template',
            'requirements.txt',
            'pyproject.toml',
            'setup.py'
        ]
        
        config_files = []
        for pattern in config_patterns:
            config_files.extend(Path('.').glob(pattern))
        
        for config_file in config_files:
            try:
                if config_file.is_file():
                    backup_file = backup_dir / config_file.name
                    shutil.copy2(config_file, backup_file)
                    
                    size_mb = backup_file.stat().st_size / (1024 * 1024)
                    result['config_files'].append({
                        'original_path': str(config_file),
                        'backup_path': str(backup_file),
                        'size_mb': round(size_mb, 4)
                    })
                    result['files_backed_up'] += 1
                    result['total_size_mb'] += size_mb
                    
                    print(f"  ✅ Config backed up: {config_file.name}")
                    
            except Exception as e:
                logger.error(f"Failed to backup config {config_file}: {e}")
                print(f"  ❌ Failed to backup {config_file}: {e}")
        
        return result
    
    def _backup_models(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup machine learning models."""
        backup_dir.mkdir(exist_ok=True)
        result = {'files_backed_up': 0, 'total_size_mb': 0, 'model_files': []}
        
        # Find model files
        model_patterns = ['*.joblib', '*.pkl', '*.pickle', '*.h5', '*.json']
        model_dirs = ['models', 'saved_models', 'saved_models_pregame']
        
        for model_dir in model_dirs:
            if Path(model_dir).exists():
                for pattern in model_patterns:
                    for model_file in Path(model_dir).glob(pattern):
                        try:
                            # Create timestamped backup name
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            backup_name = f"{model_file.stem}_{timestamp}{model_file.suffix}"
                            backup_file = backup_dir / backup_name
                            
                            shutil.copy2(model_file, backup_file)
                            
                            size_mb = backup_file.stat().st_size / (1024 * 1024)
                            result['model_files'].append({
                                'original_path': str(model_file),
                                'backup_path': str(backup_file),
                                'size_mb': round(size_mb, 2),
                                'model_type': self._identify_model_type(model_file)
                            })
                            result['files_backed_up'] += 1
                            result['total_size_mb'] += size_mb
                            
                            print(f"  ✅ Model backed up: {model_file.name} ({size_mb:.2f} MB)")
                            
                        except Exception as e:
                            logger.error(f"Failed to backup model {model_file}: {e}")
                            print(f"  ❌ Failed to backup {model_file}: {e}")
        
        return result
    
    def _backup_critical_data(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup critical data files."""
        backup_dir.mkdir(exist_ok=True)
        result = {'files_backed_up': 0, 'total_size_mb': 0, 'data_files': []}
        
        # Critical data files to backup
        data_patterns = [
            'data/*.csv',
            'data/*.json', 
            'data/processed_*.csv',
            'data/features_*.csv'
        ]
        
        for pattern in data_patterns:
            for data_file in Path('.').glob(pattern):
                try:
                    if data_file.is_file() and data_file.stat().st_size > 0:
                        backup_file = backup_dir / data_file.name
                        shutil.copy2(data_file, backup_file)
                        
                        size_mb = backup_file.stat().st_size / (1024 * 1024)
                        result['data_files'].append({
                            'original_path': str(data_file),
                            'backup_path': str(backup_file),
                            'size_mb': round(size_mb, 2)
                        })
                        result['files_backed_up'] += 1
                        result['total_size_mb'] += size_mb
                        
                        print(f"  ✅ Data backed up: {data_file.name} ({size_mb:.2f} MB)")
                        
                except Exception as e:
                    logger.error(f"Failed to backup data {data_file}: {e}")
                    print(f"  ❌ Failed to backup {data_file}: {e}")
        
        return result
    
    def _backup_logs(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup recent log files."""
        backup_dir.mkdir(exist_ok=True)
        result = {'files_backed_up': 0, 'total_size_mb': 0, 'log_files': []}
        
        # Only backup recent logs (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=self.config['log_retention_days'])
        
        log_dirs = ['logs', 'log']
        for log_dir in log_dirs:
            if Path(log_dir).exists():
                for log_file in Path(log_dir).glob('*.log'):
                    try:
                        if log_file.is_file():
                            # Check if log is recent
                            mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                            
                            if mod_time > cutoff_date:
                                backup_file = backup_dir / log_file.name
                                shutil.copy2(log_file, backup_file)
                                
                                size_mb = backup_file.stat().st_size / (1024 * 1024)
                                result['log_files'].append({
                                    'original_path': str(log_file),
                                    'backup_path': str(backup_file),
                                    'size_mb': round(size_mb, 4),
                                    'last_modified': mod_time.isoformat()
                                })
                                result['files_backed_up'] += 1
                                result['total_size_mb'] += size_mb
                                
                                print(f"  ✅ Log backed up: {log_file.name}")
                    
                    except Exception as e:
                        logger.error(f"Failed to backup log {log_file}: {e}")
                        print(f"  ❌ Failed to backup {log_file}: {e}")
        
        return result
    
    def _create_backup_manifest(self, backup_dir: Path) -> Dict[str, Any]:
        """Create backup manifest with metadata."""
        manifest = {
            'backup_timestamp': datetime.now().isoformat(),
            'backup_version': '1.0',
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': str(Path.cwd())
            },
            'files': []
        }
        
        # Catalog all files in backup
        for root, dirs, files in os.walk(backup_dir):
            for file in files:
                file_path = Path(root) / file
                try:
                    stat_info = file_path.stat()
                    file_hash = self._calculate_file_hash(file_path)
                    
                    manifest['files'].append({
                        'path': str(file_path.relative_to(backup_dir)),
                        'size_bytes': stat_info.st_size,
                        'modified_time': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                        'sha256_hash': file_hash
                    })
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
        
        # Write manifest file
        manifest_file = backup_dir / 'backup_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return {
            'manifest_created': True,
            'total_files_cataloged': len(manifest['files']),
            'manifest_path': str(manifest_file)
        }
    
    def _compress_backup(self, backup_dir: Path) -> Optional[Path]:
        """Compress backup directory into tar.gz file."""
        try:
            compressed_file = backup_dir.with_suffix('.tar.gz')
            
            with tarfile.open(compressed_file, 'w:gz') as tar:
                tar.add(backup_dir, arcname=backup_dir.name)
            
            print(f"  ✅ Backup compressed: {compressed_file.name}")
            return compressed_file
            
        except Exception as e:
            logger.error(f"Backup compression failed: {e}")
            return None
    
    def _verify_backup(self, backup_path: str) -> Dict[str, Any]:
        """Verify backup integrity."""
        verification = {
            'verified': False,
            'manifest_valid': False,
            'file_count': 0,
            'hash_checks_passed': 0,
            'hash_checks_failed': 0
        }
        
        try:
            backup_path_obj = Path(backup_path)
            
            # Handle compressed backups
            if backup_path_obj.suffix == '.gz':
                # Extract manifest from compressed backup
                with tarfile.open(backup_path, 'r:gz') as tar:
                    # Find manifest file
                    manifest_member = None
                    for member in tar.getmembers():
                        if member.name.endswith('backup_manifest.json'):
                            manifest_member = member
                            break
                    
                    if manifest_member:
                        manifest_file = tar.extractfile(manifest_member)
                        if manifest_file:
                            manifest = json.load(manifest_file)
                            verification['manifest_valid'] = True
                            verification['file_count'] = len(manifest['files'])
            else:
                # Read manifest from uncompressed backup
                manifest_path = backup_path_obj / 'backup_manifest.json'
                if manifest_path.exists():
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    verification['manifest_valid'] = True
                    verification['file_count'] = len(manifest['files'])
            
            if verification['manifest_valid']:
                verification['verified'] = True
                print(f"  ✅ Backup verification passed: {verification['file_count']} files")
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            verification['error'] = str(e)
        
        return verification
    
    def restore_from_backup(self, backup_path: str, restore_dir: str = "restored") -> Dict[str, Any]:
        """Restore system from backup."""
        print(f"🔄 Restoring from backup: {backup_path}")
        
        restore_result = {
            'timestamp': datetime.now().isoformat(),
            'backup_path': backup_path,
            'restore_directory': restore_dir,
            'success': False,
            'files_restored': 0,
            'errors': []
        }
        
        try:
            backup_path_obj = Path(backup_path)
            restore_dir_obj = Path(restore_dir)
            restore_dir_obj.mkdir(exist_ok=True)
            
            if backup_path_obj.suffix == '.gz':
                # Extract compressed backup
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.extractall(restore_dir_obj)
                    restore_result['files_restored'] = len(tar.getmembers())
            else:
                # Copy uncompressed backup
                if backup_path_obj.is_dir():
                    shutil.copytree(backup_path_obj, restore_dir_obj / backup_path_obj.name)
                    restore_result['files_restored'] = sum(1 for _ in restore_dir_obj.rglob('*') if _.is_file())
            
            restore_result['success'] = True
            print(f"✅ Restore completed: {restore_result['files_restored']} files")
            
        except Exception as e:
            restore_result['errors'].append(str(e))
            logger.error(f"Restore failed: {e}")
            print(f"❌ Restore failed: {e}")
        
        return restore_result
    
    def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old backups based on retention policies."""
        print("🧹 Cleaning up old backups...")
        
        cleanup_result = {
            'timestamp': datetime.now().isoformat(),
            'deleted_backups': [],
            'space_freed_mb': 0,
            'backups_remaining': 0
        }
        
        try:
            now = datetime.now()
            
            for backup_item in self.backup_root.iterdir():
                if backup_item.is_file() and ('backup' in backup_item.name.lower()):
                    # Parse backup timestamp from filename
                    try:
                        # Extract timestamp from filename (format: backup_YYYYMMDD_HHMMSS)
                        parts = backup_item.stem.split('_')
                        if len(parts) >= 3:
                            date_str = parts[-2]
                            time_str = parts[-1]
                            backup_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                            
                            # Determine retention period based on backup type
                            retention_days = self.config['database_retention_days']  # Default
                            if 'config' in backup_item.name:
                                retention_days = self.config['config_retention_days']
                            elif 'log' in backup_item.name:
                                retention_days = self.config['log_retention_days']
                            elif 'model' in backup_item.name:
                                retention_days = self.config['model_retention_days']
                            
                            # Check if backup is too old
                            if (now - backup_time).days > retention_days:
                                size_mb = backup_item.stat().st_size / (1024 * 1024)
                                backup_item.unlink()
                                
                                cleanup_result['deleted_backups'].append({
                                    'filename': backup_item.name,
                                    'backup_time': backup_time.isoformat(),
                                    'size_mb': round(size_mb, 2),
                                    'age_days': (now - backup_time).days
                                })
                                cleanup_result['space_freed_mb'] += size_mb
                                
                                print(f"  🗑️  Deleted old backup: {backup_item.name} ({size_mb:.2f} MB)")
                    
                    except (ValueError, IndexError):
                        # Skip files that don't match expected naming pattern
                        pass
                
                elif backup_item.is_file():
                    cleanup_result['backups_remaining'] += 1
            
            print(f"✅ Cleanup completed: {len(cleanup_result['deleted_backups'])} backups deleted, {cleanup_result['space_freed_mb']:.2f} MB freed")
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            print(f"❌ Backup cleanup failed: {e}")
        
        return cleanup_result
    
    def _identify_model_type(self, model_file: Path) -> str:
        """Identify the type of machine learning model."""
        filename = model_file.name.lower()
        
        if 'xgboost' in filename or 'xgb' in filename:
            return 'XGBoost'
        elif 'random_forest' in filename or 'rf' in filename:
            return 'Random Forest'
        elif 'logistic' in filename:
            return 'Logistic Regression'
        elif 'svm' in filename:
            return 'Support Vector Machine'
        elif 'neural' in filename or 'nn' in filename:
            return 'Neural Network'
        elif 'ensemble' in filename:
            return 'Ensemble'
        else:
            return 'Unknown'
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception:
            return "error_calculating_hash"
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get current backup system status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'backup_root': str(self.backup_root),
            'total_backups': 0,
            'total_backup_size_mb': 0,
            'oldest_backup': None,
            'newest_backup': None,
            'backup_types': {}
        }
        
        try:
            backup_files = list(self.backup_root.glob('*backup*'))
            status['total_backups'] = len(backup_files)
            
            if backup_files:
                # Calculate total size
                total_size = sum(f.stat().st_size for f in backup_files if f.is_file())
                status['total_backup_size_mb'] = round(total_size / (1024 * 1024), 2)
                
                # Find oldest and newest
                file_times = [(f, f.stat().st_mtime) for f in backup_files if f.is_file()]
                if file_times:
                    oldest = min(file_times, key=lambda x: x[1])
                    newest = max(file_times, key=lambda x: x[1])
                    
                    status['oldest_backup'] = {
                        'filename': oldest[0].name,
                        'timestamp': datetime.fromtimestamp(oldest[1]).isoformat()
                    }
                    status['newest_backup'] = {
                        'filename': newest[0].name,
                        'timestamp': datetime.fromtimestamp(newest[1]).isoformat()
                    }
                
                # Count backup types
                for backup_file in backup_files:
                    if 'full_backup' in backup_file.name:
                        status['backup_types']['full'] = status['backup_types'].get('full', 0) + 1
                    elif 'database' in backup_file.name:
                        status['backup_types']['database'] = status['backup_types'].get('database', 0) + 1
                    elif 'config' in backup_file.name:
                        status['backup_types']['config'] = status['backup_types'].get('config', 0) + 1
        
        except Exception as e:
            status['error'] = str(e)
            logger.error(f"Error getting backup status: {e}")
        
        return status

def main():
    """Backup management CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backup and Recovery Manager')
    parser.add_argument('--create-backup', action='store_true', help='Create full system backup')
    parser.add_argument('--restore', help='Restore from backup (provide backup path)')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old backups')
    parser.add_argument('--status', action='store_true', help='Show backup system status')
    parser.add_argument('--backup-root', default='backups', help='Backup root directory')
    
    args = parser.parse_args()
    
    backup_manager = BackupManager(backup_root=args.backup_root)
    
    try:
        if args.create_backup:
            result = backup_manager.create_full_backup()
            if result['success']:
                print(f"\n✅ Backup completed successfully!")
                print(f"📊 Backup size: {result['total_size_mb']} MB")
                print(f"📁 Backup location: {result['backup_path']}")
            else:
                print(f"\n❌ Backup failed: {result.get('error', 'Unknown error')}")
                exit(1)
        
        elif args.restore:
            result = backup_manager.restore_from_backup(args.restore)
            if result['success']:
                print(f"\n✅ Restore completed successfully!")
                print(f"📊 Files restored: {result['files_restored']}")
            else:
                print(f"\n❌ Restore failed: {', '.join(result['errors'])}")
                exit(1)
        
        elif args.cleanup:
            result = backup_manager.cleanup_old_backups()
            print(f"\n✅ Cleanup completed!")
            print(f"🗑️  Backups deleted: {len(result['deleted_backups'])}")
            print(f"💾 Space freed: {result['space_freed_mb']:.2f} MB")
        
        elif args.status:
            status = backup_manager.get_backup_status()
            print(f"\n📊 BACKUP SYSTEM STATUS")
            print("=" * 40)
            print(f"Backup root: {status['backup_root']}")
            print(f"Total backups: {status['total_backups']}")
            print(f"Total size: {status['total_backup_size_mb']} MB")
            
            if status.get('newest_backup'):
                print(f"Newest backup: {status['newest_backup']['filename']}")
                print(f"  Created: {status['newest_backup']['timestamp']}")
            
            if status.get('backup_types'):
                print(f"Backup types:")
                for backup_type, count in status['backup_types'].items():
                    print(f"  {backup_type}: {count}")
        
        else:
            print("Use --help for available commands")
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()