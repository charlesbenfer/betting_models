#!/usr/bin/env python3
"""
Environment Validation Script
=============================

Comprehensive environment validation for production deployment.
"""

import os
import sys
import platform
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version requirements."""
    version = sys.version_info
    required_major, required_minor = 3, 8
    
    if version.major >= required_major and version.minor >= required_minor:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} < {required_major}.{required_minor}"

def check_required_packages() -> Tuple[bool, List[str]]:
    """Check for required Python packages."""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'requests', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_environment_variables() -> Tuple[bool, Dict[str, str]]:
    """Check required environment variables."""
    required_vars = ['THEODDS_API_KEY']
    optional_vars = ['VISUALCROSSING_API_KEY']
    
    env_status = {}
    all_required_present = True
    
    for var in required_vars:
        value = os.getenv(var, '')
        if value.strip():
            env_status[var] = f"✅ Set ({value[:10]}...)"
        else:
            env_status[var] = "❌ Missing"
            all_required_present = False
    
    for var in optional_vars:
        value = os.getenv(var, '')
        if value.strip():
            env_status[var] = f"✅ Set ({value[:10]}...)"
        else:
            env_status[var] = "⚠️  Not set (optional)"
    
    return all_required_present, env_status

def check_directory_structure() -> Tuple[bool, List[str]]:
    """Check required directory structure."""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        'data',
        'saved_models_pregame',
        'logs'
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    return len(missing_dirs) == 0, missing_dirs

def check_required_files() -> Tuple[bool, List[str]]:
    """Check for required project files."""
    project_root = Path(__file__).parent.parent
    
    required_files = [
        'config.py',
        'modeling.py',
        'api_client.py',
        'live_prediction_system.py',
        'main.py'
    ]
    
    missing_files = []
    
    for file_name in required_files:
        file_path = project_root / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    return len(missing_files) == 0, missing_files

def check_system_resources() -> Dict[str, Any]:
    """Check system resources and capabilities."""
    import psutil
    
    # Get system information
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu_cores': cpu_count,
        'total_memory_gb': round(memory.total / (1024**3), 2),
        'available_memory_gb': round(memory.available / (1024**3), 2),
        'disk_free_gb': round(disk.free / (1024**3), 2),
        'platform': platform.platform()
    }

def check_network_connectivity() -> Tuple[bool, str]:
    """Check network connectivity to required services."""
    test_urls = [
        'https://api.the-odds-api.com',
        'https://statsapi.mlb.com',
        'https://weather.visualcrossing.com'
    ]
    
    import urllib.request
    import urllib.error
    
    connectivity_results = []
    all_connected = True
    
    for url in test_urls:
        try:
            urllib.request.urlopen(url, timeout=10)
            connectivity_results.append(f"✅ {url}")
        except (urllib.error.URLError, Exception) as e:
            connectivity_results.append(f"❌ {url} - {str(e)}")
            all_connected = False
    
    return all_connected, '\n'.join(connectivity_results)

def run_comprehensive_validation() -> Dict[str, Any]:
    """Run comprehensive environment validation."""
    print("🔍 Running comprehensive environment validation...")
    print("=" * 60)
    
    validation_results = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'overall_status': True,
        'checks': {}
    }
    
    # Check 1: Python version
    python_ok, python_info = check_python_version()
    validation_results['checks']['python_version'] = {
        'status': python_ok,
        'info': python_info
    }
    print(f"Python Version: {'✅' if python_ok else '❌'} {python_info}")
    
    # Check 2: Required packages
    packages_ok, missing_packages = check_required_packages()
    validation_results['checks']['packages'] = {
        'status': packages_ok,
        'missing': missing_packages
    }
    if packages_ok:
        print("Required Packages: ✅ All present")
    else:
        print(f"Required Packages: ❌ Missing: {', '.join(missing_packages)}")
    
    # Check 3: Environment variables
    env_ok, env_status = check_environment_variables()
    validation_results['checks']['environment_variables'] = {
        'status': env_ok,
        'variables': env_status
    }
    print("Environment Variables:")
    for var, status in env_status.items():
        print(f"  - {var}: {status}")
    
    # Check 4: Directory structure
    dirs_ok, missing_dirs = check_directory_structure()
    validation_results['checks']['directories'] = {
        'status': dirs_ok,
        'missing': missing_dirs
    }
    if dirs_ok:
        print("Directory Structure: ✅ All required directories present")
    else:
        print(f"Directory Structure: ⚠️  Missing: {', '.join(missing_dirs)}")
    
    # Check 5: Required files
    files_ok, missing_files = check_required_files()
    validation_results['checks']['files'] = {
        'status': files_ok,
        'missing': missing_files
    }
    if files_ok:
        print("Required Files: ✅ All present")
    else:
        print(f"Required Files: ❌ Missing: {', '.join(missing_files)}")
    
    # Check 6: System resources
    try:
        resources = check_system_resources()
        validation_results['checks']['system_resources'] = {
            'status': True,
            'resources': resources
        }
        print("System Resources:")
        print(f"  - CPU Cores: {resources['cpu_cores']}")
        print(f"  - Memory: {resources['available_memory_gb']:.1f}GB available / {resources['total_memory_gb']:.1f}GB total")
        print(f"  - Disk Space: {resources['disk_free_gb']:.1f}GB free")
        print(f"  - Platform: {resources['platform']}")
    except Exception as e:
        validation_results['checks']['system_resources'] = {
            'status': False,
            'error': str(e)
        }
        print(f"System Resources: ❌ Error checking: {e}")
    
    # Check 7: Network connectivity
    try:
        network_ok, network_info = check_network_connectivity()
        validation_results['checks']['network'] = {
            'status': network_ok,
            'info': network_info
        }
        print("Network Connectivity:")
        for line in network_info.split('\n'):
            print(f"  {line}")
    except Exception as e:
        validation_results['checks']['network'] = {
            'status': False,
            'error': str(e)
        }
        print(f"Network Connectivity: ❌ Error checking: {e}")
    
    # Overall status
    critical_checks = ['python_version', 'packages', 'environment_variables', 'files']
    validation_results['overall_status'] = all(
        validation_results['checks'][check]['status'] 
        for check in critical_checks
        if check in validation_results['checks']
    )
    
    print("=" * 60)
    if validation_results['overall_status']:
        print("🎉 Environment validation: ✅ PASSED")
        print("System is ready for production deployment!")
    else:
        print("❌ Environment validation: FAILED")
        print("Please fix the issues above before deployment.")
    
    return validation_results

def main():
    """Main validation function."""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Environment Validation Script")
        print("Usage: python3 validate_environment.py")
        print("Validates the environment for production deployment.")
        sys.exit(0)
    
    try:
        results = run_comprehensive_validation()
        sys.exit(0 if results['overall_status'] else 1)
    except Exception as e:
        print(f"❌ Validation script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()