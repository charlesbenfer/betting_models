#!/usr/bin/env python3
"""
Production Startup Script
=========================

Automated startup script for the baseball HR prediction betting system.
Handles initialization, validation, and service startup.
"""

import os
import sys
import logging
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from production_api_setup import ProductionAPISetup
from test_export_system import ExportSystemTester
from live_prediction_system import LivePredictionSystem

class ProductionStartup:
    """Production startup manager for the betting system."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.setup_logging()
        self.startup_results = {}
        
        logger.info("🚀 Production Startup Manager initialized")
    
    def setup_logging(self):
        """Configure production logging."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"startup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        global logger
        logger = logging.getLogger(__name__)
    
    def validate_environment(self) -> bool:
        """Validate production environment prerequisites."""
        logger.info("🔍 Validating production environment...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
                logger.error(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
                return False
            
            logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Check required directories
            required_dirs = [
                config.MODEL_DIR,
                config.DATA_DIR,
                config.CACHE_DIR,
                Path("logs")
            ]
            
            for dir_path in required_dirs:
                if not dir_path.exists():
                    logger.info(f"📁 Creating directory: {dir_path}")
                    dir_path.mkdir(parents=True, exist_ok=True)
                else:
                    logger.info(f"✅ Directory exists: {dir_path}")
            
            # Check for required files
            required_files = [
                "modeling.py",
                "live_prediction_system.py", 
                "api_client.py",
                "config.py"
            ]
            
            missing_files = []
            for file_name in required_files:
                if not Path(file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                logger.error(f"❌ Missing required files: {missing_files}")
                return False
            
            logger.info("✅ All required files present")
            
            # Check API keys
            api_setup = ProductionAPISetup()
            api_keys = api_setup._load_api_keys()
            
            if not api_keys['theodds']:
                logger.error("❌ THEODDS_API_KEY not configured")
                return False
            
            logger.info("✅ API keys configured")
            
            self.startup_results['environment_validation'] = {
                'status': 'success',
                'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                'directories_created': len(required_dirs),
                'api_keys_configured': bool(api_keys['theodds'])
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            self.startup_results['environment_validation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def test_system_components(self) -> bool:
        """Test all system components for production readiness."""
        logger.info("🧪 Testing system components...")
        
        try:
            # Test export system
            logger.info("Testing export system...")
            export_tester = ExportSystemTester()
            export_results = export_tester.run_system_tests()
            
            if not export_results['overall_success']:
                logger.error("❌ Export system tests failed")
                return False
            
            logger.info("✅ Export system operational")
            
            # Test API connectivity
            logger.info("Testing API connectivity...")
            api_setup = ProductionAPISetup()
            api_readiness = api_setup.verify_production_readiness()
            
            if not api_readiness['overall_ready']:
                logger.error("❌ API connectivity tests failed")
                return False
            
            logger.info(f"✅ APIs operational ({api_readiness['services']['theodds_api']['test_odds_count']} odds available)")
            
            self.startup_results['component_testing'] = {
                'status': 'success',
                'export_system': export_results['overall_success'],
                'api_connectivity': api_readiness['overall_ready'],
                'live_odds_count': api_readiness['services']['theodds_api'].get('test_odds_count', 0)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Component testing failed: {e}")
            self.startup_results['component_testing'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def initialize_live_system(self) -> bool:
        """Initialize the live prediction system."""
        logger.info("🎯 Initializing live prediction system...")
        
        try:
            # Get API key
            api_key = os.getenv("THEODDS_API_KEY", "").strip()
            
            # Initialize live system
            live_system = LivePredictionSystem(api_key=api_key)
            
            # Test initialization
            init_success = live_system.initialize()
            
            if not init_success:
                logger.error("❌ Live system initialization failed")
                return False
            
            # Get system info
            model_info = live_system.model_system.get_model_info()
            logger.info(f"✅ Live system initialized:")
            logger.info(f"   - Model type: {model_info['active_model']}")
            logger.info(f"   - Features: {model_info['active_feature_count']}")
            logger.info(f"   - APIs available: {live_system.api_client.is_odds_available()}")
            
            self.startup_results['live_system'] = {
                'status': 'success',
                'model_type': model_info['active_model'],
                'feature_count': model_info['active_feature_count'],
                'apis_available': live_system.api_client.is_odds_available()
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Live system initialization failed: {e}")
            self.startup_results['live_system'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_startup_sequence(self) -> bool:
        """Run complete production startup sequence."""
        logger.info("🚀 Starting production startup sequence...")
        
        start_time = time.time()
        
        # Step 1: Environment validation
        if not self.validate_environment():
            logger.error("❌ Startup failed at environment validation")
            return False
        
        # Step 2: Component testing
        if not self.test_system_components():
            logger.error("❌ Startup failed at component testing")
            return False
        
        # Step 3: Live system initialization
        if not self.initialize_live_system():
            logger.error("❌ Startup failed at live system initialization")
            return False
        
        end_time = time.time()
        startup_duration = end_time - start_time
        
        logger.info(f"🎉 Production startup completed successfully in {startup_duration:.2f} seconds!")
        
        self.startup_results['overall'] = {
            'status': 'success',
            'duration_seconds': startup_duration,
            'timestamp': datetime.now().isoformat()
        }
        
        return True
    
    def print_startup_summary(self):
        """Print comprehensive startup summary."""
        print("\n" + "="*70)
        print("PRODUCTION STARTUP SUMMARY")
        print("="*70)
        
        if 'overall' in self.startup_results:
            overall = self.startup_results['overall']
            duration = overall.get('duration_seconds', 0)
            print(f"Overall Status: {'✅ SUCCESS' if overall['status'] == 'success' else '❌ FAILED'}")
            print(f"Startup Time: {duration:.2f} seconds")
        
        print(f"\n📊 COMPONENT STATUS:")
        
        # Environment
        if 'environment_validation' in self.startup_results:
            env = self.startup_results['environment_validation']
            status = '✅ PASS' if env['status'] == 'success' else '❌ FAIL'
            print(f"  - Environment: {status}")
            if env['status'] == 'success':
                print(f"    Python: {env['python_version']}")
                print(f"    Directories: {env['directories_created']} created/verified")
        
        # Components
        if 'component_testing' in self.startup_results:
            comp = self.startup_results['component_testing']
            status = '✅ PASS' if comp['status'] == 'success' else '❌ FAIL'
            print(f"  - System Components: {status}")
            if comp['status'] == 'success':
                print(f"    Live Odds Available: {comp['live_odds_count']} players")
        
        # Live System
        if 'live_system' in self.startup_results:
            live = self.startup_results['live_system']
            status = '✅ PASS' if live['status'] == 'success' else '❌ FAIL'
            print(f"  - Live System: {status}")
            if live['status'] == 'success':
                print(f"    Model: {live['model_type']} ({live['feature_count']} features)")
                print(f"    APIs: {'Available' if live['apis_available'] else 'Unavailable'}")
        
        print("="*70)

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description='Production startup for baseball HR prediction system')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--validate-only', action='store_true', help='Only validate environment, don\'t start services')
    
    args = parser.parse_args()
    
    # Create startup manager
    startup = ProductionStartup(verbose=args.verbose)
    
    try:
        if args.validate_only:
            # Only run validation
            success = startup.validate_environment()
        else:
            # Full startup sequence
            success = startup.run_startup_sequence()
        
        # Print summary
        startup.print_startup_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("🛑 Startup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Startup failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()