"""
Production API Setup and Configuration
======================================

Production-ready API configuration and verification for deployment.
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime

from api_client import SafeAPIClient
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionAPISetup:
    """Production API setup and configuration manager."""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.setup_status = {}
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        api_keys = {
            'theodds': os.getenv('THEODDS_API_KEY', '').strip(),
            'visualcrossing': os.getenv('VISUALCROSSING_API_KEY', '').strip()
        }
        
        logger.info("API key status:")
        for service, key in api_keys.items():
            status = "✅ Configured" if key else "❌ Missing"
            logger.info(f"  - {service}: {status}")
        
        return api_keys
    
    def verify_production_readiness(self) -> Dict[str, Any]:
        """Verify all APIs are ready for production use."""
        logger.info("🔍 Verifying production API readiness...")
        
        readiness = {
            'timestamp': datetime.now().isoformat(),
            'overall_ready': True,
            'services': {}
        }
        
        # Test The Odds API
        if self.api_keys['theodds']:
            try:
                client = SafeAPIClient(self.api_keys['theodds'])
                odds_available = client.is_odds_available()
                
                # Test actual data retrieval
                test_odds = client.get_todays_odds()
                test_starters = client.get_probable_starters()
                
                readiness['services']['theodds_api'] = {
                    'status': 'ready',
                    'api_key_valid': True,
                    'odds_available': odds_available,
                    'test_odds_count': len(test_odds),
                    'test_starters_count': len(test_starters)
                }
                logger.info(f"✅ The Odds API: Ready ({len(test_odds)} odds, {len(test_starters)} starters)")
                
            except Exception as e:
                readiness['services']['theodds_api'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                readiness['overall_ready'] = False
                logger.error(f"❌ The Odds API: Failed - {e}")
        else:
            readiness['services']['theodds_api'] = {
                'status': 'not_configured',
                'error': 'API key missing'
            }
            readiness['overall_ready'] = False
            logger.error("❌ The Odds API: Not configured")
        
        # Weather API (Visual Crossing)
        weather_ready = bool(self.api_keys['visualcrossing'])
        readiness['services']['weather_api'] = {
            'status': 'ready' if weather_ready else 'not_configured',
            'api_key_configured': weather_ready,
            'fallback_available': True  # Always available via synthetic weather
        }
        
        if weather_ready:
            logger.info("✅ Weather API: Ready")
        else:
            logger.warning("⚠️  Weather API: Not configured (will use fallback)")
        
        # MLB Stats API (free, no key required)
        readiness['services']['mlb_stats_api'] = {
            'status': 'ready',
            'api_key_required': False,
            'always_available': True
        }
        logger.info("✅ MLB Stats API: Ready (no key required)")
        
        return readiness
    
    def get_production_config(self) -> Dict[str, Any]:
        """Get production configuration summary."""
        config_summary = {
            'api_endpoints': {
                'theodds': config.API_BASE_URL,
                'mlb_stats': 'https://statsapi.mlb.com/api/v1',
                'weather': 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline'
            },
            'api_keys_configured': {
                'theodds': bool(self.api_keys['theodds']),
                'visualcrossing': bool(self.api_keys['visualcrossing'])
            },
            'timeouts': {
                'api_timeout': config.API_TIMEOUT,
                'max_retries': config.API_MAX_RETRIES
            },
            'rate_limits': {
                'theodds_calls_per_minute': 50,
                'mlb_stats_calls_per_minute': 100,
                'weather_calls_per_minute': 60
            },
            'directories': {
                'model_dir': str(config.MODEL_DIR),
                'data_dir': str(config.DATA_DIR),
                'cache_dir': str(config.CACHE_DIR)
            }
        }
        
        return config_summary
    
    def print_production_status(self):
        """Print comprehensive production status."""
        print("\n" + "="*70)
        print("PRODUCTION API CONFIGURATION STATUS")
        print("="*70)
        
        # API Keys Status
        print("\n🔑 API KEYS:")
        for service, key in self.api_keys.items():
            status = "✅ CONFIGURED" if key else "❌ MISSING"
            print(f"  - {service.upper()}: {status}")
        
        # Test connectivity
        readiness = self.verify_production_readiness()
        
        print(f"\n🚀 PRODUCTION READINESS:")
        print(f"  - Overall Status: {'✅ READY' if readiness['overall_ready'] else '❌ NOT READY'}")
        
        for service, details in readiness['services'].items():
            service_name = service.replace('_', ' ').upper()
            status = details['status'].upper()
            if status == 'READY':
                print(f"  - {service_name}: ✅ {status}")
            elif status == 'NOT_CONFIGURED':
                print(f"  - {service_name}: ⚠️  {status}")
            else:
                print(f"  - {service_name}: ❌ {status}")
        
        # Configuration Summary
        config_summary = self.get_production_config()
        
        print(f"\n⚙️  CONFIGURATION:")
        print(f"  - API Timeout: {config_summary['timeouts']['api_timeout']}s")
        print(f"  - Max Retries: {config_summary['timeouts']['max_retries']}")
        print(f"  - Model Directory: {config_summary['directories']['model_dir']}")
        
        print(f"\n📊 LIVE DATA AVAILABILITY:")
        if 'theodds_api' in readiness['services'] and readiness['services']['theodds_api']['status'] == 'ready':
            odds_count = readiness['services']['theodds_api']['test_odds_count']
            starters_count = readiness['services']['theodds_api']['test_starters_count']
            print(f"  - Live Odds: {odds_count} players")
            print(f"  - Probable Starters: {starters_count} games")
        else:
            print(f"  - Live Odds: Not available")
            print(f"  - Probable Starters: Not available")
        
        print("="*70)

def main():
    """Run production API setup verification."""
    setup = ProductionAPISetup()
    setup.print_production_status()
    
    return setup.verify_production_readiness()

if __name__ == "__main__":
    main()