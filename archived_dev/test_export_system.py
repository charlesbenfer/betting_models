"""
Production Export System Test
=============================

Test export system components without requiring a fully trained model.
"""

import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from modeling import EnhancedDualModelSystem
from inference_features import InferenceFeatureCalculator, ProductionInferenceExample
from matchup_database import MatchupDatabase
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExportSystemTester:
    """Test production export system components."""
    
    def __init__(self):
        self.results = {}
    
    def test_inference_components(self) -> bool:
        """Test inference feature calculation components."""
        logger.info("Testing inference feature calculation system...")
        
        try:
            # Test InferenceFeatureCalculator
            feature_calc = InferenceFeatureCalculator()
            
            # Test MatchupDatabase
            matchup_db = MatchupDatabase()
            db_stats = matchup_db.get_database_stats()
            
            logger.info(f"✅ Matchup database loaded: {db_stats}")
            
            # Test ProductionInferenceExample
            inference_example = ProductionInferenceExample()
            
            self.results['inference_components'] = {
                'feature_calculator': True,
                'matchup_database': True,
                'database_stats': db_stats,
                'production_example': True
            }
            
            logger.info("✅ Inference components test passed")
            return True
            
        except Exception as e:
            logger.error(f"Inference components test failed: {e}")
            self.results['inference_components'] = {'error': str(e)}
            return False
    
    def test_model_system_structure(self) -> bool:
        """Test model system structure and directories."""
        logger.info("Testing model system structure...")
        
        try:
            model_dir = Path(config.MODEL_DIR)
            
            # Ensure model directory exists
            model_dir.mkdir(exist_ok=True)
            
            # Test model system initialization
            model_system = EnhancedDualModelSystem(str(model_dir))
            
            structure_info = {
                'model_directory': str(model_dir),
                'directory_exists': model_dir.exists(),
                'model_system_initialized': True,
                'available_features': model_system.available_features,
                'metadata': model_system.metadata
            }
            
            logger.info(f"✅ Model system structure valid: {str(model_dir)}")
            self.results['model_structure'] = structure_info
            return True
            
        except Exception as e:
            logger.error(f"Model system structure test failed: {e}")
            self.results['model_structure'] = {'error': str(e)}
            return False
    
    def test_feature_calculation_speed(self) -> Dict[str, Any]:
        """Test feature calculation speed (without model prediction)."""
        logger.info("Testing feature calculation speed...")
        
        try:
            feature_calc = InferenceFeatureCalculator()
            
            # Sample batter-pitcher pairs
            sample_pairs = [(400000 + i, 500000 + i) for i in range(50)]
            
            # Time feature calculation
            start_time = time.time()
            
            features_list = []
            for batter_id, pitcher_id in sample_pairs:
                features = feature_calc.get_game_features(
                    batter_id, pitcher_id, "2024-08-22"
                )
                features_list.append(features)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            speed_stats = {
                'total_calculations': len(features_list),
                'total_time_seconds': round(total_time, 3),
                'calculations_per_second': round(len(features_list) / total_time, 2),
                'ms_per_calculation': round((total_time * 1000) / len(features_list), 3),
                'avg_features_per_calculation': round(np.mean([len(f) for f in features_list]), 1)
            }
            
            logger.info(f"✅ Feature calculation speed test:")
            logger.info(f"   - {speed_stats['calculations_per_second']} calculations/second")
            logger.info(f"   - {speed_stats['ms_per_calculation']} ms per calculation")
            logger.info(f"   - {speed_stats['avg_features_per_calculation']} avg features per calculation")
            
            self.results['feature_speed'] = speed_stats
            return speed_stats
            
        except Exception as e:
            logger.error(f"Feature calculation speed test failed: {e}")
            self.results['feature_speed'] = {'error': str(e)}
            return {}
    
    def test_export_readiness(self) -> bool:
        """Test overall export system readiness."""
        logger.info("Testing export system readiness...")
        
        try:
            # Check required files exist
            required_files = [
                'modeling.py',
                'inference_features.py', 
                'matchup_database.py',
                'api_client.py',
                'live_prediction_system.py',
                'config.py'
            ]
            
            missing_files = []
            for file in required_files:
                if not Path(file).exists():
                    missing_files.append(file)
            
            # Check data directory
            data_dir = Path('data')
            matchup_db_exists = (data_dir / 'matchup_database.db').exists()
            
            readiness_info = {
                'required_files_present': len(missing_files) == 0,
                'missing_files': missing_files,
                'matchup_database_exists': matchup_db_exists,
                'data_directory_exists': data_dir.exists()
            }
            
            all_ready = (
                len(missing_files) == 0 and 
                matchup_db_exists and 
                data_dir.exists()
            )
            
            if all_ready:
                logger.info("✅ Export system is ready for production")
            else:
                logger.warning(f"⚠️  Export system issues: {readiness_info}")
            
            self.results['export_readiness'] = readiness_info
            return all_ready
            
        except Exception as e:
            logger.error(f"Export readiness test failed: {e}")
            self.results['export_readiness'] = {'error': str(e)}
            return False
    
    def run_system_tests(self) -> Dict[str, Any]:
        """Run all export system tests."""
        logger.info("🚀 Starting export system tests...")
        
        # Test 1: Inference components
        components_ok = self.test_inference_components()
        
        # Test 2: Model system structure
        structure_ok = self.test_model_system_structure()
        
        # Test 3: Feature calculation speed
        speed_results = self.test_feature_calculation_speed()
        
        # Test 4: Export readiness
        readiness_ok = self.test_export_readiness()
        
        # Summary
        all_passed = (
            components_ok and 
            structure_ok and 
            bool(speed_results) and 
            readiness_ok
        )
        
        summary = {
            'overall_success': all_passed,
            'components_test': components_ok,
            'structure_test': structure_ok,
            'speed_test_results': speed_results,
            'readiness_test': readiness_ok,
            'timestamp': pd.Timestamp.now().isoformat(),
            'note': 'System tests completed (model training not required)'
        }
        
        if all_passed:
            logger.info("🎉 All export system tests PASSED!")
            logger.info("System is ready for production deployment once model is trained")
        else:
            logger.warning("⚠️  Some export system tests FAILED")
        
        return summary

def main():
    """Run export system tests."""
    tester = ExportSystemTester()
    results = tester.run_system_tests()
    
    print("\n" + "="*60)
    print("EXPORT SYSTEM TEST RESULTS")
    print("="*60)
    print(f"Overall Success: {'✅ PASS' if results['overall_success'] else '❌ FAIL'}")
    print(f"Components Test: {'✅ PASS' if results['components_test'] else '❌ FAIL'}")
    print(f"Structure Test: {'✅ PASS' if results['structure_test'] else '❌ FAIL'}")
    print(f"Readiness Test: {'✅ PASS' if results['readiness_test'] else '❌ FAIL'}")
    
    if results['speed_test_results']:
        speed = results['speed_test_results']
        print(f"Speed Test: ✅ PASS")
        print(f"  - {speed.get('calculations_per_second', 0)} feature calculations/second")
        print(f"  - {speed.get('ms_per_calculation', 0)} ms per calculation")
    else:
        print(f"Speed Test: ❌ FAIL")
    
    print("\n📋 Note: Model training in progress or required for full functionality")
    print("="*60)
    
    return results

if __name__ == "__main__":
    main()