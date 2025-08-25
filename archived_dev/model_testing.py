"""
Model Testing Framework
======================

Provides testing functionality for model validation and diagnostics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelTester:
    """Simple model testing framework."""
    
    def __init__(self, model_system):
        self.model_system = model_system
        logger.info("Model tester initialized")
    
    def run_comprehensive_test(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive model testing."""
        try:
            # Basic prediction test
            predictions = self.model_system.predict_proba(test_data)
            
            results = {
                'test_passed': True,
                'prediction_count': len(predictions),
                'prediction_range': [float(predictions.min()), float(predictions.max())],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Model test completed: {len(predictions)} predictions generated")
            return results
            
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return {
                'test_passed': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def run_model_testing_suite(model_system, test_data: pd.DataFrame, 
                          comprehensive: bool = True) -> Dict[str, Any]:
    """Run the complete model testing suite."""
    logger.info("Running model testing suite...")
    
    tester = ModelTester(model_system)
    
    if comprehensive:
        results = tester.run_comprehensive_test(test_data)
    else:
        # Quick test
        try:
            predictions = model_system.predict_proba(test_data.head(10))
            results = {
                'test_passed': True,
                'quick_test': True,
                'prediction_count': len(predictions),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            results = {
                'test_passed': False,
                'quick_test': True,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    logger.info(f"Testing suite completed: {'PASSED' if results['test_passed'] else 'FAILED'}")
    return results

def quick_model_diagnosis(model_system) -> Dict[str, Any]:
    """Quick model health check."""
    logger.info("Running quick model diagnosis...")
    
    diagnosis = {
        'model_loaded': model_system is not None,
        'core_model_available': hasattr(model_system, 'core_model') and model_system.core_model is not None,
        'enhanced_model_available': hasattr(model_system, 'enhanced_model') and model_system.enhanced_model is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    if diagnosis['core_model_available']:
        try:
            feature_count = len(model_system.available_features.get('core', []))
            diagnosis['core_feature_count'] = feature_count
        except:
            diagnosis['core_feature_count'] = 'unknown'
    
    if diagnosis['enhanced_model_available']:
        try:
            feature_count = len(model_system.available_features.get('enhanced', []))
            diagnosis['enhanced_feature_count'] = feature_count
        except:
            diagnosis['enhanced_feature_count'] = 'unknown'
    
    logger.info("Quick diagnosis completed")
    return diagnosis

__all__ = ['ModelTester', 'run_model_testing_suite', 'quick_model_diagnosis']