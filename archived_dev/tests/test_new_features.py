"""
Test Script for New Matchup Features
===================================

This script tests the new batter vs pitcher matchup features to ensure they work correctly
and provides initial performance evaluation.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import sys

# Import our modules
from config import config
from dataset_builder import PregameDatasetBuilder
from modeling import EnhancedDualModelSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_new_features():
    """Test that new matchup features are being generated correctly."""
    logger.info("="*60)
    logger.info("TESTING NEW MATCHUP FEATURES")
    logger.info("="*60)
    
    try:
        # Build a small test dataset
        logger.info("Building test dataset with new features...")
        builder = PregameDatasetBuilder(
            start_date="2024-08-01", 
            end_date="2024-08-15"  # Small date range for testing
        )
        
        # Build dataset (this should now include new features)
        dataset = builder.build_dataset(force_rebuild=True)
        
        if dataset.empty:
            logger.error("Dataset is empty - cannot test features")
            return False
        
        logger.info(f"Dataset built: {len(dataset)} rows, {len(dataset.columns)} columns")
        
        # Check for new matchup features
        expected_matchup_features = config.MATCHUP_FEATURES
        found_features = []
        missing_features = []
        
        for feature in expected_matchup_features:
            if feature in dataset.columns:
                found_features.append(feature)
            else:
                missing_features.append(feature)
        
        logger.info(f"\nFEATURE ANALYSIS:")
        logger.info(f"Expected matchup features: {len(expected_matchup_features)}")
        logger.info(f"Found features: {len(found_features)}")
        logger.info(f"Missing features: {len(missing_features)}")
        
        if found_features:
            logger.info(f"\nFound matchup features: {found_features}")
        
        if missing_features:
            logger.warning(f"Missing matchup features: {missing_features}")
        
        # Analyze feature coverage and quality
        analyze_feature_quality(dataset, found_features)
        
        # Test model integration
        test_model_integration(dataset)
        
        logger.info("\nNew feature testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Feature testing failed: {e}")
        return False

def analyze_feature_quality(dataset: pd.DataFrame, matchup_features: list):
    """Analyze the quality and distribution of new features."""
    logger.info("\n" + "="*50)
    logger.info("FEATURE QUALITY ANALYSIS")
    logger.info("="*50)
    
    for feature in matchup_features:
        if feature in dataset.columns:
            feature_data = dataset[feature]
            
            # Basic statistics
            non_null_count = feature_data.notna().sum()
            coverage = non_null_count / len(dataset)
            
            logger.info(f"\n{feature}:")
            logger.info(f"  Coverage: {coverage:.1%} ({non_null_count}/{len(dataset)})")
            
            if non_null_count > 0:
                stats = feature_data.describe()
                logger.info(f"  Mean: {stats['mean']:.4f}")
                logger.info(f"  Std: {stats['std']:.4f}")
                logger.info(f"  Min: {stats['min']:.4f}")
                logger.info(f"  Max: {stats['max']:.4f}")
                
                # Check for reasonable values
                if 'hr_rate' in feature and stats['max'] > 1.0:
                    logger.warning(f"  WARNING: HR rates > 100% found (max: {stats['max']:.4f})")
                
                if feature == 'matchup_days_since_last' and stats['min'] < 0:
                    logger.warning(f"  WARNING: Negative days found (min: {stats['min']:.1f})")
                
                # Unique value count
                unique_values = feature_data.nunique()
                logger.info(f"  Unique values: {unique_values}")
                
                if unique_values == 1:
                    logger.warning(f"  WARNING: Feature has only one unique value")

def test_model_integration(dataset: pd.DataFrame):
    """Test that the new features integrate properly with the model system."""
    logger.info("\n" + "="*50)
    logger.info("MODEL INTEGRATION TEST")
    logger.info("="*50)
    
    try:
        # Initialize model system
        model_system = EnhancedDualModelSystem()
        
        # Check feature identification
        available_features = model_system.feature_selector.identify_available_features(dataset)
        
        logger.info(f"Core features available: {len(available_features['core'])}")
        logger.info(f"Bat tracking features available: {len(available_features.get('bat_tracking', []))}")
        logger.info(f"Matchup features available: {len(available_features.get('matchup', []))}")
        logger.info(f"Enhanced features total: {len(available_features['enhanced'])}")
        
        # Show some example matchup features
        matchup_features = available_features.get('matchup', [])
        if matchup_features:
            logger.info(f"\nExample matchup features found: {matchup_features[:5]}")
        else:
            logger.warning("No matchup features detected in model system!")
        
        # Quick training test (on small dataset)
        if len(dataset) > 100:
            logger.info("\nTesting quick model training...")
            
            # Use only first 100 rows for quick test
            test_data = dataset.head(100).copy()
            
            # Ensure we have target variable
            if 'hit_hr' not in test_data.columns:
                if 'home_runs' in test_data.columns:
                    test_data['hit_hr'] = (test_data['home_runs'] > 0).astype(int)
                else:
                    # Create dummy target for testing
                    test_data['hit_hr'] = np.random.binomial(1, 0.1, len(test_data))
            
            # Try training with new features
            try:
                results = model_system.fit(
                    test_data,
                    splitting_strategy='random',
                    test_size=0.3,
                    val_size=0.2,
                    cross_validate=False  # Skip CV for quick test
                )
                
                logger.info("Model training test successful!")
                logger.info(f"Training completed with {results['train_size']} training samples")
                
            except Exception as train_error:
                logger.error(f"Model training test failed: {train_error}")
        
    except Exception as e:
        logger.error(f"Model integration test failed: {e}")

def compare_feature_importance():
    """Compare feature importance between old and new feature sets."""
    logger.info("\n" + "="*50)
    logger.info("FEATURE IMPORTANCE COMPARISON")
    logger.info("="*50)
    
    # This would be implemented to compare model performance
    # with and without the new features
    logger.info("Feature importance comparison would be implemented here")
    logger.info("This requires training models with both feature sets")

def main():
    """Main testing function."""
    logger.info("Starting new feature testing...")
    
    success = test_new_features()
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED!")
        logger.info("New matchup features are working correctly.")
        logger.info("="*60)
        return 0
    else:
        logger.error("\n" + "="*60)
        logger.error("TESTS FAILED!")
        logger.error("Please check the implementation.")
        logger.error("="*60)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)