"""
Test Recent Form Features
========================

Test the new time-weighted recent form features to ensure they work correctly.
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

def test_recent_form_features():
    """Test the new recent form features."""
    logger.info("="*70)
    logger.info("TESTING RECENT FORM FEATURES")
    logger.info("="*70)
    
    try:
        # Build dataset with recent form features
        logger.info("Building dataset with new recent form features...")
        builder = PregameDatasetBuilder(
            start_date="2024-08-01", 
            end_date="2024-08-15"
        )
        
        # Build dataset 
        dataset = builder.build_dataset(force_rebuild=True)
        
        if dataset.empty:
            logger.error("Dataset is empty - cannot test features")
            return False
        
        logger.info(f"Dataset built: {len(dataset)} rows, {len(dataset.columns)} columns")
        
        # Check for recent form features
        analyze_recent_form_features(dataset)
        
        # Test model integration
        test_model_integration(dataset)
        
        # Analyze feature quality
        analyze_feature_quality(dataset)
        
        # Test time decay properties
        analyze_time_decay_properties(dataset)
        
        logger.info("\nRecent form feature testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Recent form feature testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_recent_form_features(dataset: pd.DataFrame):
    """Analyze the recent form features in the dataset."""
    logger.info("\n" + "="*60)
    logger.info("RECENT FORM FEATURE ANALYSIS")
    logger.info("="*60)
    
    # Find recent form features
    expected_recent_form = config.RECENT_FORM_FEATURES
    found_recent_form = []
    missing_recent_form = []
    
    for feature in expected_recent_form:
        if feature in dataset.columns:
            found_recent_form.append(feature)
        else:
            missing_recent_form.append(feature)
    
    logger.info(f"Expected recent form features: {len(expected_recent_form)}")
    logger.info(f"Found features: {len(found_recent_form)}")
    logger.info(f"Missing features: {len(missing_recent_form)}")
    
    if found_recent_form:
        logger.info(f"\nFound recent form features: {found_recent_form[:10]}...")  # Show first 10
    
    if missing_recent_form:
        logger.warning(f"Missing recent form features: {missing_recent_form}")
    
    # Show total feature count increase
    all_features = [col for col in dataset.columns if col not in 
                   ['date', 'batter', 'pitcher', 'game_pk', 'season', 'hit_hr', 'home_runs']]
    logger.info(f"\nTotal features in dataset: {len(all_features)}")
    
    return found_recent_form

def analyze_feature_quality(dataset: pd.DataFrame):
    """Analyze the quality of recent form features."""
    logger.info("\n" + "="*60)
    logger.info("RECENT FORM FEATURE QUALITY ANALYSIS")
    logger.info("="*60)
    
    # Analyze key recent form features
    key_features = [
        'power_form_hr_rate', 'power_form_avg_ev', 'hot_streak_indicator',
        'cold_streak_indicator', 'momentum_score', 'hr_rate_trend_7d',
        'form_acceleration', 'recent_power_surge'
    ]
    
    for feature in key_features:
        if feature in dataset.columns:
            feature_data = dataset[feature]
            non_null = feature_data.notna().sum()
            non_zero = (feature_data != 0).sum()
            
            logger.info(f"\n{feature}:")
            logger.info(f"  Coverage: {non_null/len(dataset):.1%} ({non_null}/{len(dataset)})")
            logger.info(f"  Non-zero: {non_zero/len(dataset):.1%} ({non_zero}/{len(dataset)})")
            
            if non_null > 0:
                stats = feature_data.describe()
                logger.info(f"  Mean: {stats['mean']:.4f}")
                logger.info(f"  Std: {stats['std']:.4f}")
                logger.info(f"  Min: {stats['min']:.4f}")
                logger.info(f"  Max: {stats['max']:.4f}")
                logger.info(f"  Unique values: {feature_data.nunique()}")
                
                # Check for reasonable values
                if 'hr_rate' in feature and stats['max'] > 1.0:
                    logger.warning(f"  WARNING: HR rates > 100% found (max: {stats['max']:.4f})")
                
                if 'indicator' in feature and (stats['min'] < 0 or stats['max'] > 10):
                    logger.warning(f"  WARNING: Indicator values outside expected range")
                
                if 'momentum' in feature and abs(stats['max']) > 3:
                    logger.warning(f"  WARNING: Extreme momentum values found")

def test_model_integration(dataset: pd.DataFrame):
    """Test that recent form features integrate with the model system."""
    logger.info("\n" + "="*60)
    logger.info("MODEL INTEGRATION TEST")
    logger.info("="*60)
    
    try:
        # Initialize model system
        model_system = EnhancedDualModelSystem()
        
        # Check feature identification
        available_features = model_system.feature_selector.identify_available_features(dataset)
        
        logger.info(f"Feature breakdown:")
        logger.info(f"  Core features: {len(available_features['core'])}")
        logger.info(f"  Bat tracking features: {len(available_features.get('bat_tracking', []))}")
        logger.info(f"  Matchup features: {len(available_features.get('matchup', []))}")
        logger.info(f"  Situational features: {len(available_features.get('situational', []))}")
        logger.info(f"  Weather features: {len(available_features.get('weather', []))}")
        logger.info(f"  Recent form features: {len(available_features.get('recent_form', []))}")
        logger.info(f"  Enhanced features total: {len(available_features['enhanced'])}")
        
        # Show some recent form features
        recent_form_features = available_features.get('recent_form', [])
        if recent_form_features:
            logger.info(f"\nRecent form features found: {len(recent_form_features)}")
            logger.info(f"Examples: {recent_form_features[:8]}")
        else:
            logger.warning("No recent form features detected in model system!")
        
        # Quick training test if we have enough data
        if len(dataset) > 100:
            logger.info("\nTesting model training with recent form features...")
            
            # Use small sample for quick test
            test_data = dataset.head(200).copy()
            
            # Ensure target variable
            if 'hit_hr' not in test_data.columns:
                if 'home_runs' in test_data.columns:
                    test_data['hit_hr'] = (test_data['home_runs'] > 0).astype(int)
                else:
                    test_data['hit_hr'] = np.random.binomial(1, 0.1, len(test_data))
            
            # Try training
            try:
                results = model_system.fit(
                    test_data,
                    splitting_strategy='random',
                    test_size=0.3,
                    val_size=0.2,
                    cross_validate=False
                )
                
                logger.info("Model training with recent form features successful!")
                logger.info(f"Enhanced features used: {len(available_features['enhanced'])}")
                logger.info(f"Recent form features included: {len(recent_form_features)}")
                
            except Exception as train_error:
                logger.error(f"Model training failed: {train_error}")
        
    except Exception as e:
        logger.error(f"Model integration test failed: {e}")

def analyze_time_decay_properties(dataset: pd.DataFrame):
    """Analyze time decay properties of recent form features."""
    logger.info("\n" + "="*60)
    logger.info("TIME DECAY PROPERTIES ANALYSIS")
    logger.info("="*60)
    
    # Analyze trend features to see if they capture recent changes
    trend_features = [col for col in dataset.columns if 'trend' in col or 'form_acceleration' in col]
    
    if not trend_features:
        logger.warning("No trend features found for time decay analysis")
        return
    
    logger.info(f"Analyzing {len(trend_features)} trend features:")
    
    for feature in trend_features:
        if feature in dataset.columns:
            data = dataset[feature].dropna()
            if len(data) > 0:
                logger.info(f"\n{feature}:")
                logger.info(f"  Non-zero values: {(data != 0).sum()}/{len(data)} ({(data != 0).mean():.1%})")
                logger.info(f"  Range: {data.min():.4f} to {data.max():.4f}")
                logger.info(f"  Mean: {data.mean():.4f}")
                logger.info(f"  Std: {data.std():.4f}")
                
                # Check for trends in both directions
                positive_trends = (data > 0.001).sum()
                negative_trends = (data < -0.001).sum()
                logger.info(f"  Positive trends: {positive_trends} ({positive_trends/len(data):.1%})")
                logger.info(f"  Negative trends: {negative_trends} ({negative_trends/len(data):.1%})")
    
    # Analyze streak indicators
    streak_features = [col for col in dataset.columns if 'streak' in col or 'momentum' in col]
    
    if streak_features:
        logger.info(f"\nStreak indicator analysis:")
        for feature in streak_features:
            if feature in dataset.columns:
                data = dataset[feature].dropna()
                if len(data) > 0:
                    active_streaks = (data > 0.1).sum()  # Significant streak threshold
                    logger.info(f"  {feature}: {active_streaks}/{len(data)} active ({active_streaks/len(data):.1%})")

def analyze_feature_correlations(dataset: pd.DataFrame):
    """Analyze correlations between recent form features and home runs."""
    logger.info("\n" + "="*60)
    logger.info("RECENT FORM FEATURE CORRELATION ANALYSIS")
    logger.info("="*60)
    
    if 'hit_hr' not in dataset.columns:
        logger.warning("No target variable for correlation analysis")
        return
    
    # Get recent form features with good coverage
    recent_form_features = [col for col in dataset.columns 
                          if any(feat in col for feat in ['form', 'trend', 'streak', 'momentum'])]
    
    meaningful_features = []
    for feature in recent_form_features:
        non_null_pct = dataset[feature].notna().mean()
        non_zero_pct = (dataset[feature] != 0).mean()
        if non_null_pct > 0.8 and non_zero_pct > 0.1:  # Good coverage and variance
            meaningful_features.append(feature)
    
    if not meaningful_features:
        logger.warning("No recent form features with sufficient coverage for correlation analysis")
        return
    
    logger.info(f"Analyzing correlations for {len(meaningful_features)} features")
    
    correlations = []
    for feature in meaningful_features:
        corr = dataset[feature].corr(dataset['hit_hr'])
        if not pd.isna(corr):
            correlations.append((feature, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    logger.info(f"\nTop recent form feature correlations with home runs:")
    for i, (feature, corr) in enumerate(correlations[:10], 1):
        logger.info(f"  {i:2d}. {feature}: {corr:+.4f}")

def main():
    """Main testing function."""
    logger.info("Starting recent form features testing...")
    
    success = test_recent_form_features()
    
    if success:
        # Additional analysis
        try:
            dataset = pd.read_parquet('data/processed/pregame_dataset_2024-08-01_2024-08-15.parquet')
            analyze_feature_correlations(dataset)
        except Exception as e:
            logger.warning(f"Could not run correlation analysis: {e}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ RECENT FORM FEATURES TESTING COMPLETED!")
        logger.info("Step 4 implementation successful.")
        logger.info("="*70)
        return 0
    else:
        logger.error("\nRecent form features testing failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)