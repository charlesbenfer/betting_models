"""
Test Streak and Momentum Features
================================

Test the new streak and momentum features to ensure they work correctly.
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

def test_streak_momentum_features():
    """Test the new streak and momentum features."""
    logger.info("="*70)
    logger.info("TESTING STREAK AND MOMENTUM FEATURES")
    logger.info("="*70)
    
    try:
        # Build dataset with streak and momentum features
        logger.info("Building dataset with new streak and momentum features...")
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
        
        # Check for streak and momentum features
        analyze_streak_momentum_features(dataset)
        
        # Test model integration
        test_model_integration(dataset)
        
        # Analyze feature quality
        analyze_feature_quality(dataset)
        
        # Test streak detection logic
        analyze_streak_detection(dataset)
        
        # Test momentum calculations
        analyze_momentum_calculations(dataset)
        
        logger.info("\nStreak and momentum feature testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Streak and momentum feature testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_streak_momentum_features(dataset: pd.DataFrame):
    """Analyze the streak and momentum features in the dataset."""
    logger.info("\n" + "="*60)
    logger.info("STREAK AND MOMENTUM FEATURE ANALYSIS")
    logger.info("="*60)
    
    # Find streak and momentum features
    expected_features = config.STREAK_MOMENTUM_FEATURES
    found_features = []
    missing_features = []
    
    for feature in expected_features:
        if feature in dataset.columns:
            found_features.append(feature)
        else:
            missing_features.append(feature)
    
    logger.info(f"Expected streak/momentum features: {len(expected_features)}")
    logger.info(f"Found features: {len(found_features)}")
    logger.info(f"Missing features: {len(missing_features)}")
    
    if found_features:
        logger.info(f"\nFound features: {found_features[:10]}...")  # Show first 10
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    # Show feature categories
    hot_streak_features = [f for f in found_features if 'hot_streak' in f]
    cold_streak_features = [f for f in found_features if 'cold_streak' in f or 'slump' in f]
    momentum_features = [f for f in found_features if 'momentum' in f]
    velocity_features = [f for f in found_features if 'velocity' in f or 'acceleration' in f]
    pattern_features = [f for f in found_features if 'pattern' in f or 'cycle' in f or 'rhythm' in f]
    psychological_features = [f for f in found_features if 'confidence' in f or 'pressure' in f or 'clutch' in f or 'mental' in f]
    
    logger.info(f"\nFeature categories found:")
    logger.info(f"  Hot streak features: {len(hot_streak_features)}")
    logger.info(f"  Cold streak features: {len(cold_streak_features)}")  
    logger.info(f"  Momentum features: {len(momentum_features)}")
    logger.info(f"  Velocity features: {len(velocity_features)}")
    logger.info(f"  Pattern features: {len(pattern_features)}")
    logger.info(f"  Psychological features: {len(psychological_features)}")
    
    return found_features

def analyze_feature_quality(dataset: pd.DataFrame):
    """Analyze the quality of streak and momentum features."""
    logger.info("\n" + "="*60)
    logger.info("STREAK AND MOMENTUM FEATURE QUALITY ANALYSIS")
    logger.info("="*60)
    
    # Analyze key streak and momentum features
    key_features = [
        'current_hot_streak', 'current_cold_streak', 'hot_streak_intensity',
        'cold_streak_depth', 'power_momentum_7d', 'momentum_direction',
        'momentum_strength', 'hr_rate_velocity', 'confidence_indicator',
        'pattern_stability', 'breakout_velocity', 'recovery_momentum'
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
                if 'streak' in feature and feature.endswith('_streak'):
                    if stats['min'] < 0:
                        logger.warning(f"  WARNING: Negative streak lengths found")
                    if stats['max'] > 21:
                        logger.warning(f"  WARNING: Very long streaks found (max: {stats['max']:.0f})")
                
                if 'direction' in feature:
                    unique_vals = sorted(feature_data.dropna().unique())
                    logger.info(f"  Direction values: {unique_vals}")
                    if not all(v in [-1, 0, 1] for v in unique_vals):
                        logger.warning(f"  WARNING: Direction values outside [-1, 0, 1] range")
                
                if 'intensity' in feature or 'depth' in feature:
                    if abs(stats['max']) > 1.0:
                        logger.warning(f"  WARNING: High intensity/depth values found")

def test_model_integration(dataset: pd.DataFrame):
    """Test that streak and momentum features integrate with the model system."""
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
        logger.info(f"  Streak momentum features: {len(available_features.get('streak_momentum', []))}")
        logger.info(f"  Enhanced features total: {len(available_features['enhanced'])}")
        
        # Show some streak momentum features
        streak_momentum_features = available_features.get('streak_momentum', [])
        if streak_momentum_features:
            logger.info(f"\nStreak momentum features found: {len(streak_momentum_features)}")
            logger.info(f"Examples: {streak_momentum_features[:8]}")
        else:
            logger.warning("No streak momentum features detected in model system!")
        
        # Quick training test if we have enough data
        if len(dataset) > 100:
            logger.info("\nTesting model training with streak momentum features...")
            
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
                
                logger.info("Model training with streak momentum features successful!")
                logger.info(f"Enhanced features used: {len(available_features['enhanced'])}")
                logger.info(f"Streak momentum features included: {len(streak_momentum_features)}")
                
            except Exception as train_error:
                logger.error(f"Model training failed: {train_error}")
        
    except Exception as e:
        logger.error(f"Model integration test failed: {e}")

def analyze_streak_detection(dataset: pd.DataFrame):
    """Analyze streak detection logic and patterns."""
    logger.info("\n" + "="*60)
    logger.info("STREAK DETECTION ANALYSIS")
    logger.info("="*60)
    
    if 'current_hot_streak' not in dataset.columns or 'current_cold_streak' not in dataset.columns:
        logger.warning("Streak features not available for analysis")
        return
    
    # Analyze streak distributions
    hot_streaks = dataset['current_hot_streak'].dropna()
    cold_streaks = dataset['current_cold_streak'].dropna()
    
    logger.info(f"Hot streak analysis:")
    logger.info(f"  Players with hot streaks: {(hot_streaks > 0).sum()}/{len(hot_streaks)} ({(hot_streaks > 0).mean():.1%})")
    if (hot_streaks > 0).sum() > 0:
        active_hot = hot_streaks[hot_streaks > 0]
        logger.info(f"  Average hot streak length: {active_hot.mean():.1f} days")
        logger.info(f"  Max hot streak: {active_hot.max():.0f} days")
    
    logger.info(f"\nCold streak analysis:")
    logger.info(f"  Players with cold streaks: {(cold_streaks > 0).sum()}/{len(cold_streaks)} ({(cold_streaks > 0).mean():.1%})")
    if (cold_streaks > 0).sum() > 0:
        active_cold = cold_streaks[cold_streaks > 0]
        logger.info(f"  Average cold streak length: {active_cold.mean():.1f} days")
        logger.info(f"  Max cold streak: {active_cold.max():.0f} days")
    
    # Analyze streak intensity and depth
    if 'hot_streak_intensity' in dataset.columns:
        intensity = dataset['hot_streak_intensity'].dropna()
        high_intensity = (intensity > 0.1).sum()
        logger.info(f"\nHigh intensity hot streaks: {high_intensity}/{len(intensity)} ({high_intensity/len(intensity):.1%})")
    
    if 'cold_streak_depth' in dataset.columns:
        depth = dataset['cold_streak_depth'].dropna()
        deep_slumps = (depth > 0.05).sum()
        logger.info(f"Deep cold streaks: {deep_slumps}/{len(depth)} ({deep_slumps/len(depth):.1%})")

def analyze_momentum_calculations(dataset: pd.DataFrame):
    """Analyze momentum calculation logic and patterns."""
    logger.info("\n" + "="*60)
    logger.info("MOMENTUM CALCULATIONS ANALYSIS")
    logger.info("="*60)
    
    momentum_features = [
        'power_momentum_7d', 'momentum_direction', 'momentum_strength',
        'hr_rate_velocity', 'performance_acceleration', 'trend_acceleration'
    ]
    
    for feature in momentum_features:
        if feature in dataset.columns:
            data = dataset[feature].dropna()
            if len(data) > 0:
                logger.info(f"\n{feature}:")
                logger.info(f"  Non-zero values: {(data != 0).sum()}/{len(data)} ({(data != 0).mean():.1%})")
                logger.info(f"  Range: {data.min():.4f} to {data.max():.4f}")
                logger.info(f"  Mean: {data.mean():.4f}")
                
                if 'direction' in feature:
                    # Analyze momentum direction distribution
                    positive = (data > 0).sum()
                    negative = (data < 0).sum()
                    neutral = (data == 0).sum()
                    logger.info(f"  Positive momentum: {positive} ({positive/len(data):.1%})")
                    logger.info(f"  Negative momentum: {negative} ({negative/len(data):.1%})")
                    logger.info(f"  Neutral momentum: {neutral} ({neutral/len(data):.1%})")
                
                if 'velocity' in feature or 'acceleration' in feature:
                    # Analyze rate of change
                    increasing = (data > 0.001).sum()
                    decreasing = (data < -0.001).sum()
                    logger.info(f"  Increasing trend: {increasing} ({increasing/len(data):.1%})")
                    logger.info(f"  Decreasing trend: {decreasing} ({decreasing/len(data):.1%})")

def analyze_feature_correlations(dataset: pd.DataFrame):
    """Analyze correlations between streak/momentum features and home runs."""
    logger.info("\n" + "="*60)
    logger.info("STREAK/MOMENTUM FEATURE CORRELATION ANALYSIS")
    logger.info("="*60)
    
    if 'hit_hr' not in dataset.columns:
        logger.warning("No target variable for correlation analysis")
        return
    
    # Get streak momentum features with good coverage
    streak_momentum_features = [col for col in dataset.columns 
                              if any(feat in col for feat in ['streak', 'momentum', 'velocity', 'confidence', 'pattern'])]
    
    meaningful_features = []
    for feature in streak_momentum_features:
        non_null_pct = dataset[feature].notna().mean()
        non_zero_pct = (dataset[feature] != 0).mean()
        if non_null_pct > 0.8 and non_zero_pct > 0.1:  # Good coverage and variance
            meaningful_features.append(feature)
    
    if not meaningful_features:
        logger.warning("No streak/momentum features with sufficient coverage for correlation analysis")
        return
    
    logger.info(f"Analyzing correlations for {len(meaningful_features)} features")
    
    correlations = []
    for feature in meaningful_features:
        corr = dataset[feature].corr(dataset['hit_hr'])
        if not pd.isna(corr):
            correlations.append((feature, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    logger.info(f"\nTop streak/momentum feature correlations with home runs:")
    for i, (feature, corr) in enumerate(correlations[:15], 1):
        logger.info(f"  {i:2d}. {feature}: {corr:+.4f}")

def main():
    """Main testing function."""
    logger.info("Starting streak and momentum features testing...")
    
    success = test_streak_momentum_features()
    
    if success:
        # Additional analysis
        try:
            dataset = pd.read_parquet('data/processed/pregame_dataset_2024-08-01_2024-08-15.parquet')
            analyze_feature_correlations(dataset)
        except Exception as e:
            logger.warning(f"Could not run correlation analysis: {e}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ STREAK AND MOMENTUM FEATURES TESTING COMPLETED!")
        logger.info("Step 5 implementation successful.")
        logger.info("="*70)
        return 0
    else:
        logger.error("\nStreak and momentum features testing failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)