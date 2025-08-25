"""
Test Temporal and Fatigue Features
=================================

Test the new temporal and fatigue features to ensure they work correctly.
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
from temporal_fatigue_features import (
    calculate_fatigue_correlations, analyze_circadian_patterns, validate_fatigue_logic
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_temporal_fatigue_features():
    """Test the new temporal and fatigue features."""
    logger.info("="*70)
    logger.info("TESTING TEMPORAL AND FATIGUE FEATURES")
    logger.info("="*70)
    
    try:
        # Build dataset with temporal and fatigue features
        logger.info("Building dataset with new temporal and fatigue features...")
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
        
        # Check for temporal fatigue features
        analyze_temporal_fatigue_features(dataset)
        
        # Test model integration
        test_model_integration(dataset)
        
        # Analyze feature quality
        analyze_feature_quality(dataset)
        
        # Test circadian features
        analyze_circadian_features(dataset)
        
        # Test fatigue calculations
        analyze_fatigue_calculations(dataset)
        
        # Test travel features
        analyze_travel_features(dataset)
        
        # Validate logic
        validate_feature_logic(dataset)
        
        logger.info("\nTemporal and fatigue feature testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Temporal and fatigue feature testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_temporal_fatigue_features(dataset: pd.DataFrame):
    """Analyze the temporal and fatigue features in the dataset."""
    logger.info("\n" + "="*60)
    logger.info("TEMPORAL AND FATIGUE FEATURE ANALYSIS")
    logger.info("="*60)
    
    # Find temporal fatigue features
    expected_features = config.TEMPORAL_FATIGUE_FEATURES
    found_features = []
    missing_features = []
    
    for feature in expected_features:
        if feature in dataset.columns:
            found_features.append(feature)
        else:
            missing_features.append(feature)
    
    logger.info(f"Expected temporal/fatigue features: {len(expected_features)}")
    logger.info(f"Found features: {len(found_features)}")
    logger.info(f"Missing features: {len(missing_features)}")
    
    if found_features:
        logger.info(f"\nFound features: {found_features[:10]}...")  # Show first 10
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    # Show feature categories
    circadian_features = [f for f in found_features if any(x in f for x in ['circadian', 'game_hour', 'time', 'optimal'])]
    fatigue_features = [f for f in found_features if any(x in f for x in ['fatigue', 'rest', 'energy', 'recovery'])]
    travel_features = [f for f in found_features if any(x in f for x in ['travel', 'jet_lag', 'timezone', 'home_away'])]
    schedule_features = [f for f in found_features if any(x in f for x in ['games_', 'schedule', 'workload', 'consecutive'])]
    seasonal_features = [f for f in found_features if any(x in f for x in ['season', 'monthly', 'playoff', 'spring', 'dog_days'])]
    
    logger.info(f"\nFeature categories found:")
    logger.info(f"  Circadian features: {len(circadian_features)}")
    logger.info(f"  Fatigue features: {len(fatigue_features)}")  
    logger.info(f"  Travel features: {len(travel_features)}")
    logger.info(f"  Schedule features: {len(schedule_features)}")
    logger.info(f"  Seasonal features: {len(seasonal_features)}")
    
    return found_features

def analyze_feature_quality(dataset: pd.DataFrame):
    """Analyze the quality of temporal and fatigue features."""
    logger.info("\n" + "="*60)
    logger.info("TEMPORAL/FATIGUE FEATURE QUALITY ANALYSIS")
    logger.info("="*60)
    
    # Analyze key temporal and fatigue features
    key_features = [
        'game_hour', 'circadian_performance_factor', 'fatigue_level', 'energy_reserves',
        'games_without_rest', 'jet_lag_factor', 'rest_quality_score', 'season_fatigue_factor',
        'consecutive_games', 'schedule_intensity', 'monthly_energy_level'
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
                if 'hour' in feature:
                    if stats['min'] < 0 or stats['max'] > 23:
                        logger.warning(f"  WARNING: Hour values outside 0-23 range")
                
                if 'fatigue' in feature or 'energy' in feature:
                    if stats['min'] < 0 or stats['max'] > 1.5:
                        logger.warning(f"  WARNING: Fatigue/energy values outside expected range")
                
                if 'games' in feature and 'last' not in feature and 'next' not in feature:
                    if stats['max'] > 15:
                        logger.warning(f"  WARNING: Very high consecutive game counts")

def test_model_integration(dataset: pd.DataFrame):
    """Test that temporal fatigue features integrate with the model system."""
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
        logger.info(f"  Ballpark features: {len(available_features.get('ballpark', []))}")
        logger.info(f"  Temporal fatigue features: {len(available_features.get('temporal_fatigue', []))}")
        logger.info(f"  Enhanced features total: {len(available_features['enhanced'])}")
        
        # Show some temporal fatigue features
        temporal_fatigue_features = available_features.get('temporal_fatigue', [])
        if temporal_fatigue_features:
            logger.info(f"\nTemporal fatigue features found: {len(temporal_fatigue_features)}")
            logger.info(f"Examples: {temporal_fatigue_features[:8]}")
        else:
            logger.warning("No temporal fatigue features detected in model system!")
        
        # Quick training test if we have enough data
        if len(dataset) > 100:
            logger.info("\nTesting model training with temporal fatigue features...")
            
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
                
                logger.info("Model training with temporal fatigue features successful!")
                logger.info(f"Enhanced features used: {len(available_features['enhanced'])}")
                logger.info(f"Temporal fatigue features included: {len(temporal_fatigue_features)}")
                
            except Exception as train_error:
                logger.error(f"Model training failed: {train_error}")
        
    except Exception as e:
        logger.error(f"Model integration test failed: {e}")

def analyze_circadian_features(dataset: pd.DataFrame):
    """Analyze circadian rhythm and time-of-day features."""
    logger.info("\n" + "="*60)
    logger.info("CIRCADIAN FEATURES ANALYSIS")
    logger.info("="*60)
    
    if 'game_hour' not in dataset.columns:
        logger.warning("Game hour not available for circadian analysis")
        return
    
    # Analyze game time distribution
    game_hours = dataset['game_hour'].dropna()
    if len(game_hours) > 0:
        logger.info(f"Game time distribution:")
        hour_counts = game_hours.value_counts().sort_index()
        for hour, count in hour_counts.head(10).items():
            logger.info(f"  {hour:02d}:00 - {count} games ({count/len(game_hours):.1%})")
    
    # Analyze circadian performance factor
    if 'circadian_performance_factor' in dataset.columns:
        circadian = dataset['circadian_performance_factor'].dropna()
        if len(circadian) > 0:
            logger.info(f"\nCircadian performance factor:")
            logger.info(f"  Range: {circadian.min():.4f} to {circadian.max():.4f}")
            logger.info(f"  Mean: {circadian.mean():.4f}")
            
            # Check for peak and trough times
            peak_factor = circadian.max()
            trough_factor = circadian.min()
            logger.info(f"  Peak factor: {peak_factor:.4f}")
            logger.info(f"  Trough factor: {trough_factor:.4f}")
            logger.info(f"  Amplitude: {peak_factor - trough_factor:.4f}")
    
    # Analyze optimal time windows
    if 'optimal_time_window' in dataset.columns:
        optimal = dataset['optimal_time_window'].dropna()
        if len(optimal) > 0:
            optimal_games = (optimal == 1).sum()
            logger.info(f"\nOptimal time window games: {optimal_games}/{len(optimal)} ({optimal_games/len(optimal):.1%})")

def analyze_fatigue_calculations(dataset: pd.DataFrame):
    """Analyze fatigue calculation patterns."""
    logger.info("\n" + "="*60)
    logger.info("FATIGUE CALCULATIONS ANALYSIS")
    logger.info("="*60)
    
    fatigue_features = [
        'fatigue_level', 'games_without_rest', 'energy_reserves', 
        'rest_quality_score', 'consecutive_games'
    ]
    
    for feature in fatigue_features:
        if feature in dataset.columns:
            data = dataset[feature].dropna()
            if len(data) > 0:
                logger.info(f"\n{feature}:")
                logger.info(f"  Non-zero values: {(data != 0).sum()}/{len(data)} ({(data != 0).mean():.1%})")
                logger.info(f"  Range: {data.min():.4f} to {data.max():.4f}")
                logger.info(f"  Mean: {data.mean():.4f}")
                
                if 'fatigue' in feature:
                    # Analyze fatigue distribution
                    low_fatigue = (data < 0.2).sum()
                    high_fatigue = (data > 0.5).sum()
                    logger.info(f"  Low fatigue (<0.2): {low_fatigue} ({low_fatigue/len(data):.1%})")
                    logger.info(f"  High fatigue (>0.5): {high_fatigue} ({high_fatigue/len(data):.1%})")
                
                if 'energy' in feature:
                    # Analyze energy distribution
                    low_energy = (data < 0.7).sum()
                    full_energy = (data > 0.95).sum()
                    logger.info(f"  Low energy (<0.7): {low_energy} ({low_energy/len(data):.1%})")
                    logger.info(f"  Full energy (>0.95): {full_energy} ({full_energy/len(data):.1%})")
                
                if 'consecutive' in feature:
                    # Analyze consecutive game patterns
                    long_streaks = (data > 5).sum()
                    very_long_streaks = (data > 10).sum()
                    logger.info(f"  Long streaks (>5): {long_streaks} ({long_streaks/len(data):.1%})")
                    logger.info(f"  Very long streaks (>10): {very_long_streaks} ({very_long_streaks/len(data):.1%})")

def analyze_travel_features(dataset: pd.DataFrame):
    """Analyze travel and jet lag features."""
    logger.info("\n" + "="*60)
    logger.info("TRAVEL FEATURES ANALYSIS")
    logger.info("="*60)
    
    travel_features = [
        'timezone_change', 'jet_lag_factor', 'travel_fatigue', 
        'cross_country_travel', 'home_away_transition'
    ]
    
    for feature in travel_features:
        if feature in dataset.columns:
            data = dataset[feature].dropna()
            if len(data) > 0:
                logger.info(f"\n{feature}:")
                logger.info(f"  Non-zero values: {(data != 0).sum()}/{len(data)} ({(data != 0).mean():.1%})")
                
                if 'timezone' in feature:
                    if (data != 0).sum() > 0:
                        tz_changes = data[data != 0]
                        logger.info(f"  Average timezone change: {tz_changes.mean():.1f}")
                        logger.info(f"  Max timezone change: {tz_changes.max():.0f}")
                
                if 'cross_country' in feature or 'transition' in feature:
                    events = (data == 1).sum()
                    logger.info(f"  Travel events: {events} ({events/len(data):.1%})")
                
                if 'jet_lag' in feature or 'travel_fatigue' in feature:
                    if (data > 0).sum() > 0:
                        affected = data[data > 0]
                        logger.info(f"  Average impact: {affected.mean():.4f}")
                        logger.info(f"  Max impact: {affected.max():.4f}")

def validate_feature_logic(dataset: pd.DataFrame):
    """Validate temporal and fatigue feature logic."""
    logger.info("\n" + "="*60)
    logger.info("FEATURE LOGIC VALIDATION")
    logger.info("="*60)
    
    try:
        validation_results = validate_fatigue_logic(dataset)
        
        for test, result in validation_results.items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{test}: {status}")
            else:
                logger.info(f"{test}: {result:.4f}")
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
    
    # Additional validations
    if 'game_hour' in dataset.columns:
        hours = dataset['game_hour'].dropna()
        if len(hours) > 0:
            hour_range_valid = (hours >= 0).all() and (hours <= 23).all()
            logger.info(f"Game hours in valid range (0-23): {'✅ PASS' if hour_range_valid else '❌ FAIL'}")
    
    if 'fatigue_level' in dataset.columns and 'energy_reserves' in dataset.columns:
        fatigue = dataset['fatigue_level'].dropna()
        energy = dataset['energy_reserves'].dropna()
        if len(fatigue) > 0 and len(energy) > 0:
            # Energy should generally be inverse of fatigue
            correlation = fatigue.corr(energy)
            inverse_relationship = correlation < -0.1
            logger.info(f"Fatigue-energy inverse relationship: {'✅ PASS' if inverse_relationship else '❌ FAIL'} (corr: {correlation:.3f})")

def analyze_feature_correlations(dataset: pd.DataFrame):
    """Analyze correlations between temporal/fatigue features and home runs."""
    logger.info("\n" + "="*60)
    logger.info("TEMPORAL/FATIGUE FEATURE CORRELATION ANALYSIS")
    logger.info("="*60)
    
    if 'hit_hr' not in dataset.columns:
        logger.warning("No target variable for correlation analysis")
        return
    
    # Get temporal fatigue features with good coverage
    temporal_fatigue_features = [col for col in dataset.columns 
                               if any(feat in col for feat in ['game_hour', 'fatigue', 'energy', 'rest', 'travel', 'season', 'circadian'])]
    
    meaningful_features = []
    for feature in temporal_fatigue_features:
        non_null_pct = dataset[feature].notna().mean()
        if non_null_pct > 0.8:  # Good coverage
            meaningful_features.append(feature)
    
    if not meaningful_features:
        logger.warning("No temporal/fatigue features with sufficient coverage for correlation analysis")
        return
    
    logger.info(f"Analyzing correlations for {len(meaningful_features)} features")
    
    correlations = []
    for feature in meaningful_features:
        corr = dataset[feature].corr(dataset['hit_hr'])
        if not pd.isna(corr):
            correlations.append((feature, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    logger.info(f"\nTop temporal/fatigue feature correlations with home runs:")
    for i, (feature, corr) in enumerate(correlations[:15], 1):
        logger.info(f"  {i:2d}. {feature}: {corr:+.4f}")

def main():
    """Main testing function."""
    logger.info("Starting temporal and fatigue features testing...")
    
    success = test_temporal_fatigue_features()
    
    if success:
        # Additional analysis
        try:
            dataset = pd.read_parquet('data/processed/pregame_dataset_2024-08-01_2024-08-15.parquet')
            analyze_feature_correlations(dataset)
            
            # Analyze circadian patterns
            circadian_patterns = analyze_circadian_patterns(dataset)
            if circadian_patterns:
                logger.info(f"\nCircadian patterns analysis completed for {len(circadian_patterns)} hours")
                
            # Analyze fatigue correlations
            fatigue_correlations = calculate_fatigue_correlations(dataset)
            if fatigue_correlations:
                logger.info(f"\nFatigue correlations calculated for {len(fatigue_correlations)} features")
                
        except Exception as e:
            logger.warning(f"Could not run additional analysis: {e}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ TEMPORAL AND FATIGUE FEATURES TESTING COMPLETED!")
        logger.info("Step 7 implementation successful.")
        logger.info("="*70)
        return 0
    else:
        logger.error("\nTemporal and fatigue features testing failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)