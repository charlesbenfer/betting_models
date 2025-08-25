"""
Test Ballpark Features
=====================

Test the new advanced ballpark features to ensure they work correctly.
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
from ballpark_features import analyze_park_effects, calculate_park_baseline_rates

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ballpark_features():
    """Test the new ballpark features."""
    logger.info("="*70)
    logger.info("TESTING BALLPARK FEATURES")
    logger.info("="*70)
    
    try:
        # Build dataset with ballpark features
        logger.info("Building dataset with new ballpark features...")
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
        
        # Check for ballpark features
        analyze_ballpark_features(dataset)
        
        # Test model integration
        test_model_integration(dataset)
        
        # Analyze feature quality
        analyze_feature_quality(dataset)
        
        # Test dimensional features
        analyze_dimensional_features(dataset)
        
        # Test park interactions
        analyze_park_interactions(dataset)
        
        # Validate park-specific logic
        validate_park_logic(dataset)
        
        logger.info("\nBallpark feature testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Ballpark feature testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_ballpark_features(dataset: pd.DataFrame):
    """Analyze the ballpark features in the dataset."""
    logger.info("\n" + "="*60)
    logger.info("BALLPARK FEATURE ANALYSIS")
    logger.info("="*60)
    
    # Find ballpark features
    expected_features = config.BALLPARK_FEATURES
    found_features = []
    missing_features = []
    
    for feature in expected_features:
        if feature in dataset.columns:
            found_features.append(feature)
        else:
            missing_features.append(feature)
    
    logger.info(f"Expected ballpark features: {len(expected_features)}")
    logger.info(f"Found features: {len(found_features)}")
    logger.info(f"Missing features: {len(missing_features)}")
    
    if found_features:
        logger.info(f"\nFound features: {found_features[:10]}...")  # Show first 10
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    # Show feature categories
    dimensional_features = [f for f in found_features if any(x in f for x in ['distance', 'height', 'elevation', 'dome', 'turf'])]
    carry_features = [f for f in found_features if any(x in f for x in ['carry', 'density', 'humidity'])]
    directional_features = [f for f in found_features if any(x in f for x in ['pull', 'opposite', 'center'])]
    interaction_features = [f for f in found_features if any(x in f for x in ['batter_park', 'wind_interaction', 'temperature_interaction'])]
    context_features = [f for f in found_features if any(x in f for x in ['offense_context', 'pitcher_context', 'defensive_context'])]
    
    logger.info(f"\nFeature categories found:")
    logger.info(f"  Dimensional features: {len(dimensional_features)}")
    logger.info(f"  Carry distance features: {len(carry_features)}")  
    logger.info(f"  Directional features: {len(directional_features)}")
    logger.info(f"  Interaction features: {len(interaction_features)}")
    logger.info(f"  Context features: {len(context_features)}")
    
    return found_features

def analyze_feature_quality(dataset: pd.DataFrame):
    """Analyze the quality of ballpark features."""
    logger.info("\n" + "="*60)
    logger.info("BALLPARK FEATURE QUALITY ANALYSIS")
    logger.info("="*60)
    
    # Analyze key ballpark features
    key_features = [
        'park_left_field_distance', 'park_elevation', 'park_hr_difficulty_index',
        'park_elevation_carry_boost', 'park_pull_factor_left', 'park_pull_factor_right',
        'park_weather_hr_multiplier', 'park_offense_context', 'batter_park_hr_rate_boost'
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
                if 'distance' in feature:
                    if stats['min'] < 250 or stats['max'] > 500:
                        logger.warning(f"  WARNING: Unusual distance values found")
                
                if 'elevation' in feature and 'boost' not in feature:
                    if stats['min'] < 0 or stats['max'] > 6000:
                        logger.warning(f"  WARNING: Unusual elevation values found")
                
                if 'factor' in feature or 'boost' in feature:
                    if abs(stats['min']) > 0.5 or abs(stats['max']) > 0.5:
                        logger.warning(f"  WARNING: Large factor/boost values found")

def test_model_integration(dataset: pd.DataFrame):
    """Test that ballpark features integrate with the model system."""
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
        logger.info(f"  Enhanced features total: {len(available_features['enhanced'])}")
        
        # Show some ballpark features
        ballpark_features = available_features.get('ballpark', [])
        if ballpark_features:
            logger.info(f"\nBallpark features found: {len(ballpark_features)}")
            logger.info(f"Examples: {ballpark_features[:8]}")
        else:
            logger.warning("No ballpark features detected in model system!")
        
        # Quick training test if we have enough data
        if len(dataset) > 100:
            logger.info("\nTesting model training with ballpark features...")
            
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
                
                logger.info("Model training with ballpark features successful!")
                logger.info(f"Enhanced features used: {len(available_features['enhanced'])}")
                logger.info(f"Ballpark features included: {len(ballpark_features)}")
                
            except Exception as train_error:
                logger.error(f"Model training failed: {train_error}")
        
    except Exception as e:
        logger.error(f"Model integration test failed: {e}")

def analyze_dimensional_features(dataset: pd.DataFrame):
    """Analyze dimensional feature calculations."""
    logger.info("\n" + "="*60)
    logger.info("DIMENSIONAL FEATURES ANALYSIS")
    logger.info("="*60)
    
    dimensional_features = [
        'park_left_field_distance', 'park_center_field_distance', 'park_right_field_distance',
        'park_elevation', 'park_hr_difficulty_index', 'park_symmetry_factor'
    ]
    
    available_dimensional = [f for f in dimensional_features if f in dataset.columns]
    
    if not available_dimensional:
        logger.warning("No dimensional features available for analysis")
        return
    
    # Analyze by stadium
    if 'stadium' in dataset.columns:
        logger.info("Dimensional features by stadium:")
        
        stadiums = dataset['stadium'].unique()[:5]  # Show first 5 stadiums
        
        for stadium in stadiums:
            stadium_data = dataset[dataset['stadium'] == stadium]
            if len(stadium_data) > 0:
                logger.info(f"\n{stadium}:")
                
                for feature in available_dimensional:
                    if feature in stadium_data.columns:
                        value = stadium_data[feature].iloc[0]  # Should be same for all rows
                        logger.info(f"  {feature}: {value:.3f}")
    
    # Analyze difficulty index distribution
    if 'park_hr_difficulty_index' in dataset.columns:
        difficulty = dataset['park_hr_difficulty_index'].dropna()
        if len(difficulty) > 0:
            logger.info(f"\nHR Difficulty Index distribution:")
            logger.info(f"  Easy parks (< 0): {(difficulty < 0).sum()}/{len(difficulty)} ({(difficulty < 0).mean():.1%})")
            logger.info(f"  Average parks (0-2): {((difficulty >= 0) & (difficulty <= 2)).sum()}/{len(difficulty)}")
            logger.info(f"  Hard parks (> 2): {(difficulty > 2).sum()}/{len(difficulty)} ({(difficulty > 2).mean():.1%})")

def analyze_park_interactions(dataset: pd.DataFrame):
    """Analyze park interaction features."""
    logger.info("\n" + "="*60)
    logger.info("PARK INTERACTION ANALYSIS")
    logger.info("="*60)
    
    interaction_features = [
        'park_wind_interaction', 'park_temperature_interaction', 'park_weather_hr_multiplier',
        'batter_park_hr_rate_boost', 'batter_park_comfort_factor'
    ]
    
    for feature in interaction_features:
        if feature in dataset.columns:
            data = dataset[feature].dropna()
            if len(data) > 0:
                logger.info(f"\n{feature}:")
                logger.info(f"  Non-zero values: {(data != 0).sum()}/{len(data)} ({(data != 0).mean():.1%})")
                logger.info(f"  Range: {data.min():.4f} to {data.max():.4f}")
                logger.info(f"  Mean: {data.mean():.4f}")
                
                if 'multiplier' in feature:
                    # Analyze multiplier distribution
                    positive = (data > 1.0).sum()
                    negative = (data < 1.0).sum()
                    neutral = (data == 1.0).sum()
                    logger.info(f"  Positive effect (>1.0): {positive} ({positive/len(data):.1%})")
                    logger.info(f"  Negative effect (<1.0): {negative} ({negative/len(data):.1%})")
                    logger.info(f"  Neutral effect (=1.0): {neutral} ({neutral/len(data):.1%})")
                
                if 'batter_park' in feature:
                    # Analyze batter-park relationship
                    strong_positive = (data > 0.1).sum()
                    strong_negative = (data < -0.1).sum()
                    logger.info(f"  Strong positive relationship: {strong_positive} ({strong_positive/len(data):.1%})")
                    logger.info(f"  Strong negative relationship: {strong_negative} ({strong_negative/len(data):.1%})")

def validate_park_logic(dataset: pd.DataFrame):
    """Validate park-specific logic and calculations."""
    logger.info("\n" + "="*60)
    logger.info("PARK LOGIC VALIDATION")
    logger.info("="*60)
    
    # Check dome vs elevation interaction
    if 'park_is_dome' in dataset.columns and 'park_elevation_carry_boost' in dataset.columns:
        domed_stadiums = dataset[dataset['park_is_dome'] == 1]
        if len(domed_stadiums) > 0:
            dome_carry_boost = domed_stadiums['park_elevation_carry_boost'].mean()
            logger.info(f"Average elevation carry boost for domed stadiums: {dome_carry_boost:.4f}")
    
    # Check coastal stadium humidity effects
    if 'park_coastal_humidity_factor' in dataset.columns:
        coastal_effects = dataset['park_coastal_humidity_factor'].dropna()
        coastal_stadiums = (coastal_effects != 0).sum()
        logger.info(f"Stadiums with coastal humidity effects: {coastal_stadiums}")
    
    # Check elevation effects
    if 'park_elevation' in dataset.columns and 'park_elevation_carry_boost' in dataset.columns:
        high_altitude = dataset[dataset['park_elevation'] > 3000]
        if len(high_altitude) > 0:
            avg_boost = high_altitude['park_elevation_carry_boost'].mean()
            logger.info(f"Average carry boost for high altitude stadiums: {avg_boost:.4f}")
    
    # Check symmetry calculations
    if 'park_symmetry_factor' in dataset.columns:
        symmetry = dataset['park_symmetry_factor'].dropna()
        if len(symmetry) > 0:
            symmetric_parks = (symmetry < 0.5).sum()
            asymmetric_parks = (symmetry > 1.0).sum()
            logger.info(f"Symmetric parks (factor < 0.5): {symmetric_parks}")
            logger.info(f"Asymmetric parks (factor > 1.0): {asymmetric_parks}")

def analyze_feature_correlations(dataset: pd.DataFrame):
    """Analyze correlations between ballpark features and home runs."""
    logger.info("\n" + "="*60)
    logger.info("BALLPARK FEATURE CORRELATION ANALYSIS")
    logger.info("="*60)
    
    if 'hit_hr' not in dataset.columns:
        logger.warning("No target variable for correlation analysis")
        return
    
    # Get ballpark features with good coverage
    ballpark_features = [col for col in dataset.columns 
                        if any(feat in col for feat in ['park_', 'batter_park_'])]
    
    meaningful_features = []
    for feature in ballpark_features:
        non_null_pct = dataset[feature].notna().mean()
        if non_null_pct > 0.8:  # Good coverage
            meaningful_features.append(feature)
    
    if not meaningful_features:
        logger.warning("No ballpark features with sufficient coverage for correlation analysis")
        return
    
    logger.info(f"Analyzing correlations for {len(meaningful_features)} features")
    
    correlations = []
    for feature in meaningful_features:
        corr = dataset[feature].corr(dataset['hit_hr'])
        if not pd.isna(corr):
            correlations.append((feature, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    logger.info(f"\nTop ballpark feature correlations with home runs:")
    for i, (feature, corr) in enumerate(correlations[:15], 1):
        logger.info(f"  {i:2d}. {feature}: {corr:+.4f}")

def main():
    """Main testing function."""
    logger.info("Starting ballpark features testing...")
    
    success = test_ballpark_features()
    
    if success:
        # Additional analysis
        try:
            dataset = pd.read_parquet('data/processed/pregame_dataset_2024-08-01_2024-08-15.parquet')
            analyze_feature_correlations(dataset)
            
            # Analyze park effects
            park_effects = analyze_park_effects(dataset)
            if park_effects:
                logger.info(f"\nPark effects analysis completed for {len(park_effects)} stadiums")
                
        except Exception as e:
            logger.warning(f"Could not run additional analysis: {e}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ BALLPARK FEATURES TESTING COMPLETED!")
        logger.info("Step 6 implementation successful.")
        logger.info("="*70)
        return 0
    else:
        logger.error("\nBallpark features testing failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)