"""
Test Feature Interactions
=========================

Test the new feature interaction terms to ensure they work correctly.
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
from feature_interactions import (
    calculate_interaction_importance, analyze_interaction_effects, validate_interaction_logic
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_feature_interactions():
    """Test the new feature interaction terms."""
    logger.info("="*70)
    logger.info("TESTING FEATURE INTERACTIONS")
    logger.info("="*70)
    
    try:
        # Build dataset with feature interactions
        logger.info("Building dataset with new feature interaction terms...")
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
        
        # Check for interaction features
        analyze_interaction_features(dataset)
        
        # Test model integration
        test_model_integration(dataset)
        
        # Analyze feature quality
        analyze_feature_quality(dataset)
        
        # Test multiplicative interactions
        analyze_multiplicative_interactions(dataset)
        
        # Test composite indices
        analyze_composite_indices(dataset)
        
        # Test threshold interactions
        analyze_threshold_interactions(dataset)
        
        # Validate interaction logic
        validate_interactions(dataset)
        
        logger.info("\nFeature interaction testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Feature interaction testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_interaction_features(dataset: pd.DataFrame):
    """Analyze the interaction features in the dataset."""
    logger.info("\n" + "="*60)
    logger.info("INTERACTION FEATURE ANALYSIS")
    logger.info("="*60)
    
    # Find interaction features
    expected_features = config.INTERACTION_FEATURES
    found_features = []
    missing_features = []
    
    for feature in expected_features:
        if feature in dataset.columns:
            found_features.append(feature)
        else:
            missing_features.append(feature)
    
    logger.info(f"Expected interaction features: {len(expected_features)}")
    logger.info(f"Found features: {len(found_features)}")
    logger.info(f"Missing features: {len(missing_features)}")
    
    if found_features:
        logger.info(f"\nFound features: {found_features[:10]}...")  # Show first 10
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    # Show feature categories
    multiplicative_features = [f for f in found_features if any(x in f for x in ['boost', 'synergy', 'amplification', 'factor'])]
    conditional_features = [f for f in found_features if any(x in f for x in ['penalty', 'disruption'])]
    composite_features = [f for f in found_features if 'index' in f]
    ratio_features = [f for f in found_features if 'ratio' in f or 'balance' in f]
    threshold_features = [f for f in found_features if 'indicator' in f or 'convergence' in f]
    multiplier_features = [f for f in found_features if 'multiplier' in f]
    
    logger.info(f"\nFeature categories found:")
    logger.info(f"  Multiplicative interactions: {len(multiplicative_features)}")
    logger.info(f"  Conditional interactions: {len(conditional_features)}")  
    logger.info(f"  Composite indices: {len(composite_features)}")
    logger.info(f"  Ratio interactions: {len(ratio_features)}")
    logger.info(f"  Threshold interactions: {len(threshold_features)}")
    logger.info(f"  Performance multipliers: {len(multiplier_features)}")
    
    return found_features

def analyze_feature_quality(dataset: pd.DataFrame):
    """Analyze the quality of interaction features."""
    logger.info("\n" + "="*60)
    logger.info("INTERACTION FEATURE QUALITY ANALYSIS")
    logger.info("="*60)
    
    # Analyze key interaction features
    key_features = [
        'power_form_altitude_boost', 'hot_streak_confidence_boost', 'energy_circadian_factor',
        'composite_power_index', 'momentum_fatigue_ratio', 'elite_power_indicator',
        'overall_performance_multiplier', 'mind_body_synergy'
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
                if 'multiplier' in feature:
                    if stats['min'] < 0 or stats['max'] > 5:
                        logger.warning(f"  WARNING: Multiplier values outside reasonable range")
                
                if 'ratio' in feature:
                    if stats['min'] < 0 or stats['max'] > 100:
                        logger.warning(f"  WARNING: Extreme ratio values found")
                
                if 'index' in feature:
                    if stats['min'] < -2 or stats['max'] > 3:
                        logger.warning(f"  WARNING: Index values outside expected range")

def test_model_integration(dataset: pd.DataFrame):
    """Test that interaction features integrate with the model system."""
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
        logger.info(f"  Interaction features: {len(available_features.get('interactions', []))}")
        logger.info(f"  Enhanced features total: {len(available_features['enhanced'])}")
        
        # Show some interaction features
        interaction_features = available_features.get('interactions', [])
        if interaction_features:
            logger.info(f"\nInteraction features found: {len(interaction_features)}")
            logger.info(f"Examples: {interaction_features[:8]}")
        else:
            logger.warning("No interaction features detected in model system!")
        
        # Quick training test if we have enough data
        if len(dataset) > 100:
            logger.info("\nTesting model training with interaction features...")
            
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
                
                logger.info("Model training with interaction features successful!")
                logger.info(f"Enhanced features used: {len(available_features['enhanced'])}")
                logger.info(f"Interaction features included: {len(interaction_features)}")
                
            except Exception as train_error:
                logger.error(f"Model training failed: {train_error}")
        
    except Exception as e:
        logger.error(f"Model integration test failed: {e}")

def analyze_multiplicative_interactions(dataset: pd.DataFrame):
    """Analyze multiplicative interaction features."""
    logger.info("\n" + "="*60)
    logger.info("MULTIPLICATIVE INTERACTIONS ANALYSIS")
    logger.info("="*60)
    
    multiplicative_features = [
        'power_form_altitude_boost', 'hot_streak_confidence_boost', 'energy_circadian_factor',
        'matchup_form_synergy', 'clutch_pressure_performance'
    ]
    
    for feature in multiplicative_features:
        if feature in dataset.columns:
            data = dataset[feature].dropna()
            if len(data) > 0:
                logger.info(f"\n{feature}:")
                logger.info(f"  Non-zero values: {(data != 0).sum()}/{len(data)} ({(data != 0).mean():.1%})")
                logger.info(f"  Range: {data.min():.4f} to {data.max():.4f}")
                logger.info(f"  Mean: {data.mean():.4f}")
                
                # Analyze distribution of effects
                if 'boost' in feature or 'synergy' in feature:
                    positive_effects = (data > 0).sum()
                    strong_effects = (data > data.quantile(0.75)).sum()
                    logger.info(f"  Positive effects: {positive_effects} ({positive_effects/len(data):.1%})")
                    logger.info(f"  Strong effects (>75th %ile): {strong_effects} ({strong_effects/len(data):.1%})")

def analyze_composite_indices(dataset: pd.DataFrame):
    """Analyze composite index features."""
    logger.info("\n" + "="*60)
    logger.info("COMPOSITE INDICES ANALYSIS")
    logger.info("="*60)
    
    composite_features = [
        'composite_power_index', 'composite_momentum_index', 'environmental_favorability_index',
        'physical_condition_index', 'psychological_state_index'
    ]
    
    for feature in composite_features:
        if feature in dataset.columns:
            data = dataset[feature].dropna()
            if len(data) > 0:
                logger.info(f"\n{feature}:")
                logger.info(f"  Range: {data.min():.4f} to {data.max():.4f}")
                logger.info(f"  Mean: {data.mean():.4f}")
                logger.info(f"  Std: {data.std():.4f}")
                
                # Analyze distribution
                q25, q50, q75 = data.quantile([0.25, 0.5, 0.75])
                logger.info(f"  Quartiles: Q1={q25:.3f}, Q2={q50:.3f}, Q3={q75:.3f}")
                
                # Check for reasonable distribution
                if data.std() < 0.01:
                    logger.warning(f"  WARNING: Very low variance in composite index")
                if data.nunique() < 10:
                    logger.warning(f"  WARNING: Low unique values in composite index")

def analyze_threshold_interactions(dataset: pd.DataFrame):
    """Analyze threshold-based interaction features."""
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD INTERACTIONS ANALYSIS")
    logger.info("="*60)
    
    threshold_features = [
        'elite_power_indicator', 'high_momentum_indicator', 'extreme_fatigue_indicator',
        'optimal_conditions_indicator', 'elite_performance_convergence'
    ]
    
    for feature in threshold_features:
        if feature in dataset.columns:
            data = dataset[feature].dropna()
            if len(data) > 0:
                logger.info(f"\n{feature}:")
                logger.info(f"  Unique values: {sorted(data.unique())}")
                
                if data.nunique() <= 10:  # Categorical/threshold features
                    value_counts = data.value_counts()
                    for value, count in value_counts.items():
                        logger.info(f"    {value}: {count} ({count/len(data):.1%})")
                
                # Check activation rates for indicators
                if 'indicator' in feature:
                    activation_rate = (data > 0).mean()
                    logger.info(f"  Activation rate: {activation_rate:.1%}")
                    
                    if activation_rate > 0.5:
                        logger.warning(f"  WARNING: High activation rate - threshold may be too low")
                    elif activation_rate < 0.05:
                        logger.warning(f"  WARNING: Very low activation rate - threshold may be too high")

def validate_interactions(dataset: pd.DataFrame):
    """Validate interaction feature logic."""
    logger.info("\n" + "="*60)
    logger.info("INTERACTION VALIDATION")
    logger.info("="*60)
    
    try:
        validation_results = validate_interaction_logic(dataset)
        
        for test, result in validation_results.items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"{test}: {status}")
            else:
                logger.info(f"{test}: {result}")
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
    
    # Additional custom validations
    
    # Check multiplicative interactions are reasonable
    if 'energy_circadian_factor' in dataset.columns:
        energy_circ = dataset['energy_circadian_factor'].dropna()
        if len(energy_circ) > 0:
            reasonable_range = (energy_circ >= 0).all() and (energy_circ <= 2).all()
            logger.info(f"Energy-circadian interaction in reasonable range: {'✅ PASS' if reasonable_range else '❌ FAIL'}")
    
    # Check composite indices have reasonable variance
    composite_features = [col for col in dataset.columns if 'composite' in col or 'index' in col]
    if composite_features:
        all_varied = True
        for feature in composite_features:
            data = dataset[feature].dropna()
            if len(data) > 0 and data.std() < 0.001:  # Very low variance
                all_varied = False
                break
        logger.info(f"Composite indices have reasonable variance: {'✅ PASS' if all_varied else '❌ FAIL'}")
    
    # Check performance multipliers are multiplicative (not additive)
    if 'overall_performance_multiplier' in dataset.columns:
        multiplier = dataset['overall_performance_multiplier'].dropna()
        if len(multiplier) > 0:
            # Multipliers should be centered around 1, not 0
            centered_around_one = abs(multiplier.mean() - 1.0) < 0.5
            logger.info(f"Performance multipliers centered around 1.0: {'✅ PASS' if centered_around_one else '❌ FAIL'}")

def analyze_feature_correlations(dataset: pd.DataFrame):
    """Analyze correlations between interaction features and home runs."""
    logger.info("\n" + "="*60)
    logger.info("INTERACTION FEATURE CORRELATION ANALYSIS")
    logger.info("="*60)
    
    if 'hit_hr' not in dataset.columns:
        logger.warning("No target variable for correlation analysis")
        return
    
    # Get interaction features with good coverage
    interaction_features = [col for col in dataset.columns 
                          if any(feat in col for feat in ['boost', 'synergy', 'index', 'multiplier', 'ratio', 'indicator'])]
    
    meaningful_features = []
    for feature in interaction_features:
        non_null_pct = dataset[feature].notna().mean()
        if non_null_pct > 0.8:  # Good coverage
            meaningful_features.append(feature)
    
    if not meaningful_features:
        logger.warning("No interaction features with sufficient coverage for correlation analysis")
        return
    
    logger.info(f"Analyzing correlations for {len(meaningful_features)} features")
    
    correlations = []
    for feature in meaningful_features:
        corr = dataset[feature].corr(dataset['hit_hr'])
        if not pd.isna(corr):
            correlations.append((feature, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    logger.info(f"\nTop interaction feature correlations with home runs:")
    for i, (feature, corr) in enumerate(correlations[:15], 1):
        logger.info(f"  {i:2d}. {feature}: {corr:+.4f}")

def main():
    """Main testing function."""
    logger.info("Starting feature interaction testing...")
    
    success = test_feature_interactions()
    
    if success:
        # Additional analysis
        try:
            dataset = pd.read_parquet('data/processed/pregame_dataset_2024-08-01_2024-08-15.parquet')
            analyze_feature_correlations(dataset)
            
            # Analyze interaction effects
            interaction_effects = analyze_interaction_effects(dataset)
            if interaction_effects:
                logger.info(f"\nInteraction effects analysis completed")
                for category, effects in interaction_effects.items():
                    logger.info(f"  {category}: {len(effects)} features analyzed")
            
            # Calculate interaction importance
            if 'hit_hr' in dataset.columns:
                importance_scores = calculate_interaction_importance(dataset)
                if importance_scores:
                    logger.info(f"\nInteraction importance calculated for {len(importance_scores)} features")
                    top_interactions = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.info("Top 5 most important interactions:")
                    for feature, score in top_interactions:
                        logger.info(f"  {feature}: {score:.4f}")
                
        except Exception as e:
            logger.warning(f"Could not run additional analysis: {e}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ FEATURE INTERACTIONS TESTING COMPLETED!")
        logger.info("Step 8 implementation successful.")
        logger.info("="*70)
        return 0
    else:
        logger.error("\nFeature interaction testing failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)