#!/usr/bin/env python3
"""
Fix Model Predictions by Aligning Features
==========================================

This script ensures the model uses the correct features for predictions.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_features():
    """Get the features the model should use based on config."""
    # These are the features defined in config.py
    all_features = (
        config.CORE_FEATURES +
        config.MATCHUP_FEATURES +
        config.SITUATIONAL_FEATURES +
        config.WEATHER_FEATURES +
        config.RECENT_FORM_FEATURES +
        config.STREAK_MOMENTUM_FEATURES +
        config.BALLPARK_FEATURES +
        config.TEMPORAL_FATIGUE_FEATURES +
        config.INTERACTION_FEATURES
    )
    return all_features

def align_features_for_prediction(df, feature_list):
    """Select and align features for prediction."""
    # Remove non-predictive columns
    exclude_cols = ['date', 'batter_name', 'game_pk', 'hit_hr', 'home_runs', 
                   'season', 'batter', 'bat_team', 'home_team', 'away_team',
                   'stadium', 'opp_team', 'opp_starter']
    
    # Get available features
    available = [col for col in df.columns if col not in exclude_cols]
    
    # Try to use config features first
    config_features = [f for f in feature_list if f in available]
    
    if len(config_features) < 20:  # If too few config features, use all numeric
        logger.warning(f"Only {len(config_features)} config features found, using all numeric columns")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col not in exclude_cols]
    else:
        features = config_features
    
    logger.info(f"Using {len(features)} features for prediction")
    return features

def test_prediction_alignment():
    """Test that predictions work with aligned features."""
    
    # Load recent data
    logger.info("Loading recent data...")
    df = pd.read_parquet('data/processed/pregame_dataset_latest.parquet')
    
    # Get config features
    config_features = get_model_features()
    logger.info(f"Config defines {len(config_features)} features")
    
    # Align features
    prediction_features = align_features_for_prediction(df, config_features)
    logger.info(f"Selected {len(prediction_features)} features for predictions")
    
    # Prepare data
    X = df[prediction_features].fillna(0)
    
    # Load model
    model = joblib.load('saved_models_pregame/enhanced_model.joblib')
    
    # Test prediction
    sample = X.iloc[:10]
    try:
        probs = model.predict_proba(sample)
        logger.info(f"✅ Predictions successful! Shape: {probs.shape}")
        logger.info(f"Sample probabilities: {probs[:3, 1]}")
        return True
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return False

def create_feature_aligned_model():
    """Create a new model that's properly aligned with features."""
    
    logger.info("Creating feature-aligned model...")
    
    # Load data
    df = pd.read_parquet('data/processed/pregame_dataset_latest.parquet')
    
    # Get features
    config_features = get_model_features()
    features = align_features_for_prediction(df, config_features)
    
    # Prepare training data
    X = df[features].fillna(0)
    y = df['hit_hr'] if 'hit_hr' in df.columns else (df['home_runs'] > 0).astype(int)
    
    logger.info(f"Training with {len(X)} samples, {len(features)} features")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X, y)
    
    # Save model with feature names
    model.feature_names = features
    joblib.dump(model, 'saved_models_pregame/enhanced_model_aligned.joblib')
    
    # Also save feature list
    pd.Series(features).to_csv('saved_models_pregame/model_features.csv', index=False)
    
    logger.info(f"✅ Model saved with {len(features)} aligned features")
    
    # Show feature importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    logger.info("Top 10 important features:")
    for feat, imp in importance.head(10).items():
        logger.info(f"  {feat}: {imp:.4f}")
    
    return model

if __name__ == "__main__":
    import sys
    
    print("🔧 Fixing Model Feature Alignment")
    print("=" * 50)
    
    # Test current alignment
    print("\n1. Testing current model alignment...")
    if not test_prediction_alignment():
        print("\n2. Creating properly aligned model...")
        create_feature_aligned_model()
        
        print("\n3. Testing new model...")
        # Replace old model with aligned one
        import shutil
        shutil.copy('saved_models_pregame/enhanced_model_aligned.joblib',
                   'saved_models_pregame/enhanced_model.joblib')
        
        if test_prediction_alignment():
            print("\n✅ Model fixed and ready for predictions!")
        else:
            print("\n❌ Model still has issues")
            sys.exit(1)
    else:
        print("\n✅ Model already properly aligned!")