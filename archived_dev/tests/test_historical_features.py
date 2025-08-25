"""
Historical Test Script for New Matchup Features
==============================================

This script tests the new matchup features on a longer historical period (2023-2024)
to see the career matchup features in action with established batter-pitcher relationships.
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

def test_historical_matchup_features():
    """Test matchup features on a longer historical period."""
    logger.info("="*70)
    logger.info("TESTING MATCHUP FEATURES ON HISTORICAL DATA (2023-2024)")
    logger.info("="*70)
    
    try:
        # Build dataset with longer historical period
        logger.info("Building historical dataset with new features...")
        logger.info("Date range: 2023-04-01 to 2024-09-30")
        logger.info("This may take several minutes...")
        
        builder = PregameDatasetBuilder(
            start_date="2023-04-01", 
            end_date="2024-09-30"
        )
        
        # Build dataset (this should now include rich matchup data)
        dataset = builder.build_dataset(force_rebuild=True)
        
        if dataset.empty:
            logger.error("Dataset is empty - cannot test features")
            return False
        
        logger.info(f"Historical dataset built: {len(dataset)} rows, {len(dataset.columns)} columns")
        
        # Analyze matchup feature quality on historical data
        analyze_historical_features(dataset)
        
        # Compare early vs late season performance
        compare_seasonal_performance(dataset)
        
        # Test model performance with matchup features
        test_historical_model_performance(dataset)
        
        logger.info("\nHistorical feature testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Historical feature testing failed: {e}")
        return False

def analyze_historical_features(dataset: pd.DataFrame):
    """Analyze matchup feature quality on historical data."""
    logger.info("\n" + "="*60)
    logger.info("HISTORICAL MATCHUP FEATURE ANALYSIS")
    logger.info("="*60)
    
    # Focus on key matchup features
    key_features = [
        'matchup_pa_career', 'matchup_hr_career', 'matchup_hr_rate_career',
        'matchup_familiarity_score', 'matchup_encounters_last_year',
        'vs_similar_hand_hr_rate', 'vs_similar_velocity_hr_rate'
    ]
    
    for feature in key_features:
        if feature in dataset.columns:
            feature_data = dataset[feature]
            non_zero = feature_data[feature_data > 0]
            
            logger.info(f"\n{feature}:")
            logger.info(f"  Total samples: {len(feature_data)}")
            logger.info(f"  Non-zero values: {len(non_zero)} ({len(non_zero)/len(feature_data):.1%})")
            
            if len(non_zero) > 0:
                logger.info(f"  Non-zero stats:")
                logger.info(f"    Mean: {non_zero.mean():.4f}")
                logger.info(f"    Median: {non_zero.median():.4f}")
                logger.info(f"    Max: {non_zero.max():.4f}")
                logger.info(f"    Unique values: {non_zero.nunique()}")
                
                # Show distribution for career matchups
                if 'career' in feature:
                    value_counts = non_zero.value_counts().head(10)
                    logger.info(f"    Top values: {dict(value_counts)}")

def compare_seasonal_performance(dataset: pd.DataFrame):
    """Compare matchup feature performance across seasons."""
    logger.info("\n" + "="*60)
    logger.info("SEASONAL COMPARISON OF MATCHUP FEATURES")
    logger.info("="*60)
    
    if 'season' not in dataset.columns:
        dataset['season'] = pd.to_datetime(dataset['date']).dt.year
    
    # Split by season
    data_2023 = dataset[dataset['season'] == 2023]
    data_2024 = dataset[dataset['season'] == 2024]
    
    logger.info(f"2023 data: {len(data_2023)} games")
    logger.info(f"2024 data: {len(data_2024)} games")
    
    # Compare key features across seasons
    key_features = ['matchup_pa_career', 'matchup_hr_rate_career', 'matchup_familiarity_score']
    
    for feature in key_features:
        if feature in dataset.columns:
            # Stats for each season
            stats_2023 = data_2023[feature]
            stats_2024 = data_2024[feature]
            
            non_zero_2023 = stats_2023[stats_2023 > 0]
            non_zero_2024 = stats_2024[stats_2024 > 0]
            
            logger.info(f"\n{feature} - Seasonal Comparison:")
            logger.info(f"  2023 - Non-zero: {len(non_zero_2023)} ({len(non_zero_2023)/len(stats_2023):.1%})")
            if len(non_zero_2023) > 0:
                logger.info(f"         Mean: {non_zero_2023.mean():.4f}, Max: {non_zero_2023.max():.4f}")
            
            logger.info(f"  2024 - Non-zero: {len(non_zero_2024)} ({len(non_zero_2024)/len(stats_2024):.1%})")
            if len(non_zero_2024) > 0:
                logger.info(f"         Mean: {non_zero_2024.mean():.4f}, Max: {non_zero_2024.max():.4f}")

def test_historical_model_performance(dataset: pd.DataFrame):
    """Test model performance with matchup features on historical data."""
    logger.info("\n" + "="*60)
    logger.info("HISTORICAL MODEL PERFORMANCE TEST")
    logger.info("="*60)
    
    try:
        # Use a reasonable subset for testing (last 5000 games to avoid memory issues)
        if len(dataset) > 5000:
            # Take most recent 5000 games for testing
            test_data = dataset.tail(5000).copy()
            logger.info(f"Using most recent {len(test_data)} games for model testing")
        else:
            test_data = dataset.copy()
        
        # Ensure we have target variable
        if 'hit_hr' not in test_data.columns:
            if 'home_runs' in test_data.columns:
                test_data['hit_hr'] = (test_data['home_runs'] > 0).astype(int)
            else:
                logger.error("No target variable available for model testing")
                return
        
        # Initialize model system
        model_system = EnhancedDualModelSystem()
        
        # Check feature availability
        available_features = model_system.feature_selector.identify_available_features(test_data)
        
        logger.info(f"Available feature summary:")
        logger.info(f"  Core features: {len(available_features['core'])}")
        logger.info(f"  Matchup features: {len(available_features.get('matchup', []))}")
        logger.info(f"  Bat tracking features: {len(available_features.get('bat_tracking', []))}")
        logger.info(f"  Total enhanced features: {len(available_features['enhanced'])}")
        
        # Show some key matchup features that are available
        matchup_features = available_features.get('matchup', [])
        if matchup_features:
            logger.info(f"  Key matchup features: {matchup_features[:8]}...")
            
            # Check how many have meaningful values
            meaningful_features = []
            for feature in matchup_features[:5]:  # Check first 5
                if feature in test_data.columns:
                    non_zero_count = (test_data[feature] > 0).sum()
                    if non_zero_count > 0:
                        meaningful_features.append(f"{feature}({non_zero_count})")
            
            logger.info(f"  Features with non-zero values: {meaningful_features}")
        
        # Run quick model training with time-based split
        logger.info("\nTraining model with historical matchup features...")
        
        # Use time-based split for realistic evaluation
        results = model_system.fit(
            test_data,
            splitting_strategy='time_based',
            test_size=0.2,
            val_size=0.1,
            gap_days=7,  # 7-day gap to prevent data leakage
            cross_validate=False  # Skip CV for speed
        )
        
        logger.info("Model training completed successfully!")
        logger.info(f"Training results summary:")
        logger.info(f"  Split strategy: {results['split_strategy']}")
        logger.info(f"  Training samples: {results['train_size']}")
        logger.info(f"  Validation samples: {results['val_size']}")
        logger.info(f"  Test samples: {results['test_size']}")
        
        # Show test performance if available
        if 'test_metrics' in results and 'system' in results['test_metrics']:
            test_metrics = results['test_metrics']['system']
            logger.info(f"\nModel Performance on Historical Data:")
            logger.info(f"  ROC AUC: {test_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            logger.info(f"  Precision: {test_metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall: {test_metrics.get('recall', 0):.4f}")
            logger.info(f"  HR Rate (actual): {test_metrics.get('positive_rate', 0):.4f}")
        
    except Exception as e:
        logger.error(f"Model performance testing failed: {e}")

def analyze_specific_matchups(dataset: pd.DataFrame):
    """Analyze some specific interesting matchups."""
    logger.info("\n" + "="*60)
    logger.info("SPECIFIC MATCHUP ANALYSIS")
    logger.info("="*60)
    
    # Find matchups with significant history
    high_matchup_data = dataset[dataset['matchup_pa_career'] >= 5].copy()
    
    if len(high_matchup_data) > 0:
        logger.info(f"Found {len(high_matchup_data)} games with significant matchup history (5+ career PAs)")
        
        # Show top matchups by familiarity
        top_matchups = high_matchup_data.nlargest(10, 'matchup_pa_career')
        
        logger.info("\nTop 10 most established matchups:")
        for idx, row in top_matchups.iterrows():
            logger.info(f"  Batter {row['batter']} vs Pitcher {row.get('opp_starter', 'Unknown')}: "
                       f"{row['matchup_pa_career']:.0f} career PAs, "
                       f"{row['matchup_hr_career']:.0f} HRs "
                       f"({row['matchup_hr_rate_career']:.3f} rate)")
    else:
        logger.info("No games with significant matchup history found")

def main():
    """Main testing function."""
    logger.info("Starting historical matchup feature testing...")
    logger.info("This will take several minutes due to the larger data range...")
    
    success = test_historical_matchup_features()
    
    if success:
        logger.info("\n" + "="*70)
        logger.info("HISTORICAL TESTING COMPLETED SUCCESSFULLY!")
        logger.info("Matchup features are working with real historical data.")
        logger.info("="*70)
        return 0
    else:
        logger.error("\n" + "="*70)
        logger.error("HISTORICAL TESTING FAILED!")
        logger.error("Please check the implementation.")
        logger.error("="*70)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)