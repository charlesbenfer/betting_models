"""
Focused Historical Test for Matchup Features
==========================================

Tests matchup features on a focused period (June-August 2024) to see meaningful
matchup data without the extremely long processing time.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import sys

# Import our modules
from config import config
from dataset_builder import PregameDatasetBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_focused_historical_features():
    """Test matchup features on focused historical period."""
    logger.info("="*70)
    logger.info("TESTING MATCHUP FEATURES - FOCUSED HISTORICAL (Jun-Aug 2024)")
    logger.info("="*70)
    
    try:
        # Build dataset with focused historical period (3 months)
        logger.info("Building focused historical dataset...")
        logger.info("Date range: 2024-06-01 to 2024-08-31")
        
        builder = PregameDatasetBuilder(
            start_date="2024-06-01", 
            end_date="2024-08-31"
        )
        
        # Build dataset 
        dataset = builder.build_dataset(force_rebuild=True)
        
        if dataset.empty:
            logger.error("Dataset is empty - cannot test features")
            return False
        
        logger.info(f"Focused dataset built: {len(dataset)} rows, {len(dataset.columns)} columns")
        
        # Analyze matchup features
        analyze_focused_features(dataset)
        
        # Look for established matchups
        analyze_established_matchups(dataset)
        
        # Compare with broader patterns
        analyze_matchup_patterns(dataset)
        
        logger.info("\nFocused historical testing completed!")
        return True
        
    except Exception as e:
        logger.error(f"Focused testing failed: {e}")
        return False

def analyze_focused_features(dataset: pd.DataFrame):
    """Analyze matchup features on the focused dataset."""
    logger.info("\n" + "="*60)
    logger.info("FOCUSED MATCHUP FEATURE ANALYSIS")
    logger.info("="*60)
    
    # Check all matchup features
    matchup_features = [col for col in dataset.columns if 'matchup' in col or 'vs_similar' in col]
    
    logger.info(f"Found {len(matchup_features)} matchup features")
    
    # Analyze key features in detail
    key_features = [
        'matchup_pa_career', 'matchup_hr_career', 'matchup_hr_rate_career',
        'matchup_familiarity_score', 'matchup_encounters_last_year',
        'vs_similar_hand_hr_rate', 'vs_similar_velocity_hr_rate'
    ]
    
    for feature in key_features:
        if feature in dataset.columns:
            data = dataset[feature]
            non_zero = data[data > 0]
            
            logger.info(f"\n{feature}:")
            logger.info(f"  Total samples: {len(data):,}")
            logger.info(f"  Non-zero: {len(non_zero):,} ({len(non_zero)/len(data):.2%})")
            
            if len(non_zero) > 0:
                logger.info(f"  Non-zero stats:")
                logger.info(f"    Mean: {non_zero.mean():.4f}")
                logger.info(f"    Median: {non_zero.median():.4f}")
                logger.info(f"    Max: {non_zero.max():.4f}")
                logger.info(f"    Std: {non_zero.std():.4f}")
                logger.info(f"    75th percentile: {non_zero.quantile(0.75):.4f}")
                
                # Show some example values for career features
                if 'career' in feature and len(non_zero) > 10:
                    top_values = non_zero.nlargest(5)
                    logger.info(f"    Top 5 values: {top_values.tolist()}")

def analyze_established_matchups(dataset: pd.DataFrame):
    """Find and analyze established batter-pitcher matchups."""
    logger.info("\n" + "="*60)
    logger.info("ESTABLISHED MATCHUP ANALYSIS")
    logger.info("="*60)
    
    # Find games with meaningful matchup history
    established = dataset[dataset['matchup_pa_career'] >= 3].copy()
    
    if len(established) > 0:
        logger.info(f"Found {len(established):,} games with established matchups (3+ career PAs)")
        logger.info(f"That's {len(established)/len(dataset):.1%} of all games")
        
        # Analyze the quality of these matchups
        logger.info(f"\nEstablished matchup statistics:")
        logger.info(f"  Average career PAs: {established['matchup_pa_career'].mean():.1f}")
        logger.info(f"  Average career HRs: {established['matchup_hr_career'].mean():.2f}")
        logger.info(f"  Average career HR rate: {established['matchup_hr_rate_career'].mean():.3f}")
        logger.info(f"  Max career PAs: {established['matchup_pa_career'].max():.0f}")
        
        # Find most established matchups
        top_established = established.nlargest(10, 'matchup_pa_career')
        
        logger.info(f"\nTop 10 most established matchups:")
        for i, (idx, row) in enumerate(top_established.iterrows(), 1):
            batter_name = row.get('batter_name', f"Batter {row['batter']}")
            logger.info(f"  {i:2d}. {batter_name}: "
                       f"{row['matchup_pa_career']:.0f} PAs, "
                       f"{row['matchup_hr_career']:.0f} HRs "
                       f"({row['matchup_hr_rate_career']:.3f} rate)")
        
        # Compare established vs new matchups
        new_matchups = dataset[dataset['matchup_pa_career'] == 0]
        
        logger.info(f"\nMatchup comparison:")
        logger.info(f"  Established matchups: {len(established):,} games")
        logger.info(f"  New matchups: {len(new_matchups):,} games")
        logger.info(f"  Actual HR rates:")
        logger.info(f"    Established: {established['hit_hr'].mean():.3f}")
        logger.info(f"    New: {new_matchups['hit_hr'].mean():.3f}")
        logger.info(f"    Difference: {established['hit_hr'].mean() - new_matchups['hit_hr'].mean():.3f}")
        
    else:
        logger.info("No established matchups found (this suggests the dataset period is too short)")

def analyze_matchup_patterns(dataset: pd.DataFrame):
    """Analyze broader patterns in matchup data."""
    logger.info("\n" + "="*60)
    logger.info("MATCHUP PATTERN ANALYSIS")
    logger.info("="*60)
    
    # Analyze similarity features (these should have good coverage)
    similarity_features = [col for col in dataset.columns if 'vs_similar' in col]
    
    for feature in similarity_features:
        if feature in dataset.columns:
            data = dataset[feature]
            non_zero = data[data > 0]
            
            logger.info(f"\n{feature}:")
            logger.info(f"  Coverage: {len(non_zero)/len(data):.1%}")
            
            if len(non_zero) > 100:  # Only analyze if we have enough data
                logger.info(f"  Distribution quartiles:")
                quartiles = non_zero.quantile([0.25, 0.5, 0.75])
                logger.info(f"    25th: {quartiles[0.25]:.4f}")
                logger.info(f"    50th: {quartiles[0.5]:.4f}")
                logger.info(f"    75th: {quartiles[0.75]:.4f}")
                
                if 'hr_rate' in feature:
                    # Analyze HR rate distributions
                    high_rate = non_zero[non_zero > 0.05]  # > 5% HR rate
                    logger.info(f"  High performers (>5% HR rate): {len(high_rate)/len(non_zero):.1%}")
    
    # Check days since last encounter distribution
    if 'matchup_days_since_last' in dataset.columns:
        days_data = dataset['matchup_days_since_last']
        recent_encounters = days_data[days_data < 365]  # Within last year
        
        logger.info(f"\nRecent encounter analysis:")
        logger.info(f"  Games with encounters in last year: {len(recent_encounters):,} "
                   f"({len(recent_encounters)/len(dataset):.1%})")
        
        if len(recent_encounters) > 0:
            logger.info(f"  Average days since last: {recent_encounters.mean():.0f}")
            logger.info(f"  Most recent encounter: {recent_encounters.min():.0f} days ago")

def check_feature_correlations(dataset: pd.DataFrame):
    """Check correlations between matchup features and home runs."""
    logger.info("\n" + "="*60)
    logger.info("FEATURE CORRELATION ANALYSIS")
    logger.info("="*60)
    
    if 'hit_hr' not in dataset.columns:
        logger.warning("No target variable available for correlation analysis")
        return
    
    # Get matchup features with reasonable coverage
    matchup_features = [col for col in dataset.columns if 'matchup' in col or 'vs_similar' in col]
    
    meaningful_features = []
    for feature in matchup_features:
        non_zero_pct = (dataset[feature] > 0).mean()
        if non_zero_pct > 0.05:  # At least 5% coverage
            meaningful_features.append(feature)
    
    logger.info(f"Analyzing correlations for {len(meaningful_features)} features with >5% coverage")
    
    correlations = []
    for feature in meaningful_features:
        corr = dataset[feature].corr(dataset['hit_hr'])
        if not pd.isna(corr):
            correlations.append((feature, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    logger.info(f"\nTop correlations with home runs:")
    for i, (feature, corr) in enumerate(correlations[:10], 1):
        logger.info(f"  {i:2d}. {feature}: {corr:+.4f}")

def main():
    """Main function."""
    logger.info("Starting focused historical matchup testing...")
    
    success = test_focused_historical_features()
    
    if success:
        # Also run correlation analysis
        logger.info("Loading dataset for correlation analysis...")
        try:
            dataset = pd.read_parquet('data/processed/pregame_dataset_2024-06-01_2024-08-31.parquet')
            check_feature_correlations(dataset)
        except Exception as e:
            logger.warning(f"Could not run correlation analysis: {e}")
        
        logger.info("\n" + "="*70)
        logger.info("FOCUSED HISTORICAL TESTING COMPLETED!")
        logger.info("="*70)
        return 0
    else:
        logger.error("\nFocused historical testing failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)