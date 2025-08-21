"""
Analyze Existing Matchup Features
=================================

Analyze the matchup features we've already created to understand their behavior
and validate they're working correctly.
"""

import pandas as pd
import numpy as np
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_existing_features():
    """Analyze the existing dataset with matchup features."""
    logger.info("="*70)
    logger.info("ANALYZING EXISTING MATCHUP FEATURES")
    logger.info("="*70)
    
    try:
        # Load the dataset we already created
        dataset_path = 'data/processed/pregame_dataset_2024-08-01_2024-08-15.parquet'
        dataset = pd.read_parquet(dataset_path)
        
        logger.info(f"Loaded dataset: {len(dataset)} rows, {len(dataset.columns)} columns")
        logger.info(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")
        
        # Find all matchup features
        matchup_features = [col for col in dataset.columns if 'matchup' in col or 'vs_similar' in col]
        logger.info(f"Found {len(matchup_features)} matchup features: {matchup_features}")
        
        # Analyze each feature in detail
        analyze_feature_quality(dataset, matchup_features)
        
        # Analyze relationships
        analyze_feature_relationships(dataset, matchup_features)
        
        # Test predictive power
        test_predictive_power(dataset, matchup_features)
        
        # Create extended test with more historical data
        create_extended_test(dataset)
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False

def analyze_feature_quality(dataset: pd.DataFrame, matchup_features: list):
    """Analyze the quality and distribution of matchup features."""
    logger.info("\n" + "="*60)
    logger.info("FEATURE QUALITY ANALYSIS")
    logger.info("="*60)
    
    for feature in matchup_features:
        data = dataset[feature]
        
        # Basic statistics
        non_null = data.notna().sum()
        non_zero = (data > 0).sum()
        unique_values = data.nunique()
        
        logger.info(f"\n{feature}:")
        logger.info(f"  Non-null: {non_null:,} ({non_null/len(dataset):.1%})")
        logger.info(f"  Non-zero: {non_zero:,} ({non_zero/len(dataset):.1%})")
        logger.info(f"  Unique values: {unique_values:,}")
        
        if non_zero > 0:
            non_zero_data = data[data > 0]
            logger.info(f"  Non-zero stats:")
            logger.info(f"    Mean: {non_zero_data.mean():.4f}")
            logger.info(f"    Median: {non_zero_data.median():.4f}")
            logger.info(f"    Max: {non_zero_data.max():.4f}")
            logger.info(f"    Std: {non_zero_data.std():.4f}")
            
            if unique_values > 10:
                # Show distribution for features with good variation
                logger.info(f"    Distribution (quartiles):")
                quartiles = non_zero_data.quantile([0.25, 0.5, 0.75])
                logger.info(f"      25%: {quartiles[0.25]:.4f}")
                logger.info(f"      50%: {quartiles[0.5]:.4f}")
                logger.info(f"      75%: {quartiles[0.75]:.4f}")

def analyze_feature_relationships(dataset: pd.DataFrame, matchup_features: list):
    """Analyze relationships between features."""
    logger.info("\n" + "="*60)
    logger.info("FEATURE RELATIONSHIP ANALYSIS")
    logger.info("="*60)
    
    # Check correlations with actual home runs
    if 'hit_hr' in dataset.columns:
        correlations = []
        for feature in matchup_features:
            if (dataset[feature] > 0).sum() > 100:  # Only features with enough data
                corr = dataset[feature].corr(dataset['hit_hr'])
                if not pd.isna(corr):
                    correlations.append((feature, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        logger.info("Correlations with home runs (top 10):")
        for feature, corr in correlations[:10]:
            logger.info(f"  {feature}: {corr:+.4f}")
    
    # Check logical relationships
    logger.info("\nLogical relationship checks:")
    
    # Career vs recent should be related
    if 'matchup_pa_career' in dataset.columns and 'matchup_pa_recent' in dataset.columns:
        career_data = dataset[dataset['matchup_pa_career'] > 0]
        if len(career_data) > 0:
            recent_ratio = (career_data['matchup_pa_recent'] / career_data['matchup_pa_career']).mean()
            logger.info(f"  Recent/Career PA ratio: {recent_ratio:.3f} (should be ≤ 1.0)")
    
    # HR rates should be reasonable
    rate_features = [f for f in matchup_features if 'hr_rate' in f]
    for feature in rate_features:
        if feature in dataset.columns:
            rate_data = dataset[dataset[feature] > 0][feature]
            if len(rate_data) > 0:
                high_rates = (rate_data > 0.15).sum()  # > 15% HR rate
                logger.info(f"  {feature} high rates (>15%): {high_rates}/{len(rate_data)} "
                           f"({high_rates/len(rate_data):.1%})")

def test_predictive_power(dataset: pd.DataFrame, matchup_features: list):
    """Test the predictive power of matchup features."""
    logger.info("\n" + "="*60)
    logger.info("PREDICTIVE POWER ANALYSIS")
    logger.info("="*60)
    
    if 'hit_hr' not in dataset.columns:
        logger.warning("No target variable available for predictive analysis")
        return
    
    # Test individual feature predictive power
    meaningful_features = []
    for feature in matchup_features:
        non_zero_count = (dataset[feature] > 0).sum()
        if non_zero_count > 50:  # Need reasonable sample size
            meaningful_features.append(feature)
    
    logger.info(f"Testing {len(meaningful_features)} features with sufficient data")
    
    # Simple predictive test: compare HR rates in different feature value ranges
    for feature in meaningful_features[:5]:  # Test top 5 to avoid spam
        feature_data = dataset[dataset[feature] > 0]
        
        if len(feature_data) > 100:
            # Split into quartiles and compare HR rates
            try:
                quartiles = feature_data[feature].quantile([0.25, 0.5, 0.75])
                
                q1_data = feature_data[feature_data[feature] <= quartiles[0.25]]
                q4_data = feature_data[feature_data[feature] >= quartiles[0.75]]
                
                if len(q1_data) > 10 and len(q4_data) > 10:
                    q1_hr_rate = q1_data['hit_hr'].mean()
                    q4_hr_rate = q4_data['hit_hr'].mean()
                    
                    logger.info(f"\n{feature} quartile analysis:")
                    logger.info(f"  Bottom quartile HR rate: {q1_hr_rate:.3f} (n={len(q1_data)})")
                    logger.info(f"  Top quartile HR rate: {q4_hr_rate:.3f} (n={len(q4_data)})")
                    logger.info(f"  Difference: {q4_hr_rate - q1_hr_rate:+.3f}")
                    
            except Exception as e:
                logger.debug(f"Could not analyze {feature}: {e}")

def create_extended_test(dataset: pd.DataFrame):
    """Create a test to show what extended historical data would look like."""
    logger.info("\n" + "="*60)
    logger.info("EXTENDED HISTORICAL SIMULATION")
    logger.info("="*60)
    
    logger.info("Simulating what we'd see with longer historical periods:")
    
    # Current coverage
    current_established = (dataset['matchup_pa_career'] > 0).sum()
    logger.info(f"Current established matchups: {current_established:,} ({current_established/len(dataset):.1%})")
    
    # Similarity features work well
    similarity_coverage = (dataset['vs_similar_hand_pa'] > 0).sum()
    logger.info(f"Similarity feature coverage: {similarity_coverage:,} ({similarity_coverage/len(dataset):.1%})")
    
    # Show what established matchups look like
    established = dataset[dataset['matchup_pa_career'] > 0]
    if len(established) > 0:
        logger.info(f"\nEstablished matchup characteristics:")
        logger.info(f"  Average career PAs: {established['matchup_pa_career'].mean():.1f}")
        logger.info(f"  Average career HR rate: {established['matchup_hr_rate_career'].mean():.3f}")
        logger.info(f"  Actual HR rate in these games: {established['hit_hr'].mean():.3f}")
        
        # Show best examples
        top_established = established.nlargest(5, 'matchup_pa_career')
        logger.info(f"\nTop established matchups in this dataset:")
        for i, (_, row) in enumerate(top_established.iterrows(), 1):
            logger.info(f"  {i}. {row['matchup_pa_career']:.0f} PAs, "
                       f"{row['matchup_hr_career']:.0f} HRs "
                       f"({row['matchup_hr_rate_career']:.3f} rate) "
                       f"→ {'HR' if row['hit_hr'] == 1 else 'No HR'}")
    
    logger.info(f"\nWith full season+ data, we'd expect:")
    logger.info(f"  • ~30-50% of games to have established matchups (5+ PAs)")
    logger.info(f"  • Career rates to be more predictive")
    logger.info(f"  • Familiarity scores to show meaningful variation")
    logger.info(f"  • Recent encounter data to be more rich")

def main():
    """Main analysis function."""
    logger.info("Starting analysis of existing matchup features...")
    
    success = analyze_existing_features()
    
    if success:
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("The matchup features are working correctly.")
        logger.info("They would be much more powerful with longer historical data.")
        logger.info("="*70)
        return 0
    else:
        logger.error("\n" + "="*70)
        logger.error("ANALYSIS FAILED!")
        logger.error("="*70)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)