"""
Optimized Enhanced Features for Large Datasets
==============================================

Fixes the quadratic complexity in pitcher similarity calculations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

def optimize_pitcher_similarity_calculation(statcast_df: pd.DataFrame,
                                          batter_games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized pitcher similarity feature calculation.
    Replaces the O(n²) loop with vectorized operations.
    """
    logger.info("Calculating pitcher similarity features (optimized)...")
    
    # Build pitcher profiles (vectorized)
    pitcher_profiles = (
        statcast_df.groupby('pitcher')
        .agg({
            'release_speed': 'mean',
            'p_throws': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'R',
            'pitch_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'FF'
        })
        .fillna({'release_speed': 92.0, 'p_throws': 'R', 'pitch_type': 'FF'})
    )
    
    # Add velocity tiers
    pitcher_profiles['velocity_tier'] = pd.cut(
        pitcher_profiles['release_speed'], 
        bins=[0, 90, 94, 100], 
        labels=['low', 'medium', 'high']
    )
    
    # Pre-calculate similar pitcher groups (much faster than nested loops)
    similarity_groups = {}
    for handedness in ['L', 'R']:
        for velocity_tier in ['low', 'medium', 'high']:
            similar_pitchers = pitcher_profiles[
                (pitcher_profiles['p_throws'] == handedness) &
                (pitcher_profiles['velocity_tier'] == velocity_tier)
            ].index.tolist()
            
            similarity_groups[(handedness, velocity_tier)] = similar_pitchers
    
    # Calculate batter performance vs similar pitcher groups (vectorized)
    result_df = batter_games_df.copy()
    
    # Initialize default values
    similarity_features = [
        'vs_similar_hand_pa', 'vs_similar_hand_hr', 'vs_similar_hand_hr_rate',
        'vs_similar_velocity_pa', 'vs_similar_velocity_hr', 'vs_similar_velocity_hr_rate'
    ]
    
    for feature in similarity_features:
        result_df[feature] = 0.0
    
    # Get historical data cutoff (before game date)
    historical_cutoff = result_df['date'].min() - pd.Timedelta(days=1)
    historical_data = statcast_df[statcast_df['game_date'] <= historical_cutoff]
    
    if len(historical_data) == 0:
        logger.warning("No historical data available for similarity features")
        return result_df
    
    # Pre-calculate batter performance by pitcher groups
    batter_vs_groups = {}
    
    for batter_id in result_df['batter'].unique():
        if pd.isna(batter_id):
            continue
            
        batter_historical = historical_data[historical_data['batter'] == batter_id]
        
        if len(batter_historical) == 0:
            continue
        
        # Add pitcher characteristics to batter data
        batter_with_pitcher_info = batter_historical.merge(
            pitcher_profiles[['p_throws', 'velocity_tier']],
            left_on='pitcher',
            right_index=True,
            how='left'
        )
        
        # Calculate performance vs handedness groups
        hand_stats = (
            batter_with_pitcher_info.groupby('p_throws')
            .agg({
                'events': 'count',
                'home_run': 'sum'
            })
            .fillna(0)
        )
        
        # Calculate performance vs velocity groups  
        velocity_stats = (
            batter_with_pitcher_info.groupby('velocity_tier')
            .agg({
                'events': 'count', 
                'home_run': 'sum'
            })
            .fillna(0)
        )
        
        batter_vs_groups[batter_id] = {
            'handedness': hand_stats,
            'velocity': velocity_stats
        }
    
    # Apply features to each game (much faster now)
    for idx, game in result_df.iterrows():
        batter_id = game['batter']
        pitcher_id = game['opp_starter']
        
        if pd.isna(batter_id) or pd.isna(pitcher_id) or batter_id not in batter_vs_groups:
            continue
            
        if pitcher_id not in pitcher_profiles.index:
            continue
            
        pitcher_info = pitcher_profiles.loc[pitcher_id]
        batter_stats = batter_vs_groups[batter_id]
        
        # Handedness features
        handedness = pitcher_info['p_throws']
        if handedness in batter_stats['handedness'].index:
            hand_data = batter_stats['handedness'].loc[handedness]
            result_df.loc[idx, 'vs_similar_hand_pa'] = hand_data['events']
            result_df.loc[idx, 'vs_similar_hand_hr'] = hand_data['home_run']
            if hand_data['events'] > 0:
                result_df.loc[idx, 'vs_similar_hand_hr_rate'] = hand_data['home_run'] / hand_data['events']
        
        # Velocity features
        velocity_tier = pitcher_info['velocity_tier']
        if pd.notna(velocity_tier) and velocity_tier in batter_stats['velocity'].index:
            vel_data = batter_stats['velocity'].loc[velocity_tier]
            result_df.loc[idx, 'vs_similar_velocity_pa'] = vel_data['events']
            result_df.loc[idx, 'vs_similar_velocity_hr'] = vel_data['home_run']
            if vel_data['events'] > 0:
                result_df.loc[idx, 'vs_similar_velocity_hr_rate'] = vel_data['home_run'] / vel_data['events']
    
    logger.info("Added pitcher similarity features (optimized)")
    return result_df

def patch_enhanced_features():
    """Monkey patch the slow similarity calculation with the optimized version."""
    
    import enhanced_features
    
    # Replace the slow method with our optimized version
    enhanced_features.BatterPitcherMatchupCalculator.calculate_pitcher_similarity_features = lambda self, statcast_df, batter_games_df: optimize_pitcher_similarity_calculation(statcast_df, batter_games_df)
    
    logger.info("✅ Enhanced features optimized for large datasets")

if __name__ == "__main__":
    print("Optimized enhanced features module ready")
    print("Call patch_enhanced_features() to apply optimizations")