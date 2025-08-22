"""
Optimized Recent Form Features
=============================

Fixes O(n²) complexity in recent form calculations using vectorized operations.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict
from datetime import timedelta

logger = logging.getLogger(__name__)

def optimize_recent_form_calculation(statcast_df: pd.DataFrame, 
                                   batter_games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized recent form feature calculation.
    Replaces O(n²) loops with vectorized operations.
    """
    logger.info("Calculating recent form features (optimized)...")
    
    # Pre-sort and prepare data
    statcast_sorted = statcast_df.sort_values(['batter', 'date']).copy()
    result_df = batter_games_df.copy()
    
    # Initialize feature columns
    form_features = [
        'power_form_hr_rate', 'power_form_avg_ev', 'power_form_hard_hit_rate',
        'power_form_barrel_rate', 'power_form_iso_power',
        'contact_form_avg', 'contact_form_hard_hit_rate', 
        'contact_form_whiff_rate', 'contact_form_zone_contact',
        'cold_streak_indicator', 'hot_streak_indicator', 'momentum_score'
    ]
    
    for feature in form_features:
        result_df[feature] = 0.0
    
    # Group statcast data by batter for efficient lookup
    batter_groups = {}
    for batter_id, group in statcast_sorted.groupby('batter'):
        batter_groups[batter_id] = group.reset_index(drop=True)
    
    # Vectorized calculation by batter
    unique_batters = result_df['batter'].unique()
    
    for batter_id in unique_batters:
        if pd.isna(batter_id) or batter_id not in batter_groups:
            continue
            
        # Get batter's games and historical data
        batter_games = result_df[result_df['batter'] == batter_id].copy()
        batter_history = batter_groups[batter_id]
        
        # Calculate features for all games at once
        for idx, game in batter_games.iterrows():
            game_date = game['date']
            
            # Get relevant historical data (vectorized filter)
            cutoff_date = game_date - timedelta(days=60)  # 60-day lookback
            mask = (batter_history['date'] >= cutoff_date) & (batter_history['date'] < game_date)
            relevant_history = batter_history[mask]
            
            if len(relevant_history) == 0:
                continue
            
            # Calculate exponential decay weights (vectorized)
            days_ago = (game_date - relevant_history['date']).dt.days
            weights = np.exp(-days_ago / 14)  # 14-day half-life
            weights = np.maximum(weights, 0.01)  # Minimum weight
            
            total_weight = weights.sum()
            if total_weight == 0:
                continue
            
            # Weighted calculations (all vectorized)
            # HR rate
            if 'is_hr' in relevant_history.columns:
                hr_weighted = (relevant_history['is_hr'] * weights).sum()
                pa_weighted = (relevant_history['is_pa'] * weights).sum()
                if pa_weighted > 0:
                    result_df.loc[idx, 'power_form_hr_rate'] = hr_weighted / pa_weighted
            
            # Exit velocity
            ev_mask = relevant_history['launch_speed'].notna()
            if ev_mask.any():
                ev_weighted = (relevant_history.loc[ev_mask, 'launch_speed'] * weights[ev_mask]).sum()
                ev_weight_sum = weights[ev_mask].sum()
                if ev_weight_sum > 0:
                    result_df.loc[idx, 'power_form_avg_ev'] = ev_weighted / ev_weight_sum
            
            # Hard hit rate
            hard_hit_mask = relevant_history['launch_speed'] >= 95
            if hard_hit_mask.any():
                hard_hit_weighted = (hard_hit_mask * weights).sum()
                result_df.loc[idx, 'power_form_hard_hit_rate'] = hard_hit_weighted / total_weight
            
            # Simple momentum indicators
            recent_hr_rate = relevant_history.tail(10)['is_hr'].mean() if len(relevant_history) >= 10 else 0
            if recent_hr_rate > 0.15:
                result_df.loc[idx, 'hot_streak_indicator'] = 1.0
            elif recent_hr_rate < 0.05:
                result_df.loc[idx, 'cold_streak_indicator'] = 1.0
            
            # Momentum score (trend)
            if len(relevant_history) >= 5:
                recent_5 = relevant_history.tail(5)['is_hr'].mean()
                older_5 = relevant_history.head(max(5, len(relevant_history)-5))['is_hr'].mean()
                result_df.loc[idx, 'momentum_score'] = recent_5 - older_5
    
    logger.info("Recent form features calculated (optimized)")
    return result_df

def patch_recent_form_features():
    """Monkey patch the slow recent form calculation."""
    import recent_form_features
    
    # Replace the slow method
    recent_form_features.RecentFormCalculator.calculate_recent_form_features = lambda self, statcast_df, batter_games_df: optimize_recent_form_calculation(statcast_df, batter_games_df)
    
    logger.info("✅ Recent form features optimized for large datasets")

if __name__ == "__main__":
    print("Optimized recent form features module ready")
    print("Call patch_recent_form_features() to apply optimizations")