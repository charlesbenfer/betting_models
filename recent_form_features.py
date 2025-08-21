"""
Recent Form Features with Decay Functions
=========================================

Advanced time-weighted statistics that emphasize recent performance over older games.
Uses exponential decay to weight recent games more heavily than distant ones.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from data_utils import DataValidator

logger = logging.getLogger(__name__)

@dataclass
class DecayParameters:
    """Parameters for exponential decay functions."""
    half_life_days: float  # Days for value to decay to 50%
    min_weight: float = 0.01  # Minimum weight for very old games
    max_lookback_days: int = 90  # Maximum days to look back

class RecentFormCalculator:
    """Calculate time-weighted recent form features with exponential decay."""
    
    def __init__(self):
        self.validator = DataValidator()
        
        # Define decay parameters for different types of stats
        self.decay_params = {
            'power': DecayParameters(half_life_days=14, min_weight=0.05, max_lookback_days=60),  # Power stats decay faster
            'contact': DecayParameters(half_life_days=21, min_weight=0.03, max_lookback_days=75),  # Contact stats more stable
            'plate_discipline': DecayParameters(half_life_days=28, min_weight=0.02, max_lookback_days=90),  # Most stable
            'hot_cold': DecayParameters(half_life_days=7, min_weight=0.1, max_lookback_days=30)  # Hot/cold streaks very recent
        }
    
    def calculate_recent_form_features(self, statcast_df: pd.DataFrame, 
                                     batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time-weighted recent form features.
        
        Args:
            statcast_df: Historical Statcast data
            batter_games_df: Batter games for prediction
            
        Returns:
            DataFrame with recent form features added
        """
        logger.info("Calculating recent form features with time decay...")
        
        # Validate inputs
        required_sc_cols = ['date', 'batter', 'is_pa', 'is_hr', 'launch_speed', 'launch_angle']
        self.validator.validate_required_columns(statcast_df, required_sc_cols, "Statcast for recent form")
        
        required_bg_cols = ['date', 'batter']
        self.validator.validate_required_columns(batter_games_df, required_bg_cols, "Batter games for recent form")
        
        # Ensure date columns are datetime
        statcast_df = statcast_df.copy()
        batter_games_df = batter_games_df.copy()
        statcast_df['date'] = pd.to_datetime(statcast_df['date'])
        batter_games_df['date'] = pd.to_datetime(batter_games_df['date'])
        
        # Calculate different types of weighted features
        result_df = batter_games_df.copy()
        
        # Power-focused features (home runs, exit velocity)
        result_df = self._add_power_form_features(statcast_df, result_df)
        
        # Contact quality features  
        result_df = self._add_contact_form_features(statcast_df, result_df)
        
        # Plate discipline features
        result_df = self._add_discipline_form_features(statcast_df, result_df)
        
        # Hot/cold streak indicators
        result_df = self._add_streak_indicators(statcast_df, result_df)
        
        # Form trend analysis
        result_df = self._add_trend_features(statcast_df, result_df)
        
        logger.info("Recent form feature calculation complete")
        return result_df
    
    def _add_power_form_features(self, statcast_df: pd.DataFrame, 
                               games_df: pd.DataFrame) -> pd.DataFrame:
        """Add time-weighted power-related form features."""
        logger.info("Adding power form features...")
        
        df = games_df.copy()
        decay_params = self.decay_params['power']
        
        # Initialize power form columns
        power_features = [
            'power_form_hr_rate', 'power_form_avg_ev', 'power_form_hard_hit_rate',
            'power_form_barrel_rate', 'power_form_iso_power'
        ]
        
        for feature in power_features:
            df[feature] = 0.0
        
        # Calculate for each batter-game
        for idx, game in df.iterrows():
            batter_id = game['batter']
            game_date = game['date']
            
            # Get historical data for this batter
            batter_history = statcast_df[
                (statcast_df['batter'] == batter_id) & 
                (statcast_df['date'] < game_date)
            ].copy()
            
            if batter_history.empty:
                continue
            
            # Filter to lookback window
            cutoff_date = game_date - timedelta(days=decay_params.max_lookback_days)
            batter_history = batter_history[batter_history['date'] >= cutoff_date]
            
            if batter_history.empty:
                continue
            
            # Calculate time weights
            batter_history['days_ago'] = (game_date - batter_history['date']).dt.days
            batter_history['weight'] = self._calculate_decay_weight(
                batter_history['days_ago'], decay_params
            )
            
            # Weighted power statistics
            total_weight = batter_history['weight'].sum()
            if total_weight > 0:
                # Home run rate
                hr_numerator = (batter_history['is_hr'] * batter_history['weight']).sum()
                pa_denominator = (batter_history['is_pa'] * batter_history['weight']).sum()
                df.loc[idx, 'power_form_hr_rate'] = hr_numerator / max(pa_denominator, 1)
                
                # Average exit velocity
                ev_data = batter_history[batter_history['launch_speed'].notna()]
                if not ev_data.empty:
                    weighted_ev = (ev_data['launch_speed'] * ev_data['weight']).sum()
                    weight_sum = ev_data['weight'].sum()
                    df.loc[idx, 'power_form_avg_ev'] = weighted_ev / max(weight_sum, 1)
                
                # Hard hit rate (95+ mph)
                hard_hit = batter_history['launch_speed'] >= 95
                hard_hit_numerator = (hard_hit * batter_history['weight']).sum()
                contact_denominator = (batter_history['launch_speed'].notna() * batter_history['weight']).sum()
                df.loc[idx, 'power_form_hard_hit_rate'] = hard_hit_numerator / max(contact_denominator, 1)
                
                # Barrel rate (simplified: EV 98+ and optimal launch angle)
                if 'launch_angle' in batter_history.columns:
                    barrels = (batter_history['launch_speed'] >= 98) & \
                             (batter_history['launch_angle'].between(20, 35))
                    barrel_numerator = (barrels * batter_history['weight']).sum()
                    df.loc[idx, 'power_form_barrel_rate'] = barrel_numerator / max(contact_denominator, 1)
        
        logger.info(f"Added {len(power_features)} power form features")
        return df
    
    def _add_contact_form_features(self, statcast_df: pd.DataFrame, 
                                 games_df: pd.DataFrame) -> pd.DataFrame:
        """Add time-weighted contact quality form features."""
        logger.info("Adding contact form features...")
        
        df = games_df.copy()
        decay_params = self.decay_params['contact']
        
        # Initialize contact form columns
        contact_features = [
            'contact_form_avg_la', 'contact_form_sweet_spot_rate',
            'contact_form_line_drive_rate', 'contact_form_consistency'
        ]
        
        for feature in contact_features:
            df[feature] = 0.0
        
        # Calculate for each batter-game
        for idx, game in df.iterrows():
            batter_id = game['batter']
            game_date = game['date']
            
            # Get historical data for this batter
            batter_history = statcast_df[
                (statcast_df['batter'] == batter_id) & 
                (statcast_df['date'] < game_date)
            ].copy()
            
            if batter_history.empty:
                continue
            
            # Filter to lookback window
            cutoff_date = game_date - timedelta(days=decay_params.max_lookback_days)
            batter_history = batter_history[batter_history['date'] >= cutoff_date]
            
            if batter_history.empty:
                continue
            
            # Calculate time weights
            batter_history['days_ago'] = (game_date - batter_history['date']).dt.days
            batter_history['weight'] = self._calculate_decay_weight(
                batter_history['days_ago'], decay_params
            )
            
            # Contact quality statistics
            contact_data = batter_history[batter_history['launch_angle'].notna()]
            if not contact_data.empty:
                total_weight = contact_data['weight'].sum()
                
                # Average launch angle
                weighted_la = (contact_data['launch_angle'] * contact_data['weight']).sum()
                df.loc[idx, 'contact_form_avg_la'] = weighted_la / max(total_weight, 1)
                
                # Sweet spot rate (8-32 degrees)
                sweet_spot = contact_data['launch_angle'].between(8, 32)
                sweet_spot_numerator = (sweet_spot * contact_data['weight']).sum()
                df.loc[idx, 'contact_form_sweet_spot_rate'] = sweet_spot_numerator / max(total_weight, 1)
                
                # Line drive rate (10-25 degrees, simplified)
                line_drives = contact_data['launch_angle'].between(10, 25)
                ld_numerator = (line_drives * contact_data['weight']).sum()
                df.loc[idx, 'contact_form_line_drive_rate'] = ld_numerator / max(total_weight, 1)
                
                # Contact consistency (inverse of launch angle standard deviation)
                if len(contact_data) > 3:
                    # Weighted standard deviation
                    mean_la = weighted_la / max(total_weight, 1)
                    variance = ((contact_data['launch_angle'] - mean_la) ** 2 * contact_data['weight']).sum()
                    variance = variance / max(total_weight, 1)
                    std_la = np.sqrt(variance)
                    df.loc[idx, 'contact_form_consistency'] = 1 / (1 + std_la / 10)  # Normalized consistency
        
        logger.info(f"Added {len(contact_features)} contact form features")
        return df
    
    def _add_discipline_form_features(self, statcast_df: pd.DataFrame, 
                                    games_df: pd.DataFrame) -> pd.DataFrame:
        """Add time-weighted plate discipline form features."""
        logger.info("Adding discipline form features...")
        
        df = games_df.copy()
        decay_params = self.decay_params['plate_discipline']
        
        # Initialize discipline columns (these require pitch-by-pitch data)
        discipline_features = [
            'discipline_form_contact_rate', 'discipline_form_z_contact_rate',
            'discipline_form_chase_rate', 'discipline_form_whiff_rate'
        ]
        
        for feature in discipline_features:
            df[feature] = 0.5  # Default neutral values
        
        # Note: Full implementation would require pitch-by-pitch data
        # For now, use simplified versions based on available Statcast data
        
        logger.info(f"Added {len(discipline_features)} discipline form features (simplified)")
        return df
    
    def _add_streak_indicators(self, statcast_df: pd.DataFrame, 
                             games_df: pd.DataFrame) -> pd.DataFrame:
        """Add hot/cold streak indicators with very recent emphasis."""
        logger.info("Adding streak indicators...")
        
        df = games_df.copy()
        decay_params = self.decay_params['hot_cold']
        
        # Initialize streak columns
        streak_features = [
            'hot_streak_indicator', 'cold_streak_indicator', 
            'recent_power_surge', 'recent_slump_indicator',
            'momentum_score'
        ]
        
        for feature in streak_features:
            df[feature] = 0.0
        
        # Calculate for each batter-game
        for idx, game in df.iterrows():
            batter_id = game['batter']
            game_date = game['date']
            
            # Get very recent history (last 30 days)
            cutoff_date = game_date - timedelta(days=decay_params.max_lookback_days)
            recent_history = statcast_df[
                (statcast_df['batter'] == batter_id) & 
                (statcast_df['date'] >= cutoff_date) &
                (statcast_df['date'] < game_date)
            ].copy()
            
            if recent_history.empty:
                continue
            
            # Calculate time weights (very heavy recent weighting)
            recent_history['days_ago'] = (game_date - recent_history['date']).dt.days
            recent_history['weight'] = self._calculate_decay_weight(
                recent_history['days_ago'], decay_params
            )
            
            # Hot streak indicator (recent HR rate well above average)
            total_weight = recent_history['weight'].sum()
            if total_weight > 0:
                weighted_hr_rate = (recent_history['is_hr'] * recent_history['weight']).sum()
                weighted_pa = (recent_history['is_pa'] * recent_history['weight']).sum()
                recent_hr_rate = weighted_hr_rate / max(weighted_pa, 1)
                
                # Compare to league average (~0.12)
                df.loc[idx, 'hot_streak_indicator'] = max(0, (recent_hr_rate - 0.12) * 10)  # Scale factor
                df.loc[idx, 'cold_streak_indicator'] = max(0, (0.08 - recent_hr_rate) * 10)  # Below poor performance
                
                # Power surge (recent high exit velocity)
                ev_data = recent_history[recent_history['launch_speed'].notna()]
                if not ev_data.empty:
                    weighted_ev = (ev_data['launch_speed'] * ev_data['weight']).sum()
                    ev_weight_sum = ev_data['weight'].sum()
                    recent_avg_ev = weighted_ev / max(ev_weight_sum, 1)
                    df.loc[idx, 'recent_power_surge'] = max(0, (recent_avg_ev - 88) / 10)  # Scale factor
                
                # Overall momentum score
                momentum = (df.loc[idx, 'hot_streak_indicator'] - 
                           df.loc[idx, 'cold_streak_indicator'] + 
                           df.loc[idx, 'recent_power_surge'])
                df.loc[idx, 'momentum_score'] = np.tanh(momentum)  # Bounded between -1 and 1
        
        logger.info(f"Added {len(streak_features)} streak indicator features")
        return df
    
    def _add_trend_features(self, statcast_df: pd.DataFrame, 
                          games_df: pd.DataFrame) -> pd.DataFrame:
        """Add features that capture performance trends over time."""
        logger.info("Adding trend features...")
        
        df = games_df.copy()
        
        # Initialize trend columns
        trend_features = [
            'hr_rate_trend_7d', 'hr_rate_trend_14d', 'hr_rate_trend_30d',
            'ev_trend_7d', 'ev_trend_14d', 'form_acceleration'
        ]
        
        for feature in trend_features:
            df[feature] = 0.0
        
        # Calculate for each batter-game
        for idx, game in df.iterrows():
            batter_id = game['batter']
            game_date = game['date']
            
            # Get recent history for trend analysis
            lookback_date = game_date - timedelta(days=45)  # Extended lookback for trends
            history = statcast_df[
                (statcast_df['batter'] == batter_id) & 
                (statcast_df['date'] >= lookback_date) &
                (statcast_df['date'] < game_date)
            ].copy()
            
            if len(history) < 10:  # Need sufficient data for trends
                continue
            
            # Calculate trends for different time periods
            for days in [7, 14, 30]:
                period_start = game_date - timedelta(days=days)
                period_data = history[history['date'] >= period_start]
                
                if len(period_data) >= 5:  # Minimum data for trend
                    # HR rate trend
                    period_data_daily = period_data.groupby('date').agg({
                        'is_hr': 'sum',
                        'is_pa': 'sum'
                    }).reset_index()
                    
                    if len(period_data_daily) >= 3:
                        # Simple linear trend (slope)
                        period_data_daily['days_from_start'] = (
                            period_data_daily['date'] - period_data_daily['date'].min()
                        ).dt.days
                        period_data_daily['hr_rate'] = (
                            period_data_daily['is_hr'] / 
                            period_data_daily['is_pa'].clip(lower=1)
                        )
                        
                        # Linear regression slope (simplified)
                        x = period_data_daily['days_from_start']
                        y = period_data_daily['hr_rate']
                        
                        if len(x) >= 2 and x.std() > 0:
                            slope = np.corrcoef(x, y)[0, 1] * (y.std() / x.std())
                            df.loc[idx, f'hr_rate_trend_{days}d'] = slope
                    
                    # Exit velocity trend
                    ev_data = period_data[period_data['launch_speed'].notna()]
                    if len(ev_data) >= 5:
                        ev_daily = ev_data.groupby('date')['launch_speed'].mean().reset_index()
                        if len(ev_daily) >= 3:
                            ev_daily['days_from_start'] = (
                                ev_daily['date'] - ev_daily['date'].min()
                            ).dt.days
                            
                            x = ev_daily['days_from_start']
                            y = ev_daily['launch_speed']
                            
                            if x.std() > 0:
                                slope = np.corrcoef(x, y)[0, 1] * (y.std() / x.std())
                                df.loc[idx, f'ev_trend_{days}d'] = slope
            
            # Form acceleration (change in recent trend vs longer trend)
            recent_trend = df.loc[idx, 'hr_rate_trend_7d']
            longer_trend = df.loc[idx, 'hr_rate_trend_30d']
            df.loc[idx, 'form_acceleration'] = recent_trend - longer_trend
        
        logger.info(f"Added {len(trend_features)} trend features")
        return df
    
    def _calculate_decay_weight(self, days_ago: pd.Series, 
                              params: DecayParameters) -> pd.Series:
        """Calculate exponential decay weights based on days ago."""
        # Exponential decay: weight = e^(-λ * days)
        # Where λ is chosen so that weight = 0.5 at half_life_days
        decay_constant = np.log(2) / params.half_life_days
        
        weights = np.exp(-decay_constant * days_ago)
        
        # Apply minimum weight floor
        weights = np.maximum(weights, params.min_weight)
        
        return weights

# Export the main class
__all__ = ['RecentFormCalculator', 'DecayParameters']