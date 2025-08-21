"""
Enhanced Feature Engineering Module
==================================

Advanced feature engineering for improved home run prediction.
This module implements sophisticated features that go beyond basic rolling statistics.
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
class MatchupStats:
    """Container for batter vs pitcher matchup statistics."""
    plate_appearances: int
    home_runs: int
    hr_rate: float
    avg_exit_velocity: float
    avg_launch_angle: float
    last_encounter_date: Optional[pd.Timestamp]
    encounters_last_year: int

class BatterPitcherMatchupCalculator:
    """Calculate historical batter vs pitcher matchup features."""
    
    def __init__(self):
        self.validator = DataValidator()
        self.matchup_cache = {}
    
    def calculate_matchup_features(self, statcast_df: pd.DataFrame, 
                                 batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate batter vs pitcher matchup history features.
        
        Args:
            statcast_df: Historical Statcast data
            batter_games_df: Batter games for prediction
        
        Returns:
            DataFrame with matchup features added
        """
        logger.info("Calculating batter vs pitcher matchup features...")
        
        # Validate inputs
        required_sc_cols = ['date', 'batter', 'pitcher', 'is_pa', 'is_hr', 'launch_speed', 'launch_angle']
        self.validator.validate_required_columns(statcast_df, required_sc_cols, "Statcast for matchups")
        
        required_bg_cols = ['date', 'batter', 'opp_starter']
        self.validator.validate_required_columns(batter_games_df, required_bg_cols, "Batter games for matchups")
        
        # Ensure date columns are datetime
        statcast_df = statcast_df.copy()
        batter_games_df = batter_games_df.copy()
        statcast_df['date'] = pd.to_datetime(statcast_df['date'])
        batter_games_df['date'] = pd.to_datetime(batter_games_df['date'])
        
        # Build historical matchup database
        matchup_history = self._build_matchup_history(statcast_df)
        
        # Calculate features for each game
        result_df = batter_games_df.copy()
        
        # Initialize feature columns
        default_features = self._get_default_matchup_features()
        for feature in default_features:
            result_df[feature] = default_features[feature]
        
        # Calculate features for each game
        for idx, game in result_df.iterrows():
            batter_id = game['batter']
            pitcher_id = game['opp_starter']
            game_date = game['date']
            
            if not pd.isna(pitcher_id):
                # Get historical matchup stats up to game date
                matchup_features = self._get_matchup_features(
                    matchup_history, batter_id, pitcher_id, game_date
                )
                
                # Update features for this row
                for feature, value in matchup_features.items():
                    result_df.loc[idx, feature] = value
        
        logger.info(f"Added {len(matchup_features)} matchup features")
        logger.info(f"Matchup coverage: {(result_df['matchup_pa_career'] > 0).mean():.1%}")
        
        return result_df
    
    def _build_matchup_history(self, statcast_df: pd.DataFrame) -> pd.DataFrame:
        """Build comprehensive batter vs pitcher historical database."""
        logger.info("Building matchup history database...")
        
        # Aggregate by batter-pitcher-date
        daily_matchups = (
            statcast_df.groupby(['batter', 'pitcher', 'date'])
            .agg({
                'is_pa': 'sum',
                'is_hr': 'sum',
                'launch_speed': 'mean',
                'launch_angle': 'mean'
            })
            .rename(columns={
                'is_pa': 'pa',
                'is_hr': 'hr',
                'launch_speed': 'avg_ev',
                'launch_angle': 'avg_la'
            })
            .reset_index()
            .sort_values(['batter', 'pitcher', 'date'])
        )
        
        # Calculate cumulative stats for each matchup pair
        def _calculate_cumulative_matchup_stats(group):
            group = group.sort_values('date')
            
            # Cumulative stats (lagged by 1 to avoid lookahead)
            group['cumulative_pa'] = group['pa'].cumsum().shift(1, fill_value=0)
            group['cumulative_hr'] = group['hr'].cumsum().shift(1, fill_value=0)
            
            # Rolling averages for contact quality
            group['cumulative_ev'] = group['avg_ev'].expanding().mean().shift(1)
            group['cumulative_la'] = group['avg_la'].expanding().mean().shift(1)
            
            # Recent performance (last 5 encounters)
            group['recent_pa'] = group['pa'].rolling(window=5, min_periods=1).sum().shift(1, fill_value=0)
            group['recent_hr'] = group['hr'].rolling(window=5, min_periods=1).sum().shift(1, fill_value=0)
            
            return group
        
        matchup_history = (
            daily_matchups.groupby(['batter', 'pitcher'])
            .apply(_calculate_cumulative_matchup_stats)
            .reset_index(drop=True)
        )
        
        logger.info(f"Built matchup history for {matchup_history[['batter', 'pitcher']].drop_duplicates().shape[0]} unique matchups")
        
        return matchup_history
    
    def _get_matchup_features(self, matchup_history: pd.DataFrame, 
                            batter_id: int, pitcher_id: int, 
                            game_date: pd.Timestamp) -> Dict[str, float]:
        """Get matchup features for a specific batter-pitcher-date combination."""
        
        # Filter to this matchup pair, up to game date
        matchup_data = matchup_history[
            (matchup_history['batter'] == batter_id) & 
            (matchup_history['pitcher'] == pitcher_id) &
            (matchup_history['date'] < game_date)
        ]
        
        if matchup_data.empty:
            return self._get_default_matchup_features()
        
        # Get most recent stats
        latest_stats = matchup_data.iloc[-1]
        
        # Career matchup stats
        career_pa = latest_stats['cumulative_pa']
        career_hr = latest_stats['cumulative_hr']
        career_hr_rate = career_hr / max(1, career_pa)
        career_avg_ev = latest_stats['cumulative_ev'] if pd.notna(latest_stats['cumulative_ev']) else 0
        career_avg_la = latest_stats['cumulative_la'] if pd.notna(latest_stats['cumulative_la']) else 0
        
        # Recent matchup stats
        recent_pa = latest_stats['recent_pa']
        recent_hr = latest_stats['recent_hr']
        recent_hr_rate = recent_hr / max(1, recent_pa)
        
        # Time since last encounter
        last_encounter_date = latest_stats['date']
        days_since_last = (game_date - last_encounter_date).days
        
        # Encounters in last year
        one_year_ago = game_date - timedelta(days=365)
        recent_encounters = matchup_data[matchup_data['date'] >= one_year_ago]
        encounters_last_year = len(recent_encounters)
        
        return {
            'matchup_pa_career': float(career_pa),
            'matchup_hr_career': float(career_hr),
            'matchup_hr_rate_career': float(career_hr_rate),
            'matchup_avg_ev_career': float(career_avg_ev),
            'matchup_avg_la_career': float(career_avg_la),
            'matchup_pa_recent': float(recent_pa),
            'matchup_hr_recent': float(recent_hr),
            'matchup_hr_rate_recent': float(recent_hr_rate),
            'matchup_days_since_last': float(days_since_last),
            'matchup_encounters_last_year': float(encounters_last_year),
            'matchup_familiarity_score': float(min(career_pa / 20, 1.0))  # 0-1 based on PAs
        }
    
    def _get_default_matchup_features(self) -> Dict[str, float]:
        """Return default values when no matchup history exists."""
        return {
            'matchup_pa_career': 0.0,
            'matchup_hr_career': 0.0,
            'matchup_hr_rate_career': 0.0,
            'matchup_avg_ev_career': 0.0,
            'matchup_avg_la_career': 0.0,
            'matchup_pa_recent': 0.0,
            'matchup_hr_recent': 0.0,
            'matchup_hr_rate_recent': 0.0,
            'matchup_days_since_last': 999.0,  # Large number indicates no previous encounter
            'matchup_encounters_last_year': 0.0,
            'matchup_familiarity_score': 0.0
        }
    
    def calculate_pitcher_similarity_features(self, statcast_df: pd.DataFrame,
                                           batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance vs similar pitcher types.
        
        Groups pitchers by velocity, handedness, and primary pitch type,
        then calculates batter performance vs those groups.
        """
        logger.info("Calculating pitcher similarity features...")
        
        # Build pitcher profiles
        pitcher_profiles = self._build_pitcher_profiles(statcast_df)
        
        # Calculate batter performance vs pitcher groups
        result_df = batter_games_df.copy()
        
        # Initialize similarity feature columns
        default_similarity = self._get_default_similarity_features()
        for feature in default_similarity:
            result_df[feature] = default_similarity[feature]
        
        # Calculate features for each game
        for idx, game in result_df.iterrows():
            batter_id = game['batter']
            pitcher_id = game['opp_starter']
            game_date = game['date']
            
            if not pd.isna(pitcher_id) and pitcher_id in pitcher_profiles:
                similarity_features = self._get_similarity_features(
                    statcast_df, pitcher_profiles, batter_id, pitcher_id, game_date
                )
                
                # Update features for this row
                for feature, value in similarity_features.items():
                    result_df.loc[idx, feature] = value
        logger.info("Added pitcher similarity features")
        
        return result_df
    
    def _build_pitcher_profiles(self, statcast_df: pd.DataFrame) -> Dict[int, Dict]:
        """Build pitcher profiles for similarity matching."""
        
        pitcher_profiles = {}
        
        # Aggregate pitcher characteristics
        pitcher_stats = (
            statcast_df.groupby('pitcher')
            .agg({
                'release_speed': 'mean',
                'p_throws': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'R',
                'pitch_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'FF'
            })
        )
        
        for pitcher_id, stats in pitcher_stats.iterrows():
            pitcher_profiles[pitcher_id] = {
                'avg_velocity': stats['release_speed'] if pd.notna(stats['release_speed']) else 92.0,
                'handedness': stats['p_throws'],
                'primary_pitch': stats['pitch_type'],
                'velocity_tier': 'high' if stats['release_speed'] > 94 else 'medium' if stats['release_speed'] > 90 else 'low'
            }
        
        return pitcher_profiles
    
    def _get_similarity_features(self, statcast_df: pd.DataFrame, pitcher_profiles: Dict,
                               batter_id: int, pitcher_id: int, game_date: pd.Timestamp) -> Dict[str, float]:
        """Calculate batter performance vs similar pitcher types."""
        
        target_profile = pitcher_profiles[pitcher_id]
        
        # Find similar pitchers
        similar_pitchers = []
        for pid, profile in pitcher_profiles.items():
            if pid == pitcher_id:
                continue
            
            # Same handedness
            if profile['handedness'] == target_profile['handedness']:
                similar_pitchers.append(pid)
        
        # Same velocity tier pitchers
        velocity_tier_pitchers = []
        for pid, profile in pitcher_profiles.items():
            if pid == pitcher_id:
                continue
            
            if profile['velocity_tier'] == target_profile['velocity_tier']:
                velocity_tier_pitchers.append(pid)
        
        # Calculate performance vs similar groups
        similar_hand_stats = self._calculate_group_performance(
            statcast_df, batter_id, similar_pitchers, game_date
        )
        
        velocity_tier_stats = self._calculate_group_performance(
            statcast_df, batter_id, velocity_tier_pitchers, game_date
        )
        
        return {
            'vs_similar_hand_pa': similar_hand_stats['pa'],
            'vs_similar_hand_hr': similar_hand_stats['hr'],
            'vs_similar_hand_hr_rate': similar_hand_stats['hr_rate'],
            'vs_similar_velocity_pa': velocity_tier_stats['pa'],
            'vs_similar_velocity_hr': velocity_tier_stats['hr'],
            'vs_similar_velocity_hr_rate': velocity_tier_stats['hr_rate']
        }
    
    def _calculate_group_performance(self, statcast_df: pd.DataFrame, 
                                   batter_id: int, pitcher_list: List[int],
                                   game_date: pd.Timestamp) -> Dict[str, float]:
        """Calculate batter performance vs a group of pitchers."""
        
        # Filter to this batter vs these pitchers, before game date
        group_data = statcast_df[
            (statcast_df['batter'] == batter_id) &
            (statcast_df['pitcher'].isin(pitcher_list)) &
            (statcast_df['date'] < game_date)
        ]
        
        if group_data.empty:
            return {'pa': 0.0, 'hr': 0.0, 'hr_rate': 0.0}
        
        total_pa = group_data['is_pa'].sum()
        total_hr = group_data['is_hr'].sum()
        hr_rate = total_hr / max(1, total_pa)
        
        return {
            'pa': float(total_pa),
            'hr': float(total_hr),
            'hr_rate': float(hr_rate)
        }
    
    def _get_default_similarity_features(self) -> Dict[str, float]:
        """Default values for pitcher similarity features."""
        return {
            'vs_similar_hand_pa': 0.0,
            'vs_similar_hand_hr': 0.0,
            'vs_similar_hand_hr_rate': 0.0,
            'vs_similar_velocity_pa': 0.0,
            'vs_similar_velocity_hr': 0.0,
            'vs_similar_velocity_hr_rate': 0.0
        }

# Import situational features
from situational_features import SituationalFeatureCalculator

# Import weather features
from weather_features import WeatherFeatureCalculator

# Export the calculators
__all__ = ['BatterPitcherMatchupCalculator', 'MatchupStats', 'SituationalFeatureCalculator', 'WeatherFeatureCalculator']