"""
Situational Context Features
===========================

Features that capture game situation, pressure moments, and contextual factors
that influence home run probability.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from data_utils import DataValidator

logger = logging.getLogger(__name__)

class SituationalFeatureCalculator:
    """Calculate situational context features for home run prediction."""
    
    def __init__(self):
        self.validator = DataValidator()
    
    def calculate_situational_features(self, statcast_df: pd.DataFrame, 
                                     batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all situational context features.
        
        Args:
            statcast_df: Historical Statcast data with game situations
            batter_games_df: Batter games to add features to
        
        Returns:
            DataFrame with situational features added
        """
        logger.info("Calculating situational context features...")
        
        # Start with the input dataframe
        result_df = batter_games_df.copy()
        
        # Add game situation features
        result_df = self._add_game_situation_features(statcast_df, result_df)
        
        # Add pressure situation features
        result_df = self._add_pressure_features(statcast_df, result_df)
        
        # Add inning context features
        result_df = self._add_inning_features(statcast_df, result_df)
        
        # Add leverage situation features
        result_df = self._add_leverage_features(statcast_df, result_df)
        
        # Add count-based features
        result_df = self._add_count_features(statcast_df, result_df)
        
        logger.info("Situational feature calculation complete")
        return result_df
    
    def _add_game_situation_features(self, statcast_df: pd.DataFrame, 
                                   batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Add basic game situation features."""
        logger.info("Adding game situation features...")
        
        result_df = batter_games_df.copy()
        
        # Initialize default values
        default_features = {
            'avg_runners_on_base': 0.0,
            'risp_percentage': 0.0,  # Runners in scoring position
            'bases_empty_percentage': 0.0,
            'avg_score_differential': 0.0,
            'trailing_percentage': 0.0,
            'leading_percentage': 0.0,
            'tied_percentage': 0.0
        }
        
        for feature in default_features:
            result_df[feature] = default_features[feature]
        
        # Calculate features for each batter
        batter_stats = self._calculate_batter_situation_stats(statcast_df)
        
        # Merge with result dataframe
        for idx, row in result_df.iterrows():
            batter_id = row['batter']
            if batter_id in batter_stats:
                stats = batter_stats[batter_id]
                for feature, value in stats.items():
                    result_df.loc[idx, feature] = value
        
        logger.info(f"Added game situation features with coverage: {len(batter_stats)} batters")
        return result_df
    
    def _calculate_batter_situation_stats(self, statcast_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Calculate situational statistics for each batter."""
        
        # Required columns for situation analysis
        required_cols = ['batter', 'is_pa']
        available_cols = [col for col in required_cols if col in statcast_df.columns]
        
        if len(available_cols) < len(required_cols):
            logger.warning(f"Missing columns for situation analysis: {set(required_cols) - set(available_cols)}")
            return {}
        
        batter_stats = {}
        
        # Group by batter
        for batter_id, batter_data in statcast_df.groupby('batter'):
            # Only count plate appearances
            pa_data = batter_data[batter_data['is_pa'] == 1]
            
            if len(pa_data) == 0:
                continue
            
            stats = {}
            
            # Runners on base analysis
            if 'on_3b' in pa_data.columns and 'on_2b' in pa_data.columns and 'on_1b' in pa_data.columns:
                # Count runners on base
                pa_data_copy = pa_data.copy()
                pa_data_copy['runners_on'] = (
                    pa_data_copy['on_1b'].notna().astype(int) +
                    pa_data_copy['on_2b'].notna().astype(int) +
                    pa_data_copy['on_3b'].notna().astype(int)
                )
                
                stats['avg_runners_on_base'] = pa_data_copy['runners_on'].mean()
                
                # Runners in scoring position (2nd or 3rd base)
                risp_situations = (pa_data_copy['on_2b'].notna() | pa_data_copy['on_3b'].notna()).sum()
                stats['risp_percentage'] = risp_situations / len(pa_data_copy)
                
                # Bases empty
                bases_empty = (pa_data_copy['runners_on'] == 0).sum()
                stats['bases_empty_percentage'] = bases_empty / len(pa_data_copy)
            else:
                # Use defaults if runner data not available
                stats['avg_runners_on_base'] = 0.0
                stats['risp_percentage'] = 0.0
                stats['bases_empty_percentage'] = 1.0
            
            # Score differential analysis
            if 'post_away_score' in pa_data.columns and 'post_home_score' in pa_data.columns and 'inning_topbot' in pa_data.columns:
                pa_data_copy = pa_data.copy()
                
                # Calculate score differential from batter's perspective
                pa_data_copy['score_diff'] = np.where(
                    pa_data_copy['inning_topbot'] == 'Top',  # Away team batting
                    pa_data_copy['post_away_score'] - pa_data_copy['post_home_score'],
                    pa_data_copy['post_home_score'] - pa_data_copy['post_away_score']  # Home team batting
                )
                
                stats['avg_score_differential'] = pa_data_copy['score_diff'].mean()
                stats['trailing_percentage'] = (pa_data_copy['score_diff'] < 0).mean()
                stats['leading_percentage'] = (pa_data_copy['score_diff'] > 0).mean()
                stats['tied_percentage'] = (pa_data_copy['score_diff'] == 0).mean()
            else:
                # Use defaults if score data not available
                stats['avg_score_differential'] = 0.0
                stats['trailing_percentage'] = 0.33
                stats['leading_percentage'] = 0.33
                stats['tied_percentage'] = 0.33
            
            batter_stats[batter_id] = stats
        
        return batter_stats
    
    def _add_pressure_features(self, statcast_df: pd.DataFrame, 
                             batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Add pressure situation features."""
        logger.info("Adding pressure situation features...")
        
        result_df = batter_games_df.copy()
        
        # Initialize pressure features
        pressure_features = {
            'clutch_pa_percentage': 0.0,        # High leverage situations
            'clutch_hr_rate': 0.0,              # HR rate in clutch situations
            'late_inning_pa_percentage': 0.0,   # 7th inning or later
            'late_inning_hr_rate': 0.0,
            'close_game_pa_percentage': 0.0,    # Within 2 runs
            'close_game_hr_rate': 0.0,
            'pressure_performance_index': 0.0   # Overall pressure performance
        }
        
        for feature in pressure_features:
            result_df[feature] = pressure_features[feature]
        
        # Calculate pressure stats for each batter
        pressure_stats = self._calculate_pressure_stats(statcast_df)
        
        # Merge with result dataframe
        for idx, row in result_df.iterrows():
            batter_id = row['batter']
            if batter_id in pressure_stats:
                stats = pressure_stats[batter_id]
                for feature, value in stats.items():
                    result_df.loc[idx, feature] = value
        
        logger.info(f"Added pressure features for {len(pressure_stats)} batters")
        return result_df
    
    def _calculate_pressure_stats(self, statcast_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Calculate pressure situation statistics."""
        
        pressure_stats = {}
        
        for batter_id, batter_data in statcast_df.groupby('batter'):
            pa_data = batter_data[batter_data['is_pa'] == 1]
            
            if len(pa_data) == 0:
                continue
            
            stats = {}
            
            # Late inning situations (7th inning or later)
            if 'inning' in pa_data.columns:
                late_inning_data = pa_data[pa_data['inning'] >= 7]
                stats['late_inning_pa_percentage'] = len(late_inning_data) / len(pa_data)
                
                if len(late_inning_data) > 0:
                    stats['late_inning_hr_rate'] = late_inning_data['is_hr'].mean()
                else:
                    stats['late_inning_hr_rate'] = 0.0
            else:
                stats['late_inning_pa_percentage'] = 0.0
                stats['late_inning_hr_rate'] = 0.0
            
            # Close game situations (within 2 runs)
            close_game_data = self._identify_close_games(pa_data)
            if close_game_data is not None and len(close_game_data) > 0:
                stats['close_game_pa_percentage'] = len(close_game_data) / len(pa_data)
                stats['close_game_hr_rate'] = close_game_data['is_hr'].mean()
            else:
                stats['close_game_pa_percentage'] = 0.5  # Assume 50% of games are close
                stats['close_game_hr_rate'] = pa_data['is_hr'].mean()
            
            # High leverage situations (simplified)
            clutch_data = self._identify_clutch_situations(pa_data)
            if clutch_data is not None and len(clutch_data) > 0:
                stats['clutch_pa_percentage'] = len(clutch_data) / len(pa_data)
                stats['clutch_hr_rate'] = clutch_data['is_hr'].mean()
            else:
                stats['clutch_pa_percentage'] = 0.2  # Assume 20% of PAs are clutch
                stats['clutch_hr_rate'] = pa_data['is_hr'].mean()
            
            # Pressure performance index (clutch performance vs overall)
            overall_hr_rate = pa_data['is_hr'].mean()
            if overall_hr_rate > 0:
                stats['pressure_performance_index'] = stats['clutch_hr_rate'] / overall_hr_rate
            else:
                stats['pressure_performance_index'] = 1.0
            
            pressure_stats[batter_id] = stats
        
        return pressure_stats
    
    def _identify_close_games(self, pa_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Identify plate appearances in close games (within 2 runs)."""
        
        required_cols = ['post_away_score', 'post_home_score']
        if not all(col in pa_data.columns for col in required_cols):
            return None
        
        pa_data_copy = pa_data.copy()
        pa_data_copy['score_diff'] = abs(pa_data_copy['post_away_score'] - pa_data_copy['post_home_score'])
        
        return pa_data_copy[pa_data_copy['score_diff'] <= 2]
    
    def _identify_clutch_situations(self, pa_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Identify clutch/high leverage situations."""
        
        # Simplified clutch definition: late innings + close game + runners on base
        clutch_conditions = []
        
        # Late innings (7+)
        if 'inning' in pa_data.columns:
            clutch_conditions.append(pa_data['inning'] >= 7)
        
        # Close game
        if 'post_away_score' in pa_data.columns and 'post_home_score' in pa_data.columns:
            score_diff = abs(pa_data['post_away_score'] - pa_data['post_home_score'])
            clutch_conditions.append(score_diff <= 3)
        
        # Runners on base
        if 'on_1b' in pa_data.columns or 'on_2b' in pa_data.columns or 'on_3b' in pa_data.columns:
            runners_on = (
                pa_data.get('on_1b', pd.Series()).notna() |
                pa_data.get('on_2b', pd.Series()).notna() |
                pa_data.get('on_3b', pd.Series()).notna()
            )
            clutch_conditions.append(runners_on)
        
        if len(clutch_conditions) >= 2:  # Need at least 2 conditions
            clutch_mask = clutch_conditions[0]
            for condition in clutch_conditions[1:]:
                clutch_mask = clutch_mask & condition
            
            return pa_data[clutch_mask]
        
        return None
    
    def _add_inning_features(self, statcast_df: pd.DataFrame, 
                           batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Add inning-specific features."""
        logger.info("Adding inning-specific features...")
        
        result_df = batter_games_df.copy()
        
        # Inning performance features
        inning_features = {
            'first_inning_hr_rate': 0.0,
            'middle_innings_hr_rate': 0.0,  # 3rd-6th
            'late_innings_hr_rate': 0.0,    # 7th-9th
            'extra_innings_hr_rate': 0.0,   # 10th+
            'first_pa_hr_rate': 0.0,        # First PA of the game
            'leadoff_inning_hr_rate': 0.0   # First batter of any inning
        }
        
        for feature in inning_features:
            result_df[feature] = inning_features[feature]
        
        # Calculate inning stats
        inning_stats = self._calculate_inning_stats(statcast_df)
        
        # Merge with result dataframe
        for idx, row in result_df.iterrows():
            batter_id = row['batter']
            if batter_id in inning_stats:
                stats = inning_stats[batter_id]
                for feature, value in stats.items():
                    result_df.loc[idx, feature] = value
        
        logger.info(f"Added inning features for {len(inning_stats)} batters")
        return result_df
    
    def _calculate_inning_stats(self, statcast_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Calculate inning-specific performance statistics."""
        
        inning_stats = {}
        
        for batter_id, batter_data in statcast_df.groupby('batter'):
            pa_data = batter_data[batter_data['is_pa'] == 1]
            
            if len(pa_data) == 0:
                continue
            
            stats = {}
            
            if 'inning' in pa_data.columns:
                # First inning
                first_inning = pa_data[pa_data['inning'] == 1]
                stats['first_inning_hr_rate'] = first_inning['is_hr'].mean() if len(first_inning) > 0 else 0.0
                
                # Middle innings (3-6)
                middle_innings = pa_data[pa_data['inning'].between(3, 6)]
                stats['middle_innings_hr_rate'] = middle_innings['is_hr'].mean() if len(middle_innings) > 0 else 0.0
                
                # Late innings (7-9)
                late_innings = pa_data[pa_data['inning'].between(7, 9)]
                stats['late_innings_hr_rate'] = late_innings['is_hr'].mean() if len(late_innings) > 0 else 0.0
                
                # Extra innings (10+)
                extra_innings = pa_data[pa_data['inning'] >= 10]
                stats['extra_innings_hr_rate'] = extra_innings['is_hr'].mean() if len(extra_innings) > 0 else 0.0
            else:
                # Use overall rate if inning data not available
                overall_rate = pa_data['is_hr'].mean()
                stats['first_inning_hr_rate'] = overall_rate
                stats['middle_innings_hr_rate'] = overall_rate
                stats['late_innings_hr_rate'] = overall_rate
                stats['extra_innings_hr_rate'] = overall_rate
            
            # Leadoff situations
            if 'at_bat_number' in pa_data.columns:
                # First PA of game
                first_pa = pa_data[pa_data['at_bat_number'] == 1]
                stats['first_pa_hr_rate'] = first_pa['is_hr'].mean() if len(first_pa) > 0 else 0.0
                
                # Leadoff of any inning (simplified - would need more complex logic for real leadoff)
                stats['leadoff_inning_hr_rate'] = stats['first_pa_hr_rate']  # Simplified
            else:
                stats['first_pa_hr_rate'] = pa_data['is_hr'].mean()
                stats['leadoff_inning_hr_rate'] = pa_data['is_hr'].mean()
            
            inning_stats[batter_id] = stats
        
        return inning_stats
    
    def _add_leverage_features(self, statcast_df: pd.DataFrame, 
                             batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Add leverage index based features."""
        logger.info("Adding leverage situation features...")
        
        result_df = batter_games_df.copy()
        
        # Leverage features
        leverage_features = {
            'high_leverage_pa_pct': 0.2,     # Percentage of PAs in high leverage
            'high_leverage_hr_rate': 0.0,    # HR rate in high leverage situations
            'medium_leverage_pa_pct': 0.6,   # Medium leverage
            'medium_leverage_hr_rate': 0.0,
            'low_leverage_pa_pct': 0.2,      # Low leverage
            'low_leverage_hr_rate': 0.0,
            'leverage_performance_ratio': 1.0  # High leverage vs low leverage performance
        }
        
        for feature in leverage_features:
            result_df[feature] = leverage_features[feature]
        
        # Calculate leverage stats (simplified without actual leverage index)
        leverage_stats = self._calculate_leverage_stats(statcast_df)
        
        # Merge with result dataframe
        for idx, row in result_df.iterrows():
            batter_id = row['batter']
            if batter_id in leverage_stats:
                stats = leverage_stats[batter_id]
                for feature, value in stats.items():
                    result_df.loc[idx, feature] = value
        
        logger.info(f"Added leverage features for {len(leverage_stats)} batters")
        return result_df
    
    def _calculate_leverage_stats(self, statcast_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Calculate leverage-based statistics (simplified)."""
        
        leverage_stats = {}
        
        for batter_id, batter_data in statcast_df.groupby('batter'):
            pa_data = batter_data[batter_data['is_pa'] == 1]
            
            if len(pa_data) == 0:
                continue
            
            # Simplified leverage calculation based on game situation
            # High leverage: late innings + close game + runners in scoring position
            # Medium leverage: close game OR late innings OR runners on
            # Low leverage: everything else
            
            stats = {}
            overall_hr_rate = pa_data['is_hr'].mean()
            
            # Identify leverage situations
            high_lev, med_lev, low_lev = self._classify_leverage_situations(pa_data)
            
            if high_lev is not None:
                stats['high_leverage_pa_pct'] = len(high_lev) / len(pa_data)
                stats['high_leverage_hr_rate'] = high_lev['is_hr'].mean() if len(high_lev) > 0 else overall_hr_rate
            else:
                stats['high_leverage_pa_pct'] = 0.2
                stats['high_leverage_hr_rate'] = overall_hr_rate
            
            if med_lev is not None:
                stats['medium_leverage_pa_pct'] = len(med_lev) / len(pa_data)
                stats['medium_leverage_hr_rate'] = med_lev['is_hr'].mean() if len(med_lev) > 0 else overall_hr_rate
            else:
                stats['medium_leverage_pa_pct'] = 0.6
                stats['medium_leverage_hr_rate'] = overall_hr_rate
            
            if low_lev is not None:
                stats['low_leverage_pa_pct'] = len(low_lev) / len(pa_data)
                stats['low_leverage_hr_rate'] = low_lev['is_hr'].mean() if len(low_lev) > 0 else overall_hr_rate
            else:
                stats['low_leverage_pa_pct'] = 0.2
                stats['low_leverage_hr_rate'] = overall_hr_rate
            
            # Performance ratio
            if stats['low_leverage_hr_rate'] > 0:
                stats['leverage_performance_ratio'] = stats['high_leverage_hr_rate'] / stats['low_leverage_hr_rate']
            else:
                stats['leverage_performance_ratio'] = 1.0
            
            leverage_stats[batter_id] = stats
        
        return leverage_stats
    
    def _classify_leverage_situations(self, pa_data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Classify plate appearances by leverage (simplified)."""
        
        # Check what data we have available
        has_inning = 'inning' in pa_data.columns
        has_score = 'post_away_score' in pa_data.columns and 'post_home_score' in pa_data.columns
        has_runners = any(col in pa_data.columns for col in ['on_1b', 'on_2b', 'on_3b'])
        
        if not any([has_inning, has_score, has_runners]):
            return None, None, None
        
        pa_copy = pa_data.copy()
        
        # Initialize leverage scores
        pa_copy['leverage_score'] = 0
        
        # Late innings (+2)
        if has_inning:
            pa_copy.loc[pa_copy['inning'] >= 7, 'leverage_score'] += 2
            pa_copy.loc[pa_copy['inning'] >= 9, 'leverage_score'] += 1
        
        # Close game (+2)
        if has_score:
            score_diff = abs(pa_copy['post_away_score'] - pa_copy['post_home_score'])
            pa_copy.loc[score_diff <= 1, 'leverage_score'] += 2
            pa_copy.loc[score_diff <= 3, 'leverage_score'] += 1
        
        # Runners in scoring position (+1)
        if has_runners:
            risp = (pa_copy.get('on_2b', pd.Series()).notna() | 
                   pa_copy.get('on_3b', pd.Series()).notna())
            pa_copy.loc[risp, 'leverage_score'] += 1
        
        # Classify by leverage score
        high_leverage = pa_copy[pa_copy['leverage_score'] >= 4]
        medium_leverage = pa_copy[pa_copy['leverage_score'].between(2, 3)]
        low_leverage = pa_copy[pa_copy['leverage_score'] <= 1]
        
        return (high_leverage if len(high_leverage) > 0 else None,
                medium_leverage if len(medium_leverage) > 0 else None,
                low_leverage if len(low_leverage) > 0 else None)
    
    def _add_count_features(self, statcast_df: pd.DataFrame, 
                          batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Add count-specific features."""
        logger.info("Adding count-specific features...")
        
        result_df = batter_games_df.copy()
        
        # Count features
        count_features = {
            'hitters_count_hr_rate': 0.0,     # 2-0, 3-1, 3-0 counts
            'pitchers_count_hr_rate': 0.0,    # 0-2, 1-2 counts
            'even_count_hr_rate': 0.0,        # 0-0, 1-1, 2-2 counts
            'two_strike_hr_rate': 0.0,        # Any 2-strike count
            'ahead_in_count_pct': 0.0,        # Percentage of PAs ahead in count
            'behind_in_count_pct': 0.0        # Percentage of PAs behind in count
        }
        
        for feature in count_features:
            result_df[feature] = count_features[feature]
        
        # Calculate count stats
        count_stats = self._calculate_count_stats(statcast_df)
        
        # Merge with result dataframe
        for idx, row in result_df.iterrows():
            batter_id = row['batter']
            if batter_id in count_stats:
                stats = count_stats[batter_id]
                for feature, value in stats.items():
                    result_df.loc[idx, feature] = value
        
        logger.info(f"Added count features for {len(count_stats)} batters")
        return result_df
    
    def _calculate_count_stats(self, statcast_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Calculate count-specific performance statistics."""
        
        count_stats = {}
        
        # Check if count data is available
        if 'balls' not in statcast_df.columns or 'strikes' not in statcast_df.columns:
            logger.warning("Count data (balls/strikes) not available - using defaults")
            return {}
        
        for batter_id, batter_data in statcast_df.groupby('batter'):
            # Use all pitches, not just PAs, for count analysis
            pitch_data = batter_data.copy()
            
            if len(pitch_data) == 0:
                continue
            
            stats = {}
            
            # Classify counts
            hitters_counts = pitch_data[
                ((pitch_data['balls'] == 2) & (pitch_data['strikes'] == 0)) |
                ((pitch_data['balls'] == 3) & (pitch_data['strikes'] == 1)) |
                ((pitch_data['balls'] == 3) & (pitch_data['strikes'] == 0))
            ]
            
            pitchers_counts = pitch_data[
                ((pitch_data['balls'] == 0) & (pitch_data['strikes'] == 2)) |
                ((pitch_data['balls'] == 1) & (pitch_data['strikes'] == 2))
            ]
            
            even_counts = pitch_data[
                ((pitch_data['balls'] == 0) & (pitch_data['strikes'] == 0)) |
                ((pitch_data['balls'] == 1) & (pitch_data['strikes'] == 1)) |
                ((pitch_data['balls'] == 2) & (pitch_data['strikes'] == 2))
            ]
            
            two_strike_counts = pitch_data[pitch_data['strikes'] == 2]
            
            # Calculate HR rates for each count type
            stats['hitters_count_hr_rate'] = hitters_counts['is_hr'].mean() if len(hitters_counts) > 0 else 0.0
            stats['pitchers_count_hr_rate'] = pitchers_counts['is_hr'].mean() if len(pitchers_counts) > 0 else 0.0
            stats['even_count_hr_rate'] = even_counts['is_hr'].mean() if len(even_counts) > 0 else 0.0
            stats['two_strike_hr_rate'] = two_strike_counts['is_hr'].mean() if len(two_strike_counts) > 0 else 0.0
            
            # Count advantage percentages
            total_pitches = len(pitch_data)
            if total_pitches > 0:
                stats['ahead_in_count_pct'] = len(hitters_counts) / total_pitches
                stats['behind_in_count_pct'] = len(pitchers_counts) / total_pitches
            else:
                stats['ahead_in_count_pct'] = 0.0
                stats['behind_in_count_pct'] = 0.0
            
            count_stats[batter_id] = stats
        
        return count_stats

# Export the main class
__all__ = ['SituationalFeatureCalculator']