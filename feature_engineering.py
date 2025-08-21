"""
Feature Engineering Module
=========================

Handles all feature engineering for baseball home run prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pybaseball as pb
from pybaseball import statcast, chadwick_register

from data_utils import DataValidator, StatcastUtils, DateUtils, CacheManager

logger = logging.getLogger(__name__)

# Enable pybaseball caching
pb.cache.enable()

class StatcastDataProcessor:
    """Processes raw Statcast data into structured format."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.validator = DataValidator()
    
    def fetch_statcast_data(self, start_date: str, end_date: str, 
                           use_cache: bool = True) -> pd.DataFrame:
        """Fetch and process Statcast data with caching."""
        cache_id = f"statcast_{start_date}_{end_date}"
        
        # Try to load from cache first
        if use_cache:
            cached_data = self.cache_manager.load_cache(cache_id)
            if cached_data is not None:
                logger.info(f"Loaded Statcast data from cache: {len(cached_data)} rows")
                return cached_data
        
        # Fetch fresh data
        logger.info(f"Fetching Statcast data: {start_date} to {end_date}")
        raw_data = statcast(start_dt=start_date, end_dt=end_date)
        
        if raw_data is None or raw_data.empty:
            raise RuntimeError(f"No Statcast data returned for {start_date} to {end_date}")
        
        # Process the data
        processed_data = self._process_raw_statcast(raw_data)
        
        # Cache the processed data
        if use_cache:
            self.cache_manager.save_cache(processed_data, cache_id)
        
        logger.info(f"Processed Statcast data: {len(processed_data)} rows")
        return processed_data
    
    def _process_raw_statcast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw Statcast data into standardized format."""
        # Get required and optional columns
        column_spec = StatcastUtils.get_required_columns()
        required_cols = column_spec['required']
        optional_cols = column_spec['optional']
        
        # Validate required columns
        self.validator.validate_required_columns(df, required_cols, "Statcast data")
        
        # Select available columns
        available_cols = [col for col in required_cols + optional_cols if col in df.columns]
        df_processed = df[available_cols].copy()
        
        # Add missing optional columns as NaN
        for col in optional_cols:
            if col not in df_processed.columns:
                df_processed[col] = np.nan
        
        # Add derived columns
        df_processed = StatcastUtils.add_derived_columns(df_processed)
        
        # Filter to regular season games
        df_processed = StatcastUtils.filter_regular_season(df_processed)
        
        # Clean and validate
        df_processed = self.validator.validate_date_column(df_processed)
        df_processed = self.validator.clean_numeric_data(df_processed)
        
        return df_processed

class PlayerNameResolver:
    """Resolves player names and IDs using Chadwick register."""
    
    def __init__(self):
        self._chadwick_cache = None
    
    def _get_chadwick_register(self) -> pd.DataFrame:
        """Get Chadwick register with caching."""
        if self._chadwick_cache is None:
            try:
                self._chadwick_cache = chadwick_register()
                self._chadwick_cache['key_mlbam'] = pd.to_numeric(
                    self._chadwick_cache['key_mlbam'], errors='coerce'
                ).astype('Int64')
                logger.info("Loaded Chadwick register")
            except Exception as e:
                logger.error(f"Failed to load Chadwick register: {e}")
                # Create empty DataFrame with expected columns
                self._chadwick_cache = pd.DataFrame(columns=[
                    'key_mlbam', 'name_first', 'name_last'
                ])
        
        return self._chadwick_cache
    
    def add_player_names(self, df: pd.DataFrame, id_col: str = 'batter', 
                        prefix: str = 'batter_') -> pd.DataFrame:
        """Add player names to DataFrame based on MLBAM IDs."""
        chadwick = self._get_chadwick_register()
        
        if chadwick.empty:
            # Fallback: create empty name columns
            df[f'{prefix}name'] = pd.NA
            return df
        
        # Prepare merge
        df_copy = df.copy()
        df_copy[id_col] = pd.to_numeric(df_copy[id_col], errors='coerce').astype('Int64')
        
        # Merge with Chadwick register
        merge_cols = ['key_mlbam', 'name_first', 'name_last']
        merged = df_copy.merge(
            chadwick[merge_cols], 
            left_on=id_col, 
            right_on='key_mlbam', 
            how='left'
        )
        
        # Create full name
        merged[f'{prefix}name'] = (
            merged['name_first'].fillna('') + ' ' + merged['name_last'].fillna('')
        ).str.strip().replace('', pd.NA)
        
        # Clean up temporary columns
        cols_to_drop = ['key_mlbam', 'name_first', 'name_last']
        merged = merged.drop(columns=[col for col in cols_to_drop if col in merged.columns])
        
        return merged

class RollingFeatureCalculator:
    """Calculates rolling features for players."""
    
    @staticmethod
    def calculate_game_based_rolling(df: pd.DataFrame, group_col: str, 
                                   feature_cols: List[str], window: int = 10,
                                   lag: int = 1) -> pd.DataFrame:
        """Calculate game-based rolling features with lag."""
        df_copy = df.copy()
        
        for col in feature_cols:
            if col in df_copy.columns:
                if col in ['pa', 'hr', 'is_pa', 'is_hr']:  # Count-based features
                    df_copy[f'roll{window}_{col}'] = (
                        df_copy.groupby(group_col)[col]
                        .transform(lambda x: x.rolling(window, min_periods=1).sum().shift(lag))
                    )
                else:  # Average-based features
                    df_copy[f'roll{window}_{col}'] = (
                        df_copy.groupby(group_col)[col]
                        .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(lag))
                    )
        
        return df_copy
    
    @staticmethod
    def calculate_time_based_rolling(df: pd.DataFrame, group_col: str, 
                                   feature_cols: List[str], window: str = '30D',
                                   lag: int = 1) -> pd.DataFrame:
        """Calculate time-based rolling features with lag."""
        df_copy = df.copy().sort_values([group_col, 'date'])
        
        def _apply_time_rolling(group):
            group_indexed = group.set_index('date')
            
            for col in feature_cols:
                if col in group_indexed.columns:
                    if col in ['pa', 'hr', 'is_pa', 'is_hr']:  # Count-based
                        rolled = group_indexed[col].rolling(window).sum().shift(lag)
                    else:  # Average-based
                        rolled = group_indexed[col].rolling(window).mean().shift(lag)
                    
                    group_indexed[f'roll{window}_{col}'] = rolled
            
            return group_indexed.reset_index()
        
        result = df_copy.groupby(group_col, group_keys=False).apply(_apply_time_rolling)
        return result

class HandednessSplitsCalculator:
    """Calculates batter vs pitcher handedness splits."""
    
    def __init__(self):
        self.validator = DataValidator()
    
    def calculate_handedness_splits(self, statcast_df: pd.DataFrame, 
                                  batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate handedness splits for batters."""
        # Validate inputs
        required_sc_cols = ['date', 'batter', 'p_throws', 'is_pa', 'is_hr']
        self.validator.validate_required_columns(statcast_df, required_sc_cols, 
                                               "Statcast data for handedness splits")
        
        required_bg_cols = ['date', 'batter']
        self.validator.validate_required_columns(batter_games_df, required_bg_cols,
                                               "Batter games for handedness splits")
        
        # Prepare Statcast data
        sc_splits = statcast_df[required_sc_cols + [
            'launch_speed', 'launch_angle', 'bat_speed', 'attack_angle', 
            'attack_direction', 'swing_path_tilt'
        ]].copy()
        
        # Fill missing handedness data
        sc_splits['p_throws'] = sc_splits['p_throws'].fillna('R')
        
        # Aggregate by batter-hand-date
        aggregated = (
            sc_splits.groupby(['batter', 'p_throws', 'date'], as_index=False)
            .agg({
                'is_pa': 'sum',
                'is_hr': 'sum',
                'launch_speed': 'mean',
                'launch_angle': 'mean',
                'bat_speed': 'mean',
                'attack_angle': 'mean',
                'attack_direction': 'mean',
                'swing_path_tilt': 'mean'
            })
            .rename(columns={
                'is_pa': 'pa',
                'is_hr': 'hr',
                'launch_speed': 'ev',
                'launch_angle': 'la',
                'bat_speed': 'bs',
                'attack_angle': 'aa',
                'attack_direction': 'ad',
                'swing_path_tilt': 'spt'
            })
            .sort_values(['batter', 'p_throws', 'date'])
        )
        
        # Calculate rolling features for each handedness
        feature_cols = ['pa', 'hr', 'ev', 'la', 'bs', 'aa', 'ad', 'spt']
        
        # 10-game rolling (lagged)
        for col in feature_cols:
            aggregated[f'{col}10'] = (
                aggregated.groupby(['batter', 'p_throws'])[col]
                .transform(lambda x: x.rolling(10, min_periods=1).sum().shift(1) 
                          if col in ['pa', 'hr'] else x.rolling(10, min_periods=1).mean().shift(1))
            )
        
        # 30-day time-based rolling (lagged)
        def _calc_30day_rolling(group, col):
            group_indexed = group.set_index('date')
            if col in ['pa', 'hr']:
                return group_indexed[col].rolling('30D').sum().shift(1).reset_index(drop=True)
            else:
                return group_indexed[col].rolling('30D').mean().shift(1).reset_index(drop=True)
        
        for col in feature_cols:
            aggregated[f'{col}30d'] = (
                aggregated.groupby(['batter', 'p_throws'], group_keys=False)
                .apply(lambda g: _calc_30day_rolling(g, col))
                .values
            )
        
        # Calculate rates
        eps = 1e-6
        aggregated['hr_rate10'] = aggregated['hr10'] / (aggregated['pa10'] + eps)
        aggregated['hr_rate30d'] = aggregated['hr30d'] / (aggregated['pa30d'] + eps)
        
        # Pivot to wide format for merging
        pivot_cols = [f'{col}{window}' for col in feature_cols for window in ['10', '30d']] + ['hr_rate10', 'hr_rate30d']
        
        wide_dfs = []
        for col in pivot_cols:
            pivot_df = (
                aggregated.pivot_table(
                    index=['batter', 'date'], 
                    columns='p_throws', 
                    values=col, 
                    aggfunc='first'
                )
                .add_suffix(f'_{col}')
                .reset_index()
            )
            wide_dfs.append(pivot_df)
        
        # Merge all pivot tables
        wide_combined = wide_dfs[0]
        for df in wide_dfs[1:]:
            wide_combined = wide_combined.merge(df, on=['batter', 'date'], how='outer')
        
        # Merge with batter games
        result = batter_games_df.merge(wide_combined, on=['batter', 'date'], how='left')
        
        return result

"""
Fixed Pitcher Feature Calculator
=================================

This replaces the broken PitcherFeatureCalculator class in feature_engineering.py
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PitcherFeatureCalculator:
    """Fixed pitcher-related features calculator."""
    
    def __init__(self):
        self.validator = DataValidator()  # Assuming DataValidator is imported
    
    def infer_game_starters(self, statcast_df: pd.DataFrame) -> pd.DataFrame:
        """Infer starting pitchers from game flow."""
        required_cols = ['game_pk', 'date', 'home_team', 'away_team', 'inning', 'inning_topbot', 'pitcher']
        self.validator.validate_required_columns(statcast_df, required_cols, "Statcast for starter inference")
        
        df = statcast_df[required_cols].copy()
        
        # Determine defensive team
        df['def_team'] = np.where(
            df['inning_topbot'] == 'Top', 
            df['home_team'], 
            df['away_team']
        )
        
        # Sort by inning and take first pitcher for each (game, defense) pair
        starters = (
            df.sort_values(['game_pk', 'inning'])
            .drop_duplicates(['game_pk', 'def_team'], keep='first')
            .rename(columns={'pitcher': 'starter_pitcher'})
            [['game_pk', 'date', 'def_team', 'starter_pitcher']]
        )
        
        logger.info(f"Inferred {len(starters)} game starters")
        return starters
    
    def build_pitcher_profiles(self, statcast_df: pd.DataFrame) -> pd.DataFrame:
        """Build pitcher daily profiles with FIXED rolling metrics."""
        required_cols = ['date', 'pitcher', 'is_hr', 'launch_speed', 'release_speed', 'pitch_type', 'is_pa']
        
        # Add is_pa if not present (we need this for proper rate calculation)
        if 'is_pa' not in required_cols:
            required_cols.append('is_pa')
        
        self.validator.validate_required_columns(statcast_df, required_cols, "Statcast for pitcher profiles")
        
        pitcher_data = statcast_df[required_cols].copy()
        pitcher_data['date'] = pd.to_datetime(pitcher_data['date']).dt.normalize()
        
        # CRITICAL FIX: Aggregate PITCHES FACED, not just daily stats
        # This ensures we calculate rates properly
        daily_stats = (
            pitcher_data.groupby(['pitcher', 'date'], as_index=False)
            .agg({
                'is_hr': 'sum',           # Total HRs allowed
                'is_pa': 'sum',            # Total plate appearances faced
                'launch_speed': 'mean',    # Average exit velocity allowed
                'release_speed': 'mean',   # Average pitch velocity
                'pitch_type': 'count'      # Total pitches thrown
            })
            .rename(columns={
                'is_hr': 'hr_allowed',
                'is_pa': 'pa_faced',       # CRITICAL: Track PAs faced
                'launch_speed': 'ev_allowed',
                'release_speed': 'vel',
                'pitch_type': 'pitches'
            })
            .sort_values(['pitcher', 'date'])
        )
        
        # FIX 1: Calculate PROPER rolling stats with GAME counts, not just sums
        # 10-game rolling (lagged)
        daily_stats['p_roll10_hr'] = (
            daily_stats.groupby('pitcher')['hr_allowed']
            .transform(lambda x: x.rolling(10, min_periods=1).sum().shift(1))
        )
        
        daily_stats['p_roll10_pa'] = (
            daily_stats.groupby('pitcher')['pa_faced']
            .transform(lambda x: x.rolling(10, min_periods=1).sum().shift(1))
        )
        
        daily_stats['p_roll10_ev_allowed'] = (
            daily_stats.groupby('pitcher')['ev_allowed']
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )
        
        daily_stats['p_roll10_vel'] = (
            daily_stats.groupby('pitcher')['vel']
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
        )
        
        # FIX 2: 30-day time-based rolling with PROPER PA tracking
        def _calc_30day_pitcher_rolling(group):
            try:
                group_indexed = group.set_index('date')
                
                # Sum HRs and PAs over 30 days
                group_indexed['p_roll30d_hr'] = (
                    group_indexed['hr_allowed'].rolling('30D').sum().shift(1)
                )
                
                group_indexed['p_roll30d_pa'] = (
                    group_indexed['pa_faced'].rolling('30D').sum().shift(1)
                )
                
                # Average exit velocity over 30 days
                group_indexed['p_roll30d_ev_allowed'] = (
                    group_indexed['ev_allowed'].rolling('30D').mean().shift(1)
                )
                
                return group_indexed.reset_index()
            except Exception as e:
                logger.warning(f"Error in 30-day rolling calculation: {e}")
                group['p_roll30d_hr'] = np.nan
                group['p_roll30d_pa'] = np.nan
                group['p_roll30d_ev_allowed'] = np.nan
                return group
        
        daily_stats = (
            daily_stats.groupby('pitcher', group_keys=False)
            .apply(_calc_30day_pitcher_rolling)
        )
        
        # FIX 3: Calculate PROPER HR rates using PA denominators
        # This is the CRITICAL fix - use actual PAs, not arbitrary numbers
        eps = 1e-6
        
        # 10-game HR rate (HRs per PA)
        daily_stats['p_roll10_hr_rate'] = (
            daily_stats['p_roll10_hr'] / (daily_stats['p_roll10_pa'] + eps)
        )
        
        # 30-day HR rate (HRs per PA)
        daily_stats['p_roll30d_hr_rate'] = (
            daily_stats['p_roll30d_hr'] / (daily_stats['p_roll30d_pa'] + eps)
        )
        
        # Cap rates at reasonable maximum (15% is very high for pitchers)
        daily_stats['p_roll10_hr_rate'] = daily_stats['p_roll10_hr_rate'].clip(0, 0.15)
        daily_stats['p_roll30d_hr_rate'] = daily_stats['p_roll30d_hr_rate'].clip(0, 0.15)
        
        # FIX 4: Calculate pitch mix features with proper handling
        try:
            pitch_mix = self._calculate_pitch_mix_features(pitcher_data)
            pitch_mix['date'] = pd.to_datetime(pitch_mix['date']).dt.normalize()
            
            # Merge with pitch mix
            profiles = daily_stats.merge(pitch_mix, on=['pitcher', 'date'], how='left')
        except Exception as e:
            logger.warning(f"Could not calculate pitch mix features: {e}")
            profiles = daily_stats
        
        # Ensure final date column is properly formatted
        profiles['date'] = pd.to_datetime(profiles['date']).dt.normalize()
        
        # Log summary statistics for validation
        logger.info(f"Built pitcher profiles for {profiles['pitcher'].nunique()} pitchers")
        logger.info(f"Pitcher HR rate stats - Mean: {profiles['p_roll10_hr_rate'].mean():.4f}, "
                   f"Max: {profiles['p_roll10_hr_rate'].max():.4f}, "
                   f"Unique values: {profiles['p_roll10_hr_rate'].nunique()}")
        
        return profiles
    
    def _calculate_pitch_mix_features(self, pitcher_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate pitch mix features for pitchers with proper normalization."""
        try:
            # Daily pitch type counts
            mix_daily = (
                pitcher_data.groupby(['pitcher', 'date', 'pitch_type'])
                .size()
                .unstack('pitch_type', fill_value=0)
                .reset_index()
                .sort_values(['pitcher', 'date'])
            )
            
            # 30-day rolling pitch mix (lagged)
            def _roll30_pitch_mix(group):
                try:
                    group_indexed = group.set_index('date')
                    # Sum pitches over 30 days
                    rolled = group_indexed.rolling('30D', min_periods=1).sum().shift(1)
                    return rolled.reset_index()
                except Exception as e:
                    logger.warning(f"Error in pitch mix rolling calculation: {e}")
                    return group.fillna(0)
            
            mix_rolled = (
                mix_daily.groupby('pitcher', group_keys=False)
                .apply(_roll30_pitch_mix)
            )
            
            # Convert to fractions safely
            pitch_cols = [col for col in mix_rolled.columns if col not in ['pitcher', 'date']]
            
            if pitch_cols:
                # Calculate total pitches for normalization
                mix_total = mix_rolled[pitch_cols].sum(axis=1)
                
                # Avoid division by zero
                mix_total = mix_total.replace(0, np.nan)
                
                # Calculate fractions
                mix_fractions = mix_rolled[pitch_cols].div(mix_total, axis=0)
                
                # Fill NaN with 0 (pitcher hasn't thrown yet)
                mix_fractions = mix_fractions.fillna(0)
                
                # Rename columns
                mix_fractions.columns = [f'mix_{col}' for col in pitch_cols]
                
                # Combine with identifier columns
                result = pd.concat([
                    mix_rolled[['pitcher', 'date']], 
                    mix_fractions
                ], axis=1)
            else:
                # No pitch types found, return minimal DataFrame
                result = mix_rolled[['pitcher', 'date']].copy()
            
            # Ensure date column is properly formatted
            result['date'] = pd.to_datetime(result['date']).dt.normalize()
            
            return result
            
        except Exception as e:
            logger.warning(f"Pitch mix calculation failed: {e}")
            # Return minimal DataFrame with just pitcher and date
            unique_pitchers = pitcher_data[['pitcher', 'date']].drop_duplicates()
            unique_pitchers['date'] = pd.to_datetime(unique_pitchers['date']).dt.normalize()
            return unique_pitchers


# Additional fix for the dataset_builder.py _add_pitcher_features method
def fixed_add_pitcher_features(self, statcast_df: pd.DataFrame, 
                               batter_games_df: pd.DataFrame) -> pd.DataFrame:
    """Fixed version of _add_pitcher_features with proper rate calculations."""
    logger.info("Adding pitcher features with FIXED calculations...")
    
    # Ensure date columns are properly formatted
    statcast_df = self.validator.validate_date_column(statcast_df, 'date')
    batter_games_df = self.validator.validate_date_column(batter_games_df, 'date')
    
    # Infer starting pitchers
    starters = self.pitcher_calculator.infer_game_starters(statcast_df)
    
    # Build pitcher profiles with FIXED calculations
    pitcher_profiles = self.pitcher_calculator.build_pitcher_profiles(statcast_df)
    
    # Add opponent team for each batter
    batter_games = batter_games_df.copy()
    batter_games['opp_team'] = np.where(
        batter_games['bat_team'] == batter_games['home_team'],
        batter_games['away_team'],
        batter_games['home_team']
    )
    
    # Merge with starters
    batter_games = batter_games.merge(
        starters, 
        left_on=['game_pk', 'opp_team'], 
        right_on=['game_pk', 'def_team'], 
        how='left',
        suffixes=('', '_starter')
    )
    
    # Clean up merge artifacts
    batter_games = batter_games.drop(columns=['def_team'], errors='ignore')
    if 'date_starter' in batter_games.columns:
        batter_games = batter_games.drop(columns=['date_starter'], errors='ignore')
    batter_games = batter_games.rename(columns={'starter_pitcher': 'opp_starter'})
    
    # Ensure pitcher profiles have proper date column
    pitcher_profiles = self.validator.validate_date_column(pitcher_profiles, 'date')
    
    # Merge with pitcher profiles
    batter_games = batter_games.merge(
        pitcher_profiles, 
        left_on=['opp_starter', 'date'], 
        right_on=['pitcher', 'date'], 
        how='left',
        suffixes=('', '_pitcher')
    )
    
    # Clean up merge artifacts
    batter_games = batter_games.drop(columns=['pitcher'], errors='ignore')
    if 'date_pitcher' in batter_games.columns:
        batter_games = batter_games.drop(columns=['date_pitcher'], errors='ignore')
    
    # NO NEED FOR ADDITIONAL RATE CALCULATIONS - they're already in pitcher_profiles!
    # The rates are calculated properly in build_pitcher_profiles now
    
    # Validate the pitcher features
    pitcher_rate_cols = ['p_roll10_hr_rate', 'p_roll30d_hr_rate']
    for col in pitcher_rate_cols:
        if col in batter_games.columns:
            col_stats = batter_games[col].describe()
            logger.info(f"{col} stats: Mean={col_stats['mean']:.4f}, "
                       f"Max={col_stats['max']:.4f}, "
                       f"Unique values={batter_games[col].nunique()}")
    
    logger.info("Pitcher features added successfully with proper continuous values")
    return batter_games


# Fix for the PitchMatchupCalculator to properly calculate exp_slg_vs_mix
class FixedPitchMatchupCalculator:
    """Fixed calculation of pitch-level matchup features."""
    
    def calculate_pitch_matchup_features(self, statcast_df: pd.DataFrame, 
                                       batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected SLG vs pitch mix features - FIXED version."""
        required_cols = ['date', 'batter', 'pitch_type', 'events']
        DataValidator.validate_required_columns(statcast_df, required_cols, 
                                              "Statcast for pitch matchup")
        
        pitch_data = statcast_df[required_cols].copy()
        
        # Calculate total bases and at-bats
        pitch_data['tb'] = pitch_data['events'].map({
            'single': 1, 'double': 2, 'triple': 3, 'home_run': 4
        }).fillna(0)
        
        pitch_data['ab'] = pitch_data['events'].isin([
            'single', 'double', 'triple', 'home_run',
            'field_out', 'force_out', 'grounded_into_double_play', 
            'fielders_choice', 'strikeout', 'strikeout_double_play'
        ]).astype(int)
        
        # Aggregate by batter-date-pitch_type
        daily_pitch_stats = (
            pitch_data.groupby(['batter', 'date', 'pitch_type'], as_index=False)
            .agg({'tb': 'sum', 'ab': 'sum'})
        )
        
        # Calculate career rolling stats (200 AB minimum)
        def _calc_career_slg(group):
            group = group.sort_values('date')
            # Rolling sum with lag
            group['tb_career'] = group['tb'].rolling(200, min_periods=10).sum().shift(1)
            group['ab_career'] = group['ab'].rolling(200, min_periods=10).sum().shift(1)
            return group
        
        pitch_slg = (
            daily_pitch_stats.groupby(['batter', 'pitch_type'], group_keys=False)
            .apply(_calc_career_slg)
        )
        
        # Calculate SLG by pitch type
        eps = 1e-6
        pitch_slg['slg'] = pitch_slg['tb_career'] / (pitch_slg['ab_career'] + eps)
        
        # Pivot to wide format
        batter_slg_wide = (
            pitch_slg.pivot_table(
                index=['batter', 'date'],
                columns='pitch_type',
                values='slg',
                aggfunc='first'
            )
            .add_prefix('slg_')
            .reset_index()
        )
        
        # Merge with batter games
        merged = batter_games_df.merge(batter_slg_wide, on=['batter', 'date'], how='left')
        
        # Calculate expected SLG vs mix PROPERLY
        mix_cols = [col for col in merged.columns if col.startswith('mix_')]
        slg_cols = [col for col in merged.columns if col.startswith('slg_')]
        
        if mix_cols and slg_cols:
            # Align pitch types between mix and SLG
            exp_slg_values = []
            
            for idx, row in merged.iterrows():
                weighted_slg = 0
                total_weight = 0
                
                for mix_col in mix_cols:
                    # Extract pitch type from mix column name
                    pitch_type = mix_col.replace('mix_', '')
                    slg_col = f'slg_{pitch_type}'
                    
                    if slg_col in merged.columns:
                        mix_val = row[mix_col]
                        slg_val = row[slg_col]
                        
                        if pd.notna(mix_val) and pd.notna(slg_val):
                            weighted_slg += mix_val * slg_val
                            total_weight += mix_val
                
                # Calculate weighted average
                if total_weight > 0:
                    exp_slg_values.append(weighted_slg / total_weight)
                else:
                    exp_slg_values.append(np.nan)
            
            merged['exp_slg_vs_mix'] = exp_slg_values
            
            # Fill NaN with league average SLG (~0.400)
            merged['exp_slg_vs_mix'] = merged['exp_slg_vs_mix'].fillna(0.400)
            
            # Cap at reasonable values
            merged['exp_slg_vs_mix'] = merged['exp_slg_vs_mix'].clip(0, 1.0)
            
            logger.info(f"Expected SLG vs mix - Mean: {merged['exp_slg_vs_mix'].mean():.4f}, "
                       f"Non-zero: {(merged['exp_slg_vs_mix'] > 0).sum()}")
        else:
            # No mix or SLG data available
            merged['exp_slg_vs_mix'] = 0.400  # League average default
            logger.warning("No pitch mix or SLG data available for matchup calculation")
        
        return merged

class PitchMatchupCalculator:
    """Calculates pitch-level matchup features."""
    
    def calculate_pitch_matchup_features(self, statcast_df: pd.DataFrame, 
                                       batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected SLG vs pitch mix features."""
        required_cols = ['date', 'batter', 'pitch_type', 'events']
        DataValidator.validate_required_columns(statcast_df, required_cols, 
                                              "Statcast for pitch matchup")
        
        pitch_data = statcast_df[required_cols].copy()
        
        # Calculate total bases and at-bats
        pitch_data['tb'] = pitch_data['events'].map({
            'single': 1, 'double': 2, 'triple': 3, 'home_run': 4
        }).fillna(0)
        
        pitch_data['ab'] = pitch_data['events'].isin([
            'single', 'double', 'triple', 'home_run',
            'field_out', 'force_out', 'grounded_into_double_play', 'strikeout'
        ]).astype(int)
        
        # Aggregate by batter-date-pitch_type
        batter_pitch_stats = (
            pitch_data.pivot_table(
                index=['batter', 'date'], 
                columns='pitch_type', 
                values=['tb', 'ab'], 
                aggfunc='sum'
            )
            .fillna(0)
        )
        
        # Flatten column names
        batter_pitch_stats.columns = [f"{metric}_{pitch}" for metric, pitch in batter_pitch_stats.columns]
        batter_pitch_stats = batter_pitch_stats.reset_index().sort_values(['batter', 'date'])
        
        # Calculate career-like rolling SLG by pitch type (200 PA window, lagged)
        numeric_cols = [col for col in batter_pitch_stats.columns if col.startswith(('tb_', 'ab_'))]
        
        def _calc_career_rolling(group):
            group[numeric_cols] = group[numeric_cols].rolling(200, min_periods=1).sum().shift(1)
            return group
        
        batter_pitch_stats = (
            batter_pitch_stats.groupby('batter', group_keys=False)
            .apply(_calc_career_rolling)
        )
        
        # Calculate SLG by pitch type
        pitch_types = set()
        for col in batter_pitch_stats.columns:
            if col.startswith('tb_'):
                pitch_types.add(col[3:])
        
        for pitch_type in pitch_types:
            tb_col = f'tb_{pitch_type}'
            ab_col = f'ab_{pitch_type}'
            
            if tb_col in batter_pitch_stats.columns and ab_col in batter_pitch_stats.columns:
                batter_pitch_stats[f'slg_{pitch_type}'] = (
                    batter_pitch_stats[tb_col] / batter_pitch_stats[ab_col].replace(0, np.nan)
                )
        
        # Keep only SLG columns for merging
        slg_cols = [col for col in batter_pitch_stats.columns if col.startswith('slg_')]
        batter_slg = batter_pitch_stats[['batter', 'date'] + slg_cols].copy()
        
        # Merge with batter games (which should have pitcher mix features)
        merged = batter_games_df.merge(batter_slg, on=['batter', 'date'], how='left')
        
        # Calculate expected SLG vs mix
        mix_cols = [col for col in merged.columns if col.startswith('mix_')]
        slg_cols_in_merged = [col for col in merged.columns if col.startswith('slg_')]
        
        if mix_cols and slg_cols_in_merged:
            # Align pitch types between mix and SLG columns
            slg_matrix = merged[slg_cols_in_merged].values
            mix_matrix = merged[mix_cols].values
            
            # Calculate weighted average (handle NaN values)
            exp_slg = np.nansum(slg_matrix * mix_matrix, axis=1)
            merged['exp_slg_vs_mix'] = exp_slg
        else:
            merged['exp_slg_vs_mix'] = np.nan
        
        return merged

# Import enhanced features
from enhanced_features import BatterPitcherMatchupCalculator

# Export main classes
__all__ = [
    'StatcastDataProcessor', 'PlayerNameResolver', 'RollingFeatureCalculator',
    'HandednessSplitsCalculator', 'PitcherFeatureCalculator', 'PitchMatchupCalculator',
    'BatterPitcherMatchupCalculator'
]