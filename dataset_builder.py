"""
Dataset Builder Module
=====================

Main dataset construction pipeline for baseball home run prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path

from config import config
from data_utils import DataValidator, CacheManager, DateUtils
from feature_engineering import (
    StatcastDataProcessor, PlayerNameResolver, RollingFeatureCalculator,
    HandednessSplitsCalculator, PitcherFeatureCalculator, PitchMatchupCalculator,
    BatterPitcherMatchupCalculator
)
from optimize_enhanced_features import patch_enhanced_features
from optimize_recent_form import patch_recent_form_features
from situational_features import SituationalFeatureCalculator
from weather_features import WeatherFeatureCalculator
from recent_form_features import RecentFormCalculator
from streak_momentum_features import StreakMomentumCalculator
from ballpark_features import BallparkFeatureCalculator
from temporal_fatigue_features import TemporalFatigueCalculator
from feature_interactions import FeatureInteractionCalculator

logger = logging.getLogger(__name__)

class PregameDatasetBuilder:
    """
    Production-ready dataset builder for pregame home run predictions.
    
    Features:
    - Batter rolling performance (10 games, 30 days)
    - Handedness splits (batter vs pitcher handedness)
    - Pitcher quality and pitch mix
    - Park factors and weather
    - Robust caching and error handling
    """
    
    def __init__(self, start_date: str = None, end_date: str = None, 
                 weather_csv_path: Optional[str] = None):
        self.start_date = start_date or config.DEFAULT_START_DATE
        self.end_date = end_date or config.DEFAULT_END_DATE
        self.weather_csv_path = weather_csv_path
        
        # Initialize components
        self.cache_manager = CacheManager()
        self.validator = DataValidator()
        self.statcast_processor = StatcastDataProcessor(self.cache_manager)
        self.name_resolver = PlayerNameResolver()
        self.rolling_calculator = RollingFeatureCalculator()
        self.handedness_calculator = HandednessSplitsCalculator()
        self.pitcher_calculator = PitcherFeatureCalculator()
        self.matchup_calculator = PitchMatchupCalculator()
        self.batter_pitcher_matchup_calculator = BatterPitcherMatchupCalculator()
        self.situational_calculator = SituationalFeatureCalculator()
        self.weather_calculator = WeatherFeatureCalculator()
        self.recent_form_calculator = RecentFormCalculator()
        self.streak_momentum_calculator = StreakMomentumCalculator()
        self.ballpark_calculator = BallparkFeatureCalculator()
        self.temporal_fatigue_calculator = TemporalFatigueCalculator()
        self.interaction_calculator = FeatureInteractionCalculator()
        
        # Apply optimizations for large datasets
        patch_enhanced_features()
        patch_recent_form_features()
        
        logger.info(f"Dataset builder initialized: {self.start_date} → {self.end_date}")
        logger.info("✅ Enhanced features optimized for large datasets")
        logger.info("✅ Recent form features optimized for large datasets")
    
    def build_dataset(self, force_rebuild: bool = False, 
                     cache_format: str = "parquet") -> pd.DataFrame:
        """
        Build complete pregame dataset with caching.
        
        Args:
            force_rebuild: Force rebuild even if cache exists
            cache_format: Format for caching ("parquet" or "csv")
        
        Returns:
            Complete dataset ready for modeling
        """
        # Check cache first
        cache_id = f"pregame_dataset_{self.start_date}_{self.end_date}"
        
        if not force_rebuild:
            cached_dataset = self.cache_manager.load_cache(cache_id, cache_format)
            if cached_dataset is not None:
                logger.info(f"Loaded complete dataset from cache: {len(cached_dataset)} rows")
                return cached_dataset
        
        logger.info("Building dataset from scratch...")
        
        try:
            # Step 1: Fetch and process Statcast data
            statcast_data = self.statcast_processor.fetch_statcast_data(
                self.start_date, self.end_date, use_cache=not force_rebuild
            )
            
            # Step 2: Create batter-game aggregations
            batter_games = self._create_batter_game_features(statcast_data)
            
            # Step 3: Add handedness splits
            batter_games = self.handedness_calculator.calculate_handedness_splits(
                statcast_data, batter_games
            )
            
            # Step 4: Add pitcher features
            batter_games = self._add_pitcher_features(statcast_data, batter_games)
            
            # Step 5: Add pitch matchup features
            batter_games = self.matchup_calculator.calculate_pitch_matchup_features(
                statcast_data, batter_games
            )
            
            # Step 5a: Add batter vs pitcher matchup history features
            batter_games = self.batter_pitcher_matchup_calculator.calculate_matchup_features(
                statcast_data, batter_games
            )
            
            # Step 5b: Add pitcher similarity features
            batter_games = self.batter_pitcher_matchup_calculator.calculate_pitcher_similarity_features(
                statcast_data, batter_games
            )
            
            # Step 5c: Add situational context features
            batter_games = self.situational_calculator.calculate_situational_features(
                statcast_data, batter_games
            )
            
            # Step 6: Add enhanced weather features
            batter_games = self.weather_calculator.calculate_weather_features(
                batter_games, 
                use_real_weather=True,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Step 6a: Add recent form features with time decay
            batter_games = self.recent_form_calculator.calculate_recent_form_features(
                statcast_data, batter_games
            )
            
            # Step 6b: Add streak and momentum features  
            batter_games = self.streak_momentum_calculator.calculate_streak_features(
                batter_games
            )
            
            # Step 6c: Add advanced ballpark features
            batter_games = self.ballpark_calculator.calculate_ballpark_features(
                batter_games
            )
            
            # Step 6d: Add temporal and fatigue features
            batter_games = self.temporal_fatigue_calculator.calculate_temporal_fatigue_features(
                batter_games
            )
            
            # Step 6e: Add feature interaction terms
            batter_games = self.interaction_calculator.calculate_feature_interactions(
                batter_games
            )
            
            # Step 7: Finalize dataset
            final_dataset = self._finalize_dataset(batter_games)
            
            # Cache the result
            self.cache_manager.save_cache(final_dataset, cache_id, cache_format)
            
            logger.info(f"Dataset construction complete: {len(final_dataset)} rows")
            return final_dataset
            
        except Exception as e:
            logger.error(f"Dataset construction failed: {e}")
            raise
    
    def _create_batter_game_features(self, statcast_df: pd.DataFrame) -> pd.DataFrame:
        """Create batter-game level features with rolling windows."""
        logger.info("Creating batter-game features...")
        
        # Validate required columns
        required_cols = ['date', 'game_pk', 'batter', 'home_team', 'away_team', 'bat_team']
        self.validator.validate_required_columns(statcast_df, required_cols, 
                                               "Statcast for batter-game features")
        
        # Aggregate to batter-game level
        agg_dict = {
            'is_pa': 'sum',
            'is_hr': 'sum',
            'launch_speed': 'mean',
            'launch_angle': 'mean',
        }
        
        # Add bat tracking features if available
        bat_tracking_cols = ['bat_speed', 'attack_angle', 'attack_direction', 'swing_path_tilt']
        for col in bat_tracking_cols:
            if col in statcast_df.columns:
                agg_dict[col] = 'mean'
        
        batter_games = (
            statcast_df.groupby(['date', 'game_pk', 'batter', 'home_team', 'away_team', 'bat_team'])
            .agg(agg_dict)
            .reset_index()
            .rename(columns={'is_pa': 'pa', 'is_hr': 'hr'})
        )
        
        # Add derived features
        batter_games['season'] = batter_games['date'].apply(DateUtils.get_season_from_date)
        batter_games['stadium'] = batter_games['home_team'].map(config.TEAM_TO_STADIUM)
        batter_games['park_factor'] = batter_games['stadium'].map(config.PARK_FACTORS).fillna(1.0)
        
        # Sort for rolling calculations
        batter_games = batter_games.sort_values(['batter', 'date'])
        
        # Calculate rolling features
        batter_games = self._add_rolling_batter_features(batter_games)
        
        # Add target variable
        batter_games['home_runs'] = batter_games['hr']
        batter_games['hit_hr'] = (batter_games['hr'] > 0).astype(int)
        
        logger.info(f"Created {len(batter_games)} batter-game rows")
        return batter_games
    
    def _add_rolling_batter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling batter features."""
        df = df.copy()
        
        # Basic feature columns
        basic_features = ['pa', 'hr', 'launch_speed', 'launch_angle']
        
        # Bat tracking features (if available)
        bt_features = ['bat_speed', 'attack_angle', 'attack_direction', 'swing_path_tilt']
        available_bt = [col for col in bt_features if col in df.columns]
        
        all_features = basic_features + available_bt
        
        # 10-game rolling (lagged)
        for feature in all_features:
            if feature in df.columns:
                if feature in ['pa', 'hr']:
                    df[f'roll10_{feature}'] = (
                        df.groupby('batter')[feature]
                        .transform(lambda x: x.rolling(10, min_periods=1).sum().shift(1))
                    )
                else:
                    df[f'roll10_{feature}'] = (
                        df.groupby('batter')[feature]
                        .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                    )
        
        # 30-day time-based rolling (lagged)
        def _calc_30day_batter_rolling(group):
            group_indexed = group.set_index('date')
            
            for feature in all_features:
                if feature in group_indexed.columns:
                    if feature in ['pa', 'hr']:
                        rolled = group_indexed[feature].rolling('30D').sum().shift(1)
                    else:
                        rolled = group_indexed[feature].rolling('30D').mean().shift(1)
                    
                    # Use different naming for time-based vs game-based
                    if feature in ['launch_speed', 'launch_angle'] + available_bt:
                        group_indexed[f'roll30d_{feature}_mean'] = rolled
                    else:
                        group_indexed[f'roll30d_{feature}'] = rolled
            
            return group_indexed.reset_index()
        
        df = df.groupby('batter', group_keys=False).apply(_calc_30day_batter_rolling)
        
        # Calculate rates and indicators
        eps = 1e-6
        df['roll10_hr_rate'] = df['roll10_hr'] / (df['roll10_pa'] + eps)
        df['roll30d_hr_rate'] = df['roll30d_hr'] / (df['roll30d_pa'] + eps)
        
        # Performance indicators
        if 'roll10_launch_speed' in df.columns:
            df['prior_high_ev'] = (df['roll10_launch_speed'] > 90).astype('Int8')
        else:
            df['prior_high_ev'] = 0
        
        if 'roll10_launch_angle' in df.columns:
            df['prior_opt_la'] = (
                (df['roll10_launch_angle'] > 15) & (df['roll10_launch_angle'] < 30)
            ).astype('Int8')
        else:
            df['prior_opt_la'] = 0
        
        return df
    
    def _add_pitcher_features(self, statcast_df: pd.DataFrame, 
                            batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Add pitcher-related features."""
        logger.info("Adding pitcher features...")
        
        # Ensure date columns are properly formatted
        statcast_df = self.validator.validate_date_column(statcast_df, 'date')
        batter_games_df = self.validator.validate_date_column(batter_games_df, 'date')
        
        # Infer starting pitchers
        starters = self.pitcher_calculator.infer_game_starters(statcast_df)
        
        # Build pitcher profiles
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
        
        # Clean up merge artifacts and handle date conflicts
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
        
        # Calculate pitcher rates with proper error handling
        eps = 1e-6
        try:
            # Simple rate calculations using available data
            if 'p_roll10_hr' in batter_games.columns:
                # Use a simple proxy for pitcher rate calculation
                batter_games['p_roll10_hr_rate'] = batter_games['p_roll10_hr'] / 10.0  # Simplified rate
                
            if 'p_roll30d_hr' in batter_games.columns:
                batter_games['p_roll30d_hr_rate'] = batter_games['p_roll30d_hr'] / 30.0  # Simplified rate
                
        except Exception as e:
            logger.warning(f"Could not calculate pitcher rates: {e}")
            # Add placeholder columns
            batter_games['p_roll10_hr_rate'] = np.nan
            batter_games['p_roll30d_hr_rate'] = np.nan
        
        logger.info("Pitcher features added successfully")
        return batter_games
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather features if weather data is available."""
        if not self.weather_csv_path or not Path(self.weather_csv_path).exists():
            logger.info("No weather data available, adding placeholder columns")
            for col in ['temp_f', 'wind_mph', 'wind_out_to_cf', 'air_density_idx']:
                df[col] = np.nan
            return df
        
        try:
            logger.info(f"Loading weather data from {self.weather_csv_path}")
            weather_df = pd.read_csv(self.weather_csv_path)
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            
            # Merge on date and stadium
            merged = df.merge(weather_df, on=['date', 'stadium'], how='left')
            logger.info("Weather features added successfully")
            return merged
            
        except Exception as e:
            logger.warning(f"Failed to load weather data: {e}, using placeholders")
            for col in ['temp_f', 'wind_mph', 'wind_out_to_cf', 'air_density_idx']:
                df[col] = np.nan
            return df
    
    def _finalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize dataset with cleaning and validation."""
        logger.info("Finalizing dataset...")
        
        # Add player names if not present
        if 'batter_name' not in df.columns:
            df = self.name_resolver.add_player_names(df, id_col='batter', prefix='batter_')
        
        # Define final feature set
        core_features = [
            'date', 'season', 'game_pk', 'batter', 'batter_name', 'bat_team', 
            'home_team', 'away_team', 'stadium', 'park_factor', 'opp_team', 'opp_starter',
            'home_runs', 'hit_hr'
        ]
        
        # Rolling features
        rolling_features = [
            'roll10_pa', 'roll10_hr', 'roll10_launch_speed', 'roll10_launch_angle',
            'roll30d_pa', 'roll30d_hr', 'roll30d_launch_speed_mean', 'roll30d_launch_angle_mean',
            'roll10_hr_rate', 'roll30d_hr_rate', 'prior_high_ev', 'prior_opt_la'
        ]
        
        # Bat tracking features (if available)
        bt_features = []
        for bt in ['bat_speed', 'attack_angle', 'attack_direction', 'swing_path_tilt']:
            for window in ['roll10', 'roll30d']:
                bt_col = f'{window}_{bt}' if window == 'roll10' else f'{window}_{bt}_mean'
                if bt_col in df.columns:
                    bt_features.append(bt_col)
        
        # Handedness features
        handedness_features = []
        for hand in ['R', 'L']:
            for metric in ['pa10', 'hr10', 'ev10', 'la10', 'pa30d', 'hr30d', 'hr_rate10', 'hr_rate30d']:
                col = f'{metric}_{hand}'
                if col in df.columns:
                    handedness_features.append(col)
        
        # Pitcher features
        pitcher_features = [
            'p_roll10_hr', 'p_roll30d_hr', 'p_roll10_ev_allowed', 'p_roll30d_ev_allowed',
            'p_roll10_vel', 'p_roll10_hr_rate', 'p_roll30d_hr_rate'
        ]
        
        # Pitch mix features
        mix_features = [col for col in df.columns if col.startswith('mix_')]
        
        # Matchup features
        matchup_features = ['exp_slg_vs_mix']
        
        # Enhanced matchup features (batter vs pitcher history)
        enhanced_matchup_features = [
            'matchup_pa_career', 'matchup_hr_career', 'matchup_hr_rate_career',
            'matchup_avg_ev_career', 'matchup_avg_la_career',
            'matchup_pa_recent', 'matchup_hr_recent', 'matchup_hr_rate_recent',
            'matchup_days_since_last', 'matchup_encounters_last_year', 'matchup_familiarity_score',
            'vs_similar_hand_pa', 'vs_similar_hand_hr', 'vs_similar_hand_hr_rate',
            'vs_similar_velocity_pa', 'vs_similar_velocity_hr', 'vs_similar_velocity_hr_rate'
        ]
        
        # Situational context features
        situational_features = [
            'avg_runners_on_base', 'risp_percentage', 'bases_empty_percentage',
            'avg_score_differential', 'trailing_percentage', 'leading_percentage', 'tied_percentage',
            'clutch_pa_percentage', 'clutch_hr_rate', 'late_inning_pa_percentage', 'late_inning_hr_rate',
            'close_game_pa_percentage', 'close_game_hr_rate', 'pressure_performance_index',
            'first_inning_hr_rate', 'middle_innings_hr_rate', 'late_innings_hr_rate', 'extra_innings_hr_rate',
            'first_pa_hr_rate', 'leadoff_inning_hr_rate',
            'high_leverage_pa_pct', 'high_leverage_hr_rate', 'medium_leverage_pa_pct', 'medium_leverage_hr_rate',
            'low_leverage_pa_pct', 'low_leverage_hr_rate', 'leverage_performance_ratio',
            'hitters_count_hr_rate', 'pitchers_count_hr_rate', 'even_count_hr_rate', 'two_strike_hr_rate',
            'ahead_in_count_pct', 'behind_in_count_pct'
        ]
        
        # Enhanced weather features
        weather_features = [
            'temperature', 'wind_speed', 'wind_direction', 'humidity', 'pressure',
            'temp_hr_factor', 'wind_hr_factor', 'humidity_hr_factor', 'pressure_hr_factor',
            'air_density', 'air_density_ratio', 'flight_distance_factor', 'drag_factor',
            'effective_wind_speed', 'wind_assistance_factor', 'stadium_wind_factor',
            'weather_favorability_index', 'atmospheric_carry_index', 'elevation_factor',
            'ballpark_weather_factor'
        ]
        
        # Recent form features with time decay
        recent_form_features = [
            'power_form_hr_rate', 'power_form_avg_ev', 'power_form_hard_hit_rate',
            'power_form_barrel_rate', 'power_form_iso_power',
            'contact_form_avg_la', 'contact_form_sweet_spot_rate',
            'contact_form_line_drive_rate', 'contact_form_consistency',
            'discipline_form_contact_rate', 'discipline_form_z_contact_rate',
            'discipline_form_chase_rate', 'discipline_form_whiff_rate',
            'hot_streak_indicator', 'cold_streak_indicator', 
            'recent_power_surge', 'recent_slump_indicator', 'momentum_score',
            'hr_rate_trend_7d', 'hr_rate_trend_14d', 'hr_rate_trend_30d',
            'ev_trend_7d', 'ev_trend_14d', 'form_acceleration'
        ]
        
        # Advanced streak and momentum features
        streak_momentum_features = [
            'current_hot_streak', 'max_hot_streak_21d', 'hot_streak_intensity',
            'days_since_hot_streak', 'hot_streak_frequency',
            'current_cold_streak', 'max_cold_streak_21d', 'cold_streak_depth',
            'days_since_cold_streak', 'recovery_momentum', 'slump_risk_indicator',
            'power_momentum_7d', 'consistency_momentum', 'trend_acceleration',
            'momentum_direction', 'momentum_strength', 'momentum_sustainability',
            'hr_rate_velocity', 'performance_acceleration', 'velocity_consistency',
            'breakout_velocity', 'hot_cold_cycle_position', 'pattern_stability', 
            'rhythm_indicator', 'cycle_prediction', 'confidence_indicator', 
            'pressure_response', 'clutch_momentum', 'mental_toughness'
        ]
        
        # Advanced ballpark features
        ballpark_features = [
            'park_left_field_distance', 'park_center_field_distance', 'park_right_field_distance',
            'park_left_field_height', 'park_center_field_height', 'park_right_field_height',
            'park_foul_territory_size', 'park_elevation', 'park_is_dome', 'park_surface_turf',
            'park_hr_difficulty_index', 'park_symmetry_factor', 'park_wall_height_factor',
            'park_foul_territory_hr_boost', 'park_elevation_carry_boost', 'park_air_density_factor',
            'park_dome_carry_reduction', 'park_coastal_humidity_factor', 'park_pull_factor_left',
            'park_pull_factor_right', 'park_opposite_field_factor', 'park_center_field_factor',
            'batter_park_hr_rate_boost', 'batter_park_historical_performance', 'batter_park_comfort_factor',
            'park_wind_interaction', 'park_temperature_interaction', 'park_humidity_interaction',
            'park_weather_hr_multiplier', 'park_day_night_factor', 'park_season_factor',
            'park_month_hr_factor', 'park_offense_context', 'park_pitcher_context', 'park_defensive_context'
        ]
        
        # Temporal and fatigue features
        temporal_fatigue_features = [
            'game_hour', 'circadian_performance_factor', 'optimal_time_window',
            'suboptimal_time_penalty', 'night_game_indicator', 'afternoon_game_boost',
            'evening_game_factor', 'games_without_rest', 'cumulative_fatigue', 'fatigue_level',
            'rest_deficit', 'energy_reserves', 'recovery_status', 'timezone_change',
            'jet_lag_factor', 'travel_fatigue', 'home_away_transition', 'cross_country_travel',
            'time_zone_adjustment', 'days_since_rest', 'rest_quality_score', 'consecutive_games',
            'weekly_game_density', 'rest_vs_schedule', 'recovery_time_available',
            'season_fatigue_factor', 'monthly_energy_level', 'season_progression',
            'playoff_chase_energy', 'spring_training_carryover', 'dog_days_effect',
            'games_last_7_days', 'games_last_14_days', 'games_next_7_days',
            'schedule_intensity', 'upcoming_workload', 'recent_workload',
            'optimal_performance_window', 'suboptimal_timing_penalty', 'circadian_mismatch',
            'time_preference_alignment'
        ]
        
        # Feature interaction terms
        interaction_features = [
            'power_form_altitude_boost', 'power_weather_synergy', 'hot_streak_confidence_boost',
            'momentum_toughness_factor', 'energy_circadian_factor', 'rested_momentum_boost',
            'matchup_form_synergy', 'park_wind_amplification', 'park_temp_amplification',
            'clutch_pressure_performance', 'hot_streak_power_boost', 'cold_streak_confidence_penalty',
            'fatigue_momentum_penalty', 'jet_lag_circadian_disruption', 'park_advantage_pull_boost',
            'composite_power_index', 'composite_momentum_index', 'environmental_favorability_index',
            'physical_condition_index', 'psychological_state_index', 'power_contact_ratio',
            'momentum_fatigue_ratio', 'rest_workload_ratio', 'performance_pressure_ratio',
            'hot_cold_balance', 'elite_power_indicator', 'high_momentum_indicator',
            'extreme_fatigue_indicator', 'optimal_conditions_indicator', 'elite_performance_convergence',
            'mind_body_synergy', 'form_environment_synergy', 'momentum_opportunity_synergy',
            'experience_pressure_synergy', 'rest_performance_synergy', 'overall_performance_multiplier',
            'clutch_performance_multiplier', 'hot_streak_performance_multiplier', 'fatigue_adjusted_multiplier'
        ]
        
        # Combine all feature sets
        all_features = (core_features + rolling_features + bt_features + 
                       handedness_features + pitcher_features + mix_features + 
                       matchup_features + enhanced_matchup_features + situational_features + 
                       weather_features + recent_form_features + streak_momentum_features + 
                       ballpark_features + temporal_fatigue_features + interaction_features)
        
        # Select available columns
        final_cols = [col for col in all_features if col in df.columns]
        final_df = df[final_cols].copy()
        
        # Clean numeric data
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns.tolist()
        final_df = self.validator.clean_numeric_data(final_df, numeric_cols)
        
        # Cap outlier rates
        rate_cols = [col for col in final_df.columns if 'hr_rate' in col]
        for col in rate_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].clip(0, 0.25)  # Cap at 25% HR rate
        
        # Validate final dataset
        validation_results = self._validate_final_dataset(final_df)
        if not validation_results['valid']:
            logger.warning(f"Dataset validation issues: {validation_results['issues']}")
        
        if validation_results['warnings']:
            logger.info(f"Dataset validation warnings: {validation_results['warnings']}")
        
        logger.info(f"Final dataset: {len(final_df)} rows, {len(final_df.columns)} columns")
        return final_df
    
    def _validate_final_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the final dataset."""
        validation = {'valid': True, 'issues': [], 'warnings': []}
        
        # Check for required columns
        required_cols = ['date', 'batter', 'home_runs', 'hit_hr']
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            validation['issues'].append(f"Missing required columns: {missing_required}")
            validation['valid'] = False
        
        # Check data coverage
        if len(df) == 0:
            validation['issues'].append("Dataset is empty")
            validation['valid'] = False
        
        # Check for reasonable home run rates
        if 'hit_hr' in df.columns:
            hr_rate = df['hit_hr'].mean()
            if hr_rate < 0.02 or hr_rate > 0.15:
                validation['warnings'].append(f"Unusual overall HR rate: {hr_rate:.3f}")
        
        # Check date range
        if 'date' in df.columns:
            date_range = df['date'].max() - df['date'].min()
            if date_range.days < 30:
                validation['warnings'].append(f"Short date range: {date_range.days} days")
        
        return validation

# Efficient prediction-focused builder
class EfficientPredictionBuilder(PregameDatasetBuilder):
    """Optimized builder for live predictions with minimal data requirements."""
    
    def build_prediction_dataset(self, target_date: str, 
                                max_lookback_days: int = 45) -> pd.DataFrame:
        """Build dataset optimized for predictions on a specific date."""
        target_dt = pd.to_datetime(target_date).normalize()
        start_dt = target_dt - timedelta(days=max_lookback_days)
        
        # Temporarily override date range
        original_start = self.start_date
        original_end = self.end_date
        
        self.start_date = start_dt.strftime("%Y-%m-%d")
        self.end_date = target_date
        
        try:
            # Build minimal dataset
            logger.info(f"Building prediction dataset for {target_date} "
                       f"(lookback: {max_lookback_days} days)")
            
            dataset = self.build_dataset(force_rebuild=True, cache_format="parquet")
            
            # Filter to target date
            target_rows = dataset[
                pd.to_datetime(dataset['date']).dt.normalize() == target_dt
            ].copy()
            
            logger.info(f"Prediction dataset ready: {len(target_rows)} rows for {target_date}")
            return target_rows
            
        finally:
            # Restore original date range
            self.start_date = original_start
            self.end_date = original_end

# Export main classes
__all__ = ['PregameDatasetBuilder', 'EfficientPredictionBuilder']