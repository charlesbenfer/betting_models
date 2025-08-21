"""
Baseball Home Run Prediction System
==================================

A production-ready system for predicting home runs and finding +EV betting opportunities.

Main entry point and configuration management.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseball_hr.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class Config:
    """Configuration management for the baseball HR prediction system."""
    
    def __init__(self):
        self.MODEL_DIR = Path("saved_models_pregame")
        self.DATA_DIR = Path("data")
        self.CACHE_DIR = self.DATA_DIR / "processed"
        
        # Create directories
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.CACHE_DIR.mkdir(exist_ok=True)
        
        # API Configuration
        self.THEODDS_API_KEY = os.getenv("THEODDS_API_KEY", "").strip()
        self.API_BASE_URL = "https://api.the-odds-api.com/v4"
        self.API_TIMEOUT = 30
        self.API_MAX_RETRIES = 3
        
        # Model Configuration
        self.RANDOM_STATE = 42
        self.MIN_EV_THRESHOLD = 0.05
        self.MIN_PROB_THRESHOLD = 0.06
        self.CACHE_FORMAT = "parquet"
        
        # Data Configuration
        self.DEFAULT_START_DATE = "2023-03-01"
        self.DEFAULT_END_DATE = "2024-10-01"
        self.MAX_LOOKBACK_DAYS = 45
        
        # Feature Configuration
        self.BAT_TRACKING_THRESHOLD = 0.1
        self.ROLLING_WINDOW_GAMES = 10
        self.ROLLING_WINDOW_DAYS = 30
        
        # Team to stadium mapping
        self.TEAM_TO_STADIUM = {
            'ARI': 'Chase Field', 'ATL': 'Truist Park', 'BAL': 'Oriole Park at Camden Yards',
            'BOS': 'Fenway Park', 'CHC': 'Wrigley Field', 'CIN': 'Great American Ball Park',
            'CLE': 'Progressive Field', 'COL': 'Coors Field', 'CWS': 'Guaranteed Rate Field',
            'DET': 'Comerica Park', 'HOU': 'Minute Maid Park', 'KC': 'Kauffman Stadium',
            'LAA': 'Angel Stadium', 'LAD': 'Dodger Stadium', 'MIA': 'loanDepot park',
            'MIL': 'American Family Field', 'MIN': 'Target Field', 'NYM': 'Citi Field',
            'NYY': 'Yankee Stadium', 'OAK': 'Oakland Coliseum', 'PHI': 'Citizens Bank Park',
            'PIT': 'PNC Park', 'SD': 'Petco Park', 'SEA': 'T-Mobile Park',
            'SF': 'Oracle Park', 'STL': 'Busch Stadium', 'TB': 'Tropicana Field',
            'TEX': 'Globe Life Field', 'TOR': 'Rogers Centre', 'WSH': 'Nationals Park',
        }
        
        # Park factors (conservative estimates)
        self.PARK_FACTORS = {
            'Coors Field': 1.25, 'Great American Ball Park': 1.15, 'Yankee Stadium': 1.12,
            'Globe Life Field': 1.10, 'Citizens Bank Park': 1.08, 'Fenway Park': 1.06,
            'Oriole Park at Camden Yards': 1.05, 'American Family Field': 1.04,
            'Chase Field': 0.98, 'Dodger Stadium': 0.96, 'Oracle Park': 0.92, 'Petco Park': 0.90
        }
        
        # Core features (available in all time periods)
        self.CORE_FEATURES = [
            'roll10_pa', 'roll10_hr', 'roll10_ev', 'roll10_la',
            'roll30d_pa', 'roll30d_hr', 'roll30d_ev_mean', 'roll30d_la_mean',
            'roll10_hr_rate', 'roll30d_hr_rate', 'prior_high_ev', 'prior_opt_la',
            'park_factor', 'p_roll10_hr', 'p_roll30d_hr', 'p_roll10_ev_allowed',
            'p_roll30d_ev_allowed', 'p_roll10_vel', 'exp_slg_vs_mix',
            'pa10_R', 'hr10_R', 'ev10_R', 'la10_R', 'pa10_L', 'hr10_L', 'ev10_L', 'la10_L',
            'pa30d_R', 'hr30d_R', 'pa30d_L', 'hr30d_L', 'hr_rate10_R', 'hr_rate10_L',
            'hr_rate30d_R', 'hr_rate30d_L', 'p_roll10_hr_rate', 'p_roll30d_hr_rate'
        ]
        
        # Enhanced matchup features (new in this version)
        self.MATCHUP_FEATURES = [
            'matchup_pa_career', 'matchup_hr_career', 'matchup_hr_rate_career',
            'matchup_avg_ev_career', 'matchup_avg_la_career',
            'matchup_pa_recent', 'matchup_hr_recent', 'matchup_hr_rate_recent',
            'matchup_days_since_last', 'matchup_encounters_last_year', 'matchup_familiarity_score',
            'vs_similar_hand_pa', 'vs_similar_hand_hr', 'vs_similar_hand_hr_rate',
            'vs_similar_velocity_pa', 'vs_similar_velocity_hr', 'vs_similar_velocity_hr_rate'
        ]
        
        # Situational context features (Step 2)
        self.SITUATIONAL_FEATURES = [
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
        
        # Weather impact features (Step 3)
        self.WEATHER_FEATURES = [
            'temperature', 'wind_speed', 'wind_direction', 'humidity', 'pressure',
            'temp_hr_factor', 'wind_hr_factor', 'humidity_hr_factor', 'pressure_hr_factor',
            'air_density', 'air_density_ratio', 'flight_distance_factor', 'drag_factor',
            'effective_wind_speed', 'wind_assistance_factor', 'stadium_wind_factor',
            'weather_favorability_index', 'atmospheric_carry_index', 'elevation_factor',
            'ballpark_weather_factor'
        ]
        
        # Recent form features with time decay (Step 4)
        self.RECENT_FORM_FEATURES = [
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
        
        # Advanced streak and momentum features (Step 5)
        self.STREAK_MOMENTUM_FEATURES = [
            # Hot streak features
            'current_hot_streak', 'max_hot_streak_21d', 'hot_streak_intensity',
            'days_since_hot_streak', 'hot_streak_frequency',
            
            # Cold streak features  
            'current_cold_streak', 'max_cold_streak_21d', 'cold_streak_depth',
            'days_since_cold_streak', 'recovery_momentum', 'slump_risk_indicator',
            
            # Momentum features
            'power_momentum_7d', 'consistency_momentum', 'trend_acceleration',
            'momentum_direction', 'momentum_strength', 'momentum_sustainability',
            
            # Velocity features
            'hr_rate_velocity', 'performance_acceleration', 'velocity_consistency',
            'breakout_velocity',
            
            # Pattern recognition features
            'hot_cold_cycle_position', 'pattern_stability', 'rhythm_indicator',
            'cycle_prediction',
            
            # Psychological features
            'confidence_indicator', 'pressure_response', 'clutch_momentum',
            'mental_toughness'
        ]
        
        # Advanced ballpark-specific features (Step 6)
        self.BALLPARK_FEATURES = [
            # Dimensional features
            'park_left_field_distance', 'park_center_field_distance', 'park_right_field_distance',
            'park_left_field_height', 'park_center_field_height', 'park_right_field_height',
            'park_foul_territory_size', 'park_elevation', 'park_is_dome', 'park_surface_turf',
            'park_hr_difficulty_index', 'park_symmetry_factor', 'park_wall_height_factor',
            'park_foul_territory_hr_boost',
            
            # Carry distance features
            'park_elevation_carry_boost', 'park_air_density_factor', 'park_dome_carry_reduction',
            'park_coastal_humidity_factor',
            
            # Directional features
            'park_pull_factor_left', 'park_pull_factor_right', 'park_opposite_field_factor',
            'park_center_field_factor',
            
            # Batter-park interaction
            'batter_park_hr_rate_boost', 'batter_park_historical_performance',
            'batter_park_comfort_factor',
            
            # Weather-park interaction
            'park_wind_interaction', 'park_temperature_interaction', 'park_humidity_interaction',
            'park_weather_hr_multiplier',
            
            # Temporal effects
            'park_day_night_factor', 'park_season_factor', 'park_month_hr_factor',
            
            # Context features
            'park_offense_context', 'park_pitcher_context', 'park_defensive_context'
        ]
        
        # Temporal and fatigue features (Step 7)
        self.TEMPORAL_FATIGUE_FEATURES = [
            # Circadian rhythm features
            'game_hour', 'circadian_performance_factor', 'optimal_time_window',
            'suboptimal_time_penalty', 'night_game_indicator', 'afternoon_game_boost',
            'evening_game_factor',
            
            # Fatigue accumulation features
            'games_without_rest', 'cumulative_fatigue', 'fatigue_level', 'rest_deficit',
            'energy_reserves', 'recovery_status',
            
            # Travel and jet lag features
            'timezone_change', 'jet_lag_factor', 'travel_fatigue', 'home_away_transition',
            'cross_country_travel', 'time_zone_adjustment',
            
            # Rest pattern features
            'days_since_rest', 'rest_quality_score', 'consecutive_games', 'weekly_game_density',
            'rest_vs_schedule', 'recovery_time_available',
            
            # Seasonal energy features
            'season_fatigue_factor', 'monthly_energy_level', 'season_progression',
            'playoff_chase_energy', 'spring_training_carryover', 'dog_days_effect',
            
            # Schedule density features
            'games_last_7_days', 'games_last_14_days', 'games_next_7_days',
            'schedule_intensity', 'upcoming_workload', 'recent_workload',
            
            # Performance timing features
            'optimal_performance_window', 'suboptimal_timing_penalty', 'circadian_mismatch',
            'time_preference_alignment'
        ]
        
        # Feature interaction terms (Step 8)
        self.INTERACTION_FEATURES = [
            # Multiplicative interactions
            'power_form_altitude_boost', 'power_weather_synergy', 'hot_streak_confidence_boost',
            'momentum_toughness_factor', 'energy_circadian_factor', 'rested_momentum_boost',
            'matchup_form_synergy', 'park_wind_amplification', 'park_temp_amplification',
            'clutch_pressure_performance',
            
            # Conditional interactions
            'hot_streak_power_boost', 'cold_streak_confidence_penalty', 'fatigue_momentum_penalty',
            'jet_lag_circadian_disruption', 'park_advantage_pull_boost',
            
            # Composite indices
            'composite_power_index', 'composite_momentum_index', 'environmental_favorability_index',
            'physical_condition_index', 'psychological_state_index',
            
            # Ratio interactions
            'power_contact_ratio', 'momentum_fatigue_ratio', 'rest_workload_ratio',
            'performance_pressure_ratio', 'hot_cold_balance',
            
            # Threshold interactions
            'elite_power_indicator', 'high_momentum_indicator', 'extreme_fatigue_indicator',
            'optimal_conditions_indicator', 'elite_performance_convergence',
            
            # Cross-domain synergies
            'mind_body_synergy', 'form_environment_synergy', 'momentum_opportunity_synergy',
            'experience_pressure_synergy', 'rest_performance_synergy',
            
            # Performance multipliers
            'overall_performance_multiplier', 'clutch_performance_multiplier',
            'hot_streak_performance_multiplier', 'fatigue_adjusted_multiplier'
        ]
        
        # Bat tracking features (available from 2024+)
        self.BAT_TRACKING_FEATURES = [
            'roll10_bs', 'roll10_aa', 'roll10_ad', 'roll10_spt',
            'roll30d_bs_mean', 'roll30d_aa_mean', 'roll30d_ad_mean', 'roll30d_spt_mean'
        ]
    
    def validate_api_key(self) -> bool:
        """Validate that API key is available."""
        if not self.THEODDS_API_KEY:
            logger.warning("No THEODDS_API_KEY found in environment variables")
            return False
        return True
    
    def get_file_paths(self) -> Dict[str, Path]:
        """Get all important file paths."""
        return {
            "core_model": self.MODEL_DIR / "core_model.joblib",
            "enhanced_model": self.MODEL_DIR / "enhanced_model.joblib",
            "metadata": self.MODEL_DIR / "dual_model_metadata.json",
            "cache": self.CACHE_DIR / f"pregame_v2_{self.DEFAULT_START_DATE}_{self.DEFAULT_END_DATE}.parquet"
        }

# Global configuration instance
config = Config()

if __name__ == "__main__":
    logger.info("Baseball HR Prediction System initialized")
    logger.info(f"Model directory: {config.MODEL_DIR}")
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"API key available: {config.validate_api_key()}")