"""
Temporal and Fatigue Features (Step 7)
=====================================

Advanced time-based and fatigue modeling including:
- Circadian rhythm analysis
- Travel fatigue and jet lag effects
- Rest vs fatigue patterns
- Time zone impacts
- Game time preferences
- Seasonal energy cycles
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, time
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class CircadianParameters:
    """Parameters for circadian rhythm modeling."""
    peak_performance_hour: int = 14  # 2 PM typical peak
    trough_performance_hour: int = 6  # 6 AM typical trough
    amplitude: float = 0.1  # 10% swing from peak to trough
    individual_variation: float = 0.03  # 3% individual variation

@dataclass
class FatigueParameters:
    """Parameters for fatigue modeling."""
    max_games_without_rest: int = 10  # Maximum sustainable games
    fatigue_accumulation_rate: float = 0.02  # 2% per game without rest
    recovery_rate_per_day: float = 0.05  # 5% recovery per rest day
    travel_fatigue_threshold: int = 2  # Time zones for significant fatigue
    jet_lag_recovery_days: int = 1  # Days to recover per time zone

class TemporalFatigueCalculator:
    """Calculate temporal and fatigue-based features."""
    
    def __init__(self, circadian_params: CircadianParameters = None, 
                 fatigue_params: FatigueParameters = None):
        self.circadian_params = circadian_params or CircadianParameters()
        self.fatigue_params = fatigue_params or FatigueParameters()
        self.timezone_map = self._initialize_timezone_data()
        
    def calculate_temporal_fatigue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive temporal and fatigue features.
        
        Args:
            df: DataFrame with game data including dates and times
            
        Returns:
            DataFrame with added temporal/fatigue features
        """
        logger.info("Calculating temporal and fatigue features...")
        
        # Ensure required columns
        required_cols = ['date', 'batter']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return df
            
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        # Sort by batter and date
        df_copy = df_copy.sort_values(['batter', 'date']).reset_index(drop=True)
        
        # Calculate different categories of temporal/fatigue features
        df_copy = self._calculate_circadian_features(df_copy)
        df_copy = self._calculate_fatigue_features(df_copy)
        df_copy = self._calculate_travel_features(df_copy)
        df_copy = self._calculate_rest_patterns(df_copy)
        df_copy = self._calculate_seasonal_energy(df_copy)
        df_copy = self._calculate_schedule_density(df_copy)
        df_copy = self._calculate_performance_timing(df_copy)
        
        logger.info("Temporal and fatigue features calculation completed")
        return df_copy
    
    def _initialize_timezone_data(self) -> Dict[str, int]:
        """Initialize timezone data for MLB stadiums."""
        return {
            # Eastern Time (UTC-5)
            'Fenway Park': -5, 'Yankee Stadium': -5, 'Oriole Park at Camden Yards': -5,
            'Tropicana Field': -5, 'Nationals Park': -5, 'Citizens Bank Park': -5,
            'Citi Field': -5, 'Truist Park': -5, 'loanDepot park': -5,
            'PNC Park': -5, 'Great American Ball Park': -5, 'Comerica Park': -5,
            'Progressive Field': -5, 'Rogers Centre': -5,
            
            # Central Time (UTC-6)  
            'American Family Field': -6, 'Guaranteed Rate Field': -6, 'Target Field': -6,
            'Kauffman Stadium': -6, 'Globe Life Field': -6, 'Minute Maid Park': -6,
            'Busch Stadium': -6, 'Wrigley Field': -6,
            
            # Mountain Time (UTC-7)
            'Coors Field': -7, 'Chase Field': -7,
            
            # Pacific Time (UTC-8)
            'T-Mobile Park': -8, 'Oakland Coliseum': -8, 'Angel Stadium': -8,
            'Dodger Stadium': -8, 'Petco Park': -8, 'Oracle Park': -8
        }
    
    def _calculate_circadian_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate circadian rhythm and time-of-day features."""
        
        # Initialize features
        df['game_hour'] = 0
        df['circadian_performance_factor'] = 1.0
        df['optimal_time_window'] = 0
        df['suboptimal_time_penalty'] = 0.0
        df['night_game_indicator'] = 0
        df['afternoon_game_boost'] = 0.0
        df['evening_game_factor'] = 0.0
        
        # Extract game time information
        # Note: In production, you'd have actual game times. Using date proxy here.
        df['game_hour'] = 14  # Default to 2 PM (typical day game)
        
        # If we have actual game time data, use it
        if 'game_time' in df.columns:
            df['game_hour'] = pd.to_datetime(df['game_time']).dt.hour
        elif 'start_time' in df.columns:
            df['game_hour'] = pd.to_datetime(df['start_time']).dt.hour
        else:
            # Estimate based on day of week and month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            
            # Weekend games often earlier, weekday games often later
            weekend_mask = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
            df.loc[weekend_mask, 'game_hour'] = 13  # 1 PM weekends
            df.loc[~weekend_mask, 'game_hour'] = 19  # 7 PM weekdays
            
            # Summer games often later to avoid heat
            summer_mask = df['month'].isin([6, 7, 8])
            df.loc[summer_mask & ~weekend_mask, 'game_hour'] = 20  # 8 PM summer weekdays
        
        # Calculate circadian performance factor
        for i in range(len(df)):
            hour = df.iloc[i]['game_hour']
            circadian_factor = self._calculate_circadian_factor(hour)
            df.iloc[i, df.columns.get_loc('circadian_performance_factor')] = circadian_factor
        
        # Optimal time window (1-4 PM typically best)
        df['optimal_time_window'] = ((df['game_hour'] >= 13) & (df['game_hour'] <= 16)).astype(int)
        
        # Suboptimal time penalty
        early_morning_penalty = np.where(df['game_hour'] < 10, 0.05, 0.0)
        late_night_penalty = np.where(df['game_hour'] > 22, 0.03, 0.0)
        df['suboptimal_time_penalty'] = early_morning_penalty + late_night_penalty
        
        # Night game indicator
        df['night_game_indicator'] = (df['game_hour'] >= 18).astype(int)
        
        # Afternoon game boost (traditional baseball time)
        afternoon_hours = (df['game_hour'] >= 12) & (df['game_hour'] <= 16)
        df['afternoon_game_boost'] = np.where(afternoon_hours, 0.02, 0.0)
        
        # Evening game factor
        evening_hours = (df['game_hour'] >= 17) & (df['game_hour'] <= 21)
        df['evening_game_factor'] = np.where(evening_hours, 0.01, 0.0)
        
        return df
    
    def _calculate_fatigue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fatigue accumulation features."""
        
        def calculate_batter_fatigue(batter_data):
            """Calculate fatigue features for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            # Initialize features
            data['games_without_rest'] = 0
            data['cumulative_fatigue'] = 0.0
            data['fatigue_level'] = 0.0  # 0-1 scale
            data['rest_deficit'] = 0.0
            data['energy_reserves'] = 1.0  # 1 = full energy, 0 = exhausted
            data['recovery_status'] = 1.0  # 1 = fully recovered
            
            current_streak = 0
            cumulative_fatigue = 0.0
            
            for i in range(len(data)):
                current_date = data.iloc[i]['date']
                
                if i > 0:
                    prev_date = data.iloc[i-1]['date']
                    days_gap = (current_date - prev_date).days
                    
                    if days_gap <= 1:
                        # Consecutive or same day (doubleheader)
                        current_streak += 1
                        cumulative_fatigue += self.fatigue_params.fatigue_accumulation_rate
                        
                        # Extra fatigue for doubleheaders
                        if days_gap == 0:
                            cumulative_fatigue += 0.01
                    else:
                        # Rest day(s) - recovery
                        rest_days = days_gap - 1
                        recovery = rest_days * self.fatigue_params.recovery_rate_per_day
                        cumulative_fatigue = max(0, cumulative_fatigue - recovery)
                        current_streak = 0 if rest_days > 0 else current_streak
                
                # Set values
                data.iloc[i, data.columns.get_loc('games_without_rest')] = current_streak
                data.iloc[i, data.columns.get_loc('cumulative_fatigue')] = cumulative_fatigue
                
                # Fatigue level (0-1)
                fatigue_level = min(1.0, cumulative_fatigue / 0.2)  # Cap at 20% cumulative
                data.iloc[i, data.columns.get_loc('fatigue_level')] = fatigue_level
                
                # Rest deficit (how much rest is needed)
                optimal_rest_ratio = 0.15  # 15% rest days optimal
                games_last_week = min(i + 1, 7)
                recent_games = data.iloc[max(0, i-6):i+1]
                if len(recent_games) > 1:
                    date_range = (recent_games['date'].max() - recent_games['date'].min()).days + 1
                    actual_rest_ratio = max(0, (date_range - len(recent_games)) / date_range)
                    rest_deficit = max(0, optimal_rest_ratio - actual_rest_ratio)
                    data.iloc[i, data.columns.get_loc('rest_deficit')] = rest_deficit
                
                # Energy reserves
                energy = max(0.3, 1.0 - fatigue_level)  # Never below 30%
                data.iloc[i, data.columns.get_loc('energy_reserves')] = energy
                
                # Recovery status
                recovery = max(0.5, 1.0 - cumulative_fatigue)  # Never below 50%
                data.iloc[i, data.columns.get_loc('recovery_status')] = recovery
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_fatigue)
    
    def _calculate_travel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate travel and jet lag features."""
        
        # Initialize features
        df['timezone_change'] = 0
        df['jet_lag_factor'] = 0.0
        df['travel_fatigue'] = 0.0
        df['home_away_transition'] = 0
        df['cross_country_travel'] = 0
        df['time_zone_adjustment'] = 0.0
        
        if 'stadium' not in df.columns:
            logger.warning("Stadium data not available for travel analysis")
            return df
        
        def calculate_batter_travel(batter_data):
            """Calculate travel features for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            for i in range(1, len(data)):
                current_stadium = data.iloc[i]['stadium']
                prev_stadium = data.iloc[i-1]['stadium']
                current_date = data.iloc[i]['date']
                prev_date = data.iloc[i-1]['date']
                
                days_gap = (current_date - prev_date).days
                
                if current_stadium != prev_stadium and days_gap <= 3:
                    # Stadium change within 3 days = travel
                    current_tz = self.timezone_map.get(current_stadium, -5)
                    prev_tz = self.timezone_map.get(prev_stadium, -5)
                    tz_change = abs(current_tz - prev_tz)
                    
                    data.iloc[i, data.columns.get_loc('timezone_change')] = tz_change
                    
                    # Jet lag factor (more severe for eastward travel)
                    if tz_change >= self.fatigue_params.travel_fatigue_threshold:
                        # Eastward travel is harder (negative timezone change)
                        eastward_penalty = 1.5 if (current_tz - prev_tz) > 0 else 1.0
                        jet_lag = (tz_change / 3) * 0.05 * eastward_penalty  # Up to 5% penalty
                        data.iloc[i, data.columns.get_loc('jet_lag_factor')] = jet_lag
                        
                        # Travel fatigue
                        travel_fatigue = min(0.1, tz_change * 0.02)  # 2% per timezone, max 10%
                        data.iloc[i, data.columns.get_loc('travel_fatigue')] = travel_fatigue
                        
                        # Cross country travel
                        if tz_change >= 3:
                            data.iloc[i, data.columns.get_loc('cross_country_travel')] = 1
                    
                    # Home/away transition indicator
                    if 'home_team' in data.columns and 'bat_team' in data.columns:
                        current_home = data.iloc[i]['home_team'] == data.iloc[i]['bat_team']
                        prev_home = data.iloc[i-1]['home_team'] == data.iloc[i-1]['bat_team']
                        if current_home != prev_home:
                            data.iloc[i, data.columns.get_loc('home_away_transition')] = 1
                
                # Time zone adjustment (recovery from jet lag)
                if i > 0:
                    prev_jet_lag = data.iloc[i-1]['jet_lag_factor']
                    if prev_jet_lag > 0 and days_gap >= 1:
                        # Recovery at 50% per day
                        recovery_rate = 0.5 * min(days_gap, 3)  # Full recovery in 3 days
                        adjustment = prev_jet_lag * (1 - recovery_rate)
                        data.iloc[i, data.columns.get_loc('time_zone_adjustment')] = adjustment
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_travel)
    
    def _calculate_rest_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rest and recovery patterns."""
        
        def calculate_batter_rest(batter_data):
            """Calculate rest patterns for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            # Initialize features
            data['days_since_rest'] = 0
            data['rest_quality_score'] = 1.0
            data['consecutive_games'] = 1
            data['weekly_game_density'] = 0.0
            data['rest_vs_schedule'] = 0.0
            data['recovery_time_available'] = 0.0
            
            for i in range(len(data)):
                current_date = data.iloc[i]['date']
                
                # Days since last rest day
                days_since_rest = 0
                for j in range(i-1, -1, -1):
                    prev_date = data.iloc[j]['date']
                    gap = (current_date - prev_date).days
                    if gap > 1:  # Rest day found
                        days_since_rest = (current_date - prev_date).days - 1
                        break
                    days_since_rest += 1
                
                data.iloc[i, data.columns.get_loc('days_since_rest')] = days_since_rest
                
                # Rest quality score (decreases with consecutive games)
                rest_quality = max(0.5, 1.0 - (days_since_rest * 0.05))
                data.iloc[i, data.columns.get_loc('rest_quality_score')] = rest_quality
                
                # Consecutive games
                consecutive = 1
                for j in range(i-1, -1, -1):
                    prev_date = data.iloc[j]['date']
                    if (current_date - prev_date).days <= consecutive:
                        consecutive += 1
                        current_date = prev_date
                    else:
                        break
                data.iloc[i, data.columns.get_loc('consecutive_games')] = consecutive
                
                # Weekly game density (games in last 7 days)
                week_start = current_date - timedelta(days=7)
                recent_games = data[(data['date'] >= week_start) & (data['date'] <= current_date)]
                weekly_density = len(recent_games) / 7.0
                data.iloc[i, data.columns.get_loc('weekly_game_density')] = weekly_density
                
                # Rest vs schedule (how well rested vs typical)
                optimal_density = 4.5 / 7.0  # ~4.5 games per week optimal
                rest_vs_schedule = optimal_density - weekly_density
                data.iloc[i, data.columns.get_loc('rest_vs_schedule')] = rest_vs_schedule
                
                # Recovery time available (rest before next game)
                recovery_time = 1.0  # Default 1 day
                if i < len(data) - 1:
                    next_date = data.iloc[i+1]['date']
                    recovery_time = (next_date - current_date).days
                data.iloc[i, data.columns.get_loc('recovery_time_available')] = min(3.0, recovery_time)
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_rest)
    
    def _calculate_seasonal_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate seasonal energy and motivation cycles."""
        
        # Initialize features
        df['season_fatigue_factor'] = 0.0
        df['monthly_energy_level'] = 1.0
        df['season_progression'] = 0.0
        df['playoff_chase_energy'] = 0.0
        df['spring_training_carryover'] = 0.0
        df['dog_days_effect'] = 0.0
        
        # Season progression (0 = start, 1 = end)
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Baseball season roughly April (day 90) to October (day 300)
        season_start = 90  # April 1st
        season_end = 300   # October 27th
        season_length = season_end - season_start
        
        df['season_progression'] = np.clip(
            (df['day_of_year'] - season_start) / season_length, 0, 1
        )
        
        # Season fatigue (builds up over the year)
        df['season_fatigue_factor'] = df['season_progression'] * 0.15  # Up to 15% fatigue
        
        # Monthly energy patterns
        month_energy_map = {
            3: 1.05,   # March - Spring training energy
            4: 1.02,   # April - Season opener boost
            5: 1.0,    # May - Baseline
            6: 0.98,   # June - First fatigue signs
            7: 0.95,   # July - Mid-season fatigue
            8: 0.93,   # August - Dog days of summer
            9: 1.01,   # September - Playoff push
            10: 1.03,  # October - Playoff intensity
            11: 0.9    # November - Season end exhaustion
        }
        
        df['monthly_energy_level'] = df['date'].dt.month.map(month_energy_map).fillna(1.0)
        
        # Dog days effect (late July through August)
        dog_days_mask = (df['date'].dt.month == 7) & (df['date'].dt.day > 20) | (df['date'].dt.month == 8)
        df['dog_days_effect'] = np.where(dog_days_mask, -0.02, 0.0)
        
        # Spring training carryover (early season boost)
        early_season_mask = (df['date'].dt.month.isin([3, 4])) | ((df['date'].dt.month == 5) & (df['date'].dt.day <= 15))
        df['spring_training_carryover'] = np.where(early_season_mask, 0.01, 0.0)
        
        # Playoff chase energy (September boost for competitive teams)
        september_mask = df['date'].dt.month == 9
        df['playoff_chase_energy'] = np.where(september_mask, 0.015, 0.0)
        
        return df
    
    def _calculate_schedule_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate schedule density and workload features."""
        
        def calculate_batter_schedule(batter_data):
            """Calculate schedule density for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            # Initialize features
            data['games_last_7_days'] = 0
            data['games_last_14_days'] = 0
            data['games_next_7_days'] = 0
            data['schedule_intensity'] = 0.0
            data['upcoming_workload'] = 0.0
            data['recent_workload'] = 0.0
            
            for i in range(len(data)):
                current_date = data.iloc[i]['date']
                
                # Games in last 7 days
                week_start = current_date - timedelta(days=7)
                recent_7d = data[(data['date'] >= week_start) & (data['date'] <= current_date)]
                data.iloc[i, data.columns.get_loc('games_last_7_days')] = len(recent_7d)
                
                # Games in last 14 days
                two_weeks_start = current_date - timedelta(days=14)
                recent_14d = data[(data['date'] >= two_weeks_start) & (data['date'] <= current_date)]
                data.iloc[i, data.columns.get_loc('games_last_14_days')] = len(recent_14d)
                
                # Games in next 7 days
                week_end = current_date + timedelta(days=7)
                upcoming_7d = data[(data['date'] >= current_date) & (data['date'] <= week_end)]
                data.iloc[i, data.columns.get_loc('games_next_7_days')] = len(upcoming_7d)
                
                # Schedule intensity (recent + upcoming)
                intensity = (len(recent_7d) + len(upcoming_7d)) / 14.0  # Games per day
                data.iloc[i, data.columns.get_loc('schedule_intensity')] = intensity
                
                # Recent workload score
                recent_workload = len(recent_7d) / 7.0  # Average games per day
                data.iloc[i, data.columns.get_loc('recent_workload')] = recent_workload
                
                # Upcoming workload score
                upcoming_workload = len(upcoming_7d) / 7.0
                data.iloc[i, data.columns.get_loc('upcoming_workload')] = upcoming_workload
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_schedule)
    
    def _calculate_performance_timing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance timing and rhythm features."""
        
        # Initialize features
        df['optimal_performance_window'] = 0
        df['suboptimal_timing_penalty'] = 0.0
        df['circadian_mismatch'] = 0.0
        df['time_preference_alignment'] = 0.0
        
        # Individual player preferences (simplified - in practice would use historical data)
        # For now, assume most players prefer afternoon games
        
        df['optimal_performance_window'] = ((df['game_hour'] >= 13) & (df['game_hour'] <= 17)).astype(int)
        
        # Suboptimal timing penalties
        very_early = df['game_hour'] < 11
        very_late = df['game_hour'] > 22
        df['suboptimal_timing_penalty'] = np.where(very_early, 0.03, 0.0) + np.where(very_late, 0.02, 0.0)
        
        # Circadian mismatch (distance from optimal time)
        optimal_hour = 15  # 3 PM
        hour_distance = np.abs(df['game_hour'] - optimal_hour)
        df['circadian_mismatch'] = np.minimum(hour_distance / 12.0, 1.0)  # 0-1 scale
        
        # Time preference alignment (how well game time matches player preference)
        # Simplified: most players prefer 13-17 hour window
        preference_match = ((df['game_hour'] >= 13) & (df['game_hour'] <= 17)).astype(float)
        df['time_preference_alignment'] = preference_match
        
        return df
    
    def _calculate_circadian_factor(self, hour: int) -> float:
        """Calculate circadian performance factor for given hour."""
        # Sinusoidal model with peak at 2 PM and trough at 6 AM
        peak_hour = self.circadian_params.peak_performance_hour
        amplitude = self.circadian_params.amplitude
        
        # Convert to radians (24 hour cycle)
        hour_angle = (hour - peak_hour) * 2 * math.pi / 24
        
        # Cosine function: 1.0 at peak, (1.0 - 2*amplitude) at trough
        factor = 1.0 - amplitude * (1 - math.cos(hour_angle))
        
        return factor

def calculate_fatigue_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate correlations between fatigue features and performance."""
    if 'hit_hr' not in df.columns:
        return {}
    
    fatigue_features = [
        'fatigue_level', 'games_without_rest', 'jet_lag_factor', 
        'season_fatigue_factor', 'rest_quality_score'
    ]
    
    correlations = {}
    for feature in fatigue_features:
        if feature in df.columns:
            corr = df[feature].corr(df['hit_hr'])
            if not pd.isna(corr):
                correlations[feature] = corr
    
    return correlations

def analyze_circadian_patterns(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze circadian performance patterns."""
    if 'game_hour' not in df.columns or 'hit_hr' not in df.columns:
        return {}
    
    hourly_performance = {}
    
    for hour in range(24):
        hour_data = df[df['game_hour'] == hour]
        if len(hour_data) > 10:  # Minimum sample size
            hr_rate = hour_data['hit_hr'].mean()
            games = len(hour_data)
            hourly_performance[hour] = {
                'hr_rate': hr_rate,
                'games': games
            }
    
    return hourly_performance

def validate_fatigue_logic(df: pd.DataFrame) -> Dict[str, any]:
    """Validate fatigue calculation logic."""
    validation = {}
    
    if 'consecutive_games' in df.columns and 'fatigue_level' in df.columns:
        # Check if fatigue increases with consecutive games
        correlation = df['consecutive_games'].corr(df['fatigue_level'])
        validation['fatigue_consecutive_correlation'] = correlation
        validation['fatigue_increases_with_games'] = correlation > 0.1
    
    if 'rest_quality_score' in df.columns and 'days_since_rest' in df.columns:
        # Check if rest quality decreases with days since rest
        correlation = df['days_since_rest'].corr(df['rest_quality_score'])
        validation['rest_quality_correlation'] = correlation
        validation['rest_quality_decreases'] = correlation < -0.1
    
    if 'jet_lag_factor' in df.columns and 'timezone_change' in df.columns:
        # Check if jet lag increases with timezone changes
        correlation = df['timezone_change'].corr(df['jet_lag_factor'])
        validation['jet_lag_correlation'] = correlation
        validation['jet_lag_increases_with_tz'] = correlation > 0.3
    
    return validation