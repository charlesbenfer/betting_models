"""
Ballpark-Specific Advanced Features (Step 6)
==========================================

Advanced ballpark modeling beyond basic park factors, including:
- Dimensional analysis and carry distances
- Weather-ballpark interactions
- Batter-specific park performance
- Time-based park effects
- Directional pull factors
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class BallparkDimensions:
    """Detailed ballpark dimensions and characteristics."""
    name: str
    left_field_distance: int
    center_field_distance: int
    right_field_distance: int
    left_field_height: int
    center_field_height: int
    right_field_height: int
    foul_territory: str  # 'small', 'medium', 'large'
    elevation: int  # feet above sea level
    latitude: float
    longitude: float
    dome: bool
    surface: str  # 'grass', 'turf'
    
class BallparkFeatureCalculator:
    """Calculate advanced ballpark-specific features."""
    
    def __init__(self):
        self.ballpark_dimensions = self._initialize_ballpark_data()
        self.baseline_hr_rates = {}  # Stadium baseline HR rates
        
    def calculate_ballpark_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive ballpark features.
        
        Args:
            df: DataFrame with batting data including stadium info
            
        Returns:
            DataFrame with added ballpark features
        """
        logger.info("Calculating advanced ballpark features...")
        
        # Ensure required columns
        required_cols = ['stadium']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return df
            
        df_copy = df.copy()
        
        # Calculate different categories of ballpark features
        df_copy = self._calculate_dimensional_features(df_copy)
        df_copy = self._calculate_carry_distance_features(df_copy)
        df_copy = self._calculate_directional_features(df_copy)
        df_copy = self._calculate_batter_park_interaction(df_copy)
        df_copy = self._calculate_weather_park_interaction(df_copy)
        df_copy = self._calculate_temporal_park_effects(df_copy)
        df_copy = self._calculate_park_context_features(df_copy)
        
        logger.info("Advanced ballpark features calculation completed")
        return df_copy
    
    def _initialize_ballpark_data(self) -> Dict[str, BallparkDimensions]:
        """Initialize detailed ballpark dimensional data."""
        return {
            'Fenway Park': BallparkDimensions(
                name='Fenway Park',
                left_field_distance=310, center_field_distance=420, right_field_distance=302,
                left_field_height=37, center_field_height=17, right_field_height=3,
                foul_territory='small', elevation=20, latitude=42.3467, longitude=-71.0972,
                dome=False, surface='grass'
            ),
            'Yankee Stadium': BallparkDimensions(
                name='Yankee Stadium',
                left_field_distance=318, center_field_distance=408, right_field_distance=314,
                left_field_height=6, center_field_height=8, right_field_height=8,
                foul_territory='medium', elevation=55, latitude=40.8296, longitude=-73.9262,
                dome=False, surface='grass'
            ),
            'Coors Field': BallparkDimensions(
                name='Coors Field',
                left_field_distance=347, center_field_distance=415, right_field_distance=350,
                left_field_height=12, center_field_height=12, right_field_height=12,
                foul_territory='large', elevation=5200, latitude=39.7559, longitude=-104.9942,
                dome=False, surface='grass'
            ),
            'Petco Park': BallparkDimensions(
                name='Petco Park',
                left_field_distance=334, center_field_distance=396, right_field_distance=322,
                left_field_height=13, center_field_height=12, right_field_height=12,
                foul_territory='large', elevation=62, latitude=32.7073, longitude=-117.1566,
                dome=False, surface='grass'
            ),
            'Great American Ball Park': BallparkDimensions(
                name='Great American Ball Park',
                left_field_distance=325, center_field_distance=404, right_field_distance=325,
                left_field_height=12, center_field_height=12, right_field_height=12,
                foul_territory='medium', elevation=550, latitude=39.0974, longitude=-84.5081,
                dome=False, surface='grass'
            ),
            'Minute Maid Park': BallparkDimensions(
                name='Minute Maid Park',
                left_field_distance=315, center_field_distance=436, right_field_distance=326,
                left_field_height=19, center_field_height=13, right_field_height=7,
                foul_territory='small', elevation=22, latitude=29.7572, longitude=-95.3555,
                dome=True, surface='grass'
            ),
            'Oracle Park': BallparkDimensions(
                name='Oracle Park',
                left_field_distance=339, center_field_distance=399, right_field_distance=309,
                left_field_height=25, center_field_height=8, right_field_height=25,
                foul_territory='large', elevation=13, latitude=37.7786, longitude=-122.3893,
                dome=False, surface='grass'
            ),
            'Wrigley Field': BallparkDimensions(
                name='Wrigley Field',
                left_field_distance=355, center_field_distance=400, right_field_distance=353,
                left_field_height=11, center_field_height=11, right_field_height=11,
                foul_territory='small', elevation=595, latitude=41.9484, longitude=-87.6553,
                dome=False, surface='grass'
            ),
            'Tropicana Field': BallparkDimensions(
                name='Tropicana Field',
                left_field_distance=315, center_field_distance=404, right_field_distance=322,
                left_field_height=10, center_field_height=10, right_field_height=10,
                foul_territory='medium', elevation=11, latitude=27.7682, longitude=-82.6534,
                dome=True, surface='turf'
            ),
            'Kauffman Stadium': BallparkDimensions(
                name='Kauffman Stadium',
                left_field_distance=330, center_field_distance=410, right_field_distance=330,
                left_field_height=12, center_field_height=12, right_field_height=12,
                foul_territory='large', elevation=750, latitude=39.0517, longitude=-94.4803,
                dome=False, surface='grass'
            ),
            # Add more stadiums as needed - simplified for key examples
        }
    
    def _calculate_dimensional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features based on ballpark dimensions."""
        
        # Initialize features
        df['park_left_field_distance'] = 0
        df['park_center_field_distance'] = 0  
        df['park_right_field_distance'] = 0
        df['park_left_field_height'] = 0
        df['park_center_field_height'] = 0
        df['park_right_field_height'] = 0
        df['park_foul_territory_size'] = 0  # 1=small, 2=medium, 3=large
        df['park_elevation'] = 0
        df['park_is_dome'] = 0
        df['park_surface_turf'] = 0
        
        # Advanced dimensional features
        df['park_hr_difficulty_index'] = 0.0
        df['park_symmetry_factor'] = 0.0
        df['park_wall_height_factor'] = 0.0
        df['park_foul_territory_hr_boost'] = 0.0
        
        for stadium, dims in self.ballpark_dimensions.items():
            mask = df['stadium'] == stadium
            if mask.any():
                # Basic dimensions
                df.loc[mask, 'park_left_field_distance'] = dims.left_field_distance
                df.loc[mask, 'park_center_field_distance'] = dims.center_field_distance
                df.loc[mask, 'park_right_field_distance'] = dims.right_field_distance
                df.loc[mask, 'park_left_field_height'] = dims.left_field_height
                df.loc[mask, 'park_center_field_height'] = dims.center_field_height
                df.loc[mask, 'park_right_field_height'] = dims.right_field_height
                df.loc[mask, 'park_elevation'] = dims.elevation
                df.loc[mask, 'park_is_dome'] = int(dims.dome)
                df.loc[mask, 'park_surface_turf'] = int(dims.surface == 'turf')
                
                # Foul territory encoding
                foul_territory_map = {'small': 1, 'medium': 2, 'large': 3}
                df.loc[mask, 'park_foul_territory_size'] = foul_territory_map.get(dims.foul_territory, 2)
                
                # HR difficulty index (higher = harder to hit HRs)
                avg_distance = (dims.left_field_distance + dims.center_field_distance + dims.right_field_distance) / 3
                avg_height = (dims.left_field_height + dims.center_field_height + dims.right_field_height) / 3
                difficulty = (avg_distance - 320) / 10 + (avg_height - 10) / 5  # Normalized
                df.loc[mask, 'park_hr_difficulty_index'] = difficulty
                
                # Symmetry factor (0 = symmetric, higher = more asymmetric)
                left_right_diff = abs(dims.left_field_distance - dims.right_field_distance)
                symmetry = left_right_diff / 50  # Normalized
                df.loc[mask, 'park_symmetry_factor'] = symmetry
                
                # Wall height factor (impact on home runs)
                wall_factor = (avg_height - 8) / 10  # Normalized around 8ft baseline
                df.loc[mask, 'park_wall_height_factor'] = wall_factor
                
                # Foul territory HR boost (large foul territory = fewer HRs caught)
                foul_boost_map = {'small': 0.05, 'medium': 0.0, 'large': -0.03}
                df.loc[mask, 'park_foul_territory_hr_boost'] = foul_boost_map.get(dims.foul_territory, 0.0)
        
        return df
    
    def _calculate_carry_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features related to ball carry distance and flight."""
        
        # Initialize features
        df['park_elevation_carry_boost'] = 0.0
        df['park_air_density_factor'] = 0.0
        df['park_dome_carry_reduction'] = 0.0
        df['park_coastal_humidity_factor'] = 0.0
        
        for stadium, dims in self.ballpark_dimensions.items():
            mask = df['stadium'] == stadium
            if mask.any():
                # Elevation carry boost (every 1000ft adds ~6% distance)
                elevation_boost = (dims.elevation - 500) / 1000 * 0.06
                df.loc[mask, 'park_elevation_carry_boost'] = elevation_boost
                
                # Air density factor (elevation + temperature effects)
                if 'temperature' in df.columns:
                    temp = df.loc[mask, 'temperature'].fillna(70)  # Default 70F
                else:
                    temp = 70  # Default temperature
                
                # Air density calculation (simplified)
                standard_density = 1.225  # kg/m³ at sea level, 15°C
                altitude_factor = np.exp(-dims.elevation * 0.00012)  # Exponential atmosphere
                temp_factor = (273.15 + 15) / (273.15 + (temp - 32) * 5/9)  # Temperature correction
                air_density_ratio = altitude_factor * temp_factor
                df.loc[mask, 'park_air_density_factor'] = 1 - air_density_ratio  # Lower density = more carry
                
                # Dome carry reduction (closed environment reduces carry)
                if dims.dome:
                    df.loc[mask, 'park_dome_carry_reduction'] = -0.02  # 2% reduction
                
                # Coastal humidity factor (higher humidity = less carry)
                coastal_stadiums = ['Oracle Park', 'Petco Park', 'Tropicana Field', 'Marlins Park']
                if stadium in coastal_stadiums:
                    df.loc[mask, 'park_coastal_humidity_factor'] = -0.015  # 1.5% reduction
        
        return df
    
    def _calculate_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate directional pull and spray features."""
        
        # Requires batter handedness and launch direction if available
        df['park_pull_factor_left'] = 0.0
        df['park_pull_factor_right'] = 0.0
        df['park_opposite_field_factor'] = 0.0
        df['park_center_field_factor'] = 0.0
        
        for stadium, dims in self.ballpark_dimensions.items():
            mask = df['stadium'] == stadium
            if mask.any():
                # Pull factors for left-handed batters (right field)
                rf_distance = dims.right_field_distance
                rf_height = dims.right_field_height
                pull_factor_left = (330 - rf_distance) / 20 + (8 - rf_height) / 10
                df.loc[mask, 'park_pull_factor_left'] = pull_factor_left
                
                # Pull factors for right-handed batters (left field)  
                lf_distance = dims.left_field_distance
                lf_height = dims.left_field_height
                pull_factor_right = (330 - lf_distance) / 20 + (8 - lf_height) / 10
                df.loc[mask, 'park_pull_factor_right'] = pull_factor_right
                
                # Opposite field factor (harder direction)
                opp_factor = min(pull_factor_left, pull_factor_right) * 0.7  # Opposite field is harder
                df.loc[mask, 'park_opposite_field_factor'] = opp_factor
                
                # Center field factor
                cf_distance = dims.center_field_distance
                cf_height = dims.center_field_height
                center_factor = (410 - cf_distance) / 30 + (10 - cf_height) / 15
                df.loc[mask, 'park_center_field_factor'] = center_factor
        
        return df
    
    def _calculate_batter_park_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate batter-specific park performance features."""
        
        # These require historical data analysis
        df['batter_park_hr_rate_boost'] = 0.0
        df['batter_park_historical_performance'] = 0.0
        df['batter_park_comfort_factor'] = 0.0
        
        if 'batter' in df.columns:
            # Group by batter and stadium to calculate historical performance
            for batter in df['batter'].unique():
                batter_mask = df['batter'] == batter
                batter_data = df[batter_mask]
                
                for stadium in batter_data['stadium'].unique():
                    stadium_mask = batter_data['stadium'] == stadium
                    combined_mask = batter_mask & (df['stadium'] == stadium)
                    
                    if combined_mask.sum() > 0:
                        # Historical HR rate at this park vs overall
                        if 'hit_hr' in df.columns:
                            park_hr_rate = batter_data[stadium_mask]['hit_hr'].mean()
                            overall_hr_rate = df[batter_mask]['hit_hr'].mean()
                            
                            if overall_hr_rate > 0:
                                hr_rate_boost = (park_hr_rate - overall_hr_rate) / overall_hr_rate
                                df.loc[combined_mask, 'batter_park_hr_rate_boost'] = hr_rate_boost
                        
                        # Games played at this park (comfort factor)
                        games_at_park = stadium_mask.sum()
                        comfort_factor = min(games_at_park / 20, 1.0)  # Max out at 20 games
                        df.loc[combined_mask, 'batter_park_comfort_factor'] = comfort_factor
        
        return df
    
    def _calculate_weather_park_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weather-ballpark interaction features."""
        
        df['park_wind_interaction'] = 0.0
        df['park_temperature_interaction'] = 0.0
        df['park_humidity_interaction'] = 0.0
        df['park_weather_hr_multiplier'] = 1.0
        
        weather_cols = ['wind_speed', 'wind_direction', 'temperature', 'humidity']
        available_weather = [col for col in weather_cols if col in df.columns]
        
        if not available_weather:
            logger.info("No weather data available for park interactions")
            return df
        
        for stadium, dims in self.ballpark_dimensions.items():
            mask = df['stadium'] == stadium
            if not mask.any():
                continue
                
            # Wind interactions
            if 'wind_speed' in df.columns and 'wind_direction' in df.columns:
                wind_speed = df.loc[mask, 'wind_speed'].fillna(5)  # Default 5 mph
                wind_direction = df.loc[mask, 'wind_direction'].fillna(180)  # Default south
                
                # Stadium-specific wind effects
                if stadium == 'Wrigley Field':
                    # Wrigley is famous for wind effects
                    wind_interaction = wind_speed * 0.3  # Strong wind effect
                elif stadium == 'Oracle Park':
                    # Oracle Park has strong marine winds
                    wind_interaction = wind_speed * 0.25
                elif dims.dome:
                    # Domed stadiums have no wind
                    wind_interaction = 0
                else:
                    wind_interaction = wind_speed * 0.15  # Standard wind effect
                
                df.loc[mask, 'park_wind_interaction'] = wind_interaction
            
            # Temperature interactions
            if 'temperature' in df.columns:
                temp = df.loc[mask, 'temperature'].fillna(70)
                
                # High altitude + high temperature = maximum carry
                if dims.elevation > 3000:
                    temp_interaction = (temp - 70) * 0.02  # 2% per 10 degrees at altitude
                elif dims.dome:
                    temp_interaction = 0  # Controlled environment
                else:
                    temp_interaction = (temp - 70) * 0.01  # 1% per 10 degrees
                
                df.loc[mask, 'park_temperature_interaction'] = temp_interaction
            
            # Humidity interactions
            if 'humidity' in df.columns:
                humidity = df.loc[mask, 'humidity'].fillna(50)
                
                # High humidity reduces carry more at sea level
                if dims.elevation < 500:
                    humidity_interaction = -(humidity - 50) * 0.001  # Stronger effect at sea level
                else:
                    humidity_interaction = -(humidity - 50) * 0.0005  # Weaker at altitude
                
                df.loc[mask, 'park_humidity_interaction'] = humidity_interaction
            
            # Combined weather HR multiplier
            wind_factor = df.loc[mask, 'park_wind_interaction'].fillna(0) * 0.01
            temp_factor = df.loc[mask, 'park_temperature_interaction'].fillna(0)
            humidity_factor = df.loc[mask, 'park_humidity_interaction'].fillna(0)
            
            weather_multiplier = 1 + wind_factor + temp_factor + humidity_factor
            df.loc[mask, 'park_weather_hr_multiplier'] = weather_multiplier
        
        return df
    
    def _calculate_temporal_park_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based park effects (day/night, season, etc.)."""
        
        df['park_day_night_factor'] = 0.0
        df['park_season_factor'] = 0.0
        df['park_month_hr_factor'] = 0.0
        
        # Day/night effects
        if 'game_date' in df.columns:
            # Assume day games if hour < 17, night games otherwise
            # This would need actual game time data for precision
            
            for stadium, dims in self.ballpark_dimensions.items():
                mask = df['stadium'] == stadium
                if not mask.any():
                    continue
                
                # Some parks favor day/night differently
                if stadium == 'Wrigley Field':
                    # Wrigley historically favored day games
                    df.loc[mask, 'park_day_night_factor'] = 0.02  # Slight day game boost
                elif stadium == 'Fenway Park':
                    # Green Monster creates shadows in day games
                    df.loc[mask, 'park_day_night_factor'] = -0.01  # Slight night game boost
        
        # Seasonal effects
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            month = df['date'].dt.month
            
            for stadium, dims in self.ballpark_dimensions.items():
                mask = df['stadium'] == stadium
                if not mask.any():
                    continue
                
                # Cold weather parks have stronger seasonal effects
                if dims.latitude > 42:  # Northern parks
                    # April/May/September/October penalties for cold
                    cold_months = month.isin([4, 5, 9, 10])
                    df.loc[mask & cold_months, 'park_season_factor'] = -0.03
                    # Peak summer boost
                    hot_months = month.isin([6, 7, 8])
                    df.loc[mask & hot_months, 'park_season_factor'] = 0.02
                elif dims.latitude < 30:  # Southern parks
                    # Hot summer penalties
                    very_hot_months = month.isin([7, 8])
                    df.loc[mask & very_hot_months, 'park_season_factor'] = -0.01
        
        return df
    
    def _calculate_park_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate contextual park features."""
        
        df['park_offense_context'] = 0.0
        df['park_pitcher_context'] = 0.0
        df['park_defensive_context'] = 0.0
        
        for stadium, dims in self.ballpark_dimensions.items():
            mask = df['stadium'] == stadium
            if not mask.any():
                continue
            
            # Offensive context (how park affects hitting)
            offense_factors = []
            
            # Distance factor
            avg_distance = (dims.left_field_distance + dims.center_field_distance + dims.right_field_distance) / 3
            distance_factor = (330 - avg_distance) / 100  # Normalized
            offense_factors.append(distance_factor)
            
            # Height factor
            avg_height = (dims.left_field_height + dims.center_field_height + dims.right_field_height) / 3
            height_factor = (8 - avg_height) / 20  # Normalized
            offense_factors.append(height_factor)
            
            # Foul territory factor
            foul_factor_map = {'small': 0.02, 'medium': 0.0, 'large': -0.02}
            foul_factor = foul_factor_map.get(dims.foul_territory, 0.0)
            offense_factors.append(foul_factor)
            
            # Elevation factor
            elevation_factor = dims.elevation / 5000 * 0.1  # Up to 10% boost at 5000ft
            offense_factors.append(elevation_factor)
            
            offense_context = sum(offense_factors)
            df.loc[mask, 'park_offense_context'] = offense_context
            
            # Pitcher context (how park affects pitchers)
            pitcher_context = -offense_context  # Inverse of offense
            df.loc[mask, 'park_pitcher_context'] = pitcher_context
            
            # Defensive context (foul territory mainly)
            defensive_context = foul_factor * 2  # Foul territory helps defense more
            if dims.surface == 'turf':
                defensive_context += 0.01  # Turf helps defense slightly
            df.loc[mask, 'park_defensive_context'] = defensive_context
        
        return df

def calculate_park_baseline_rates(df: pd.DataFrame, min_games: int = 50) -> Dict[str, float]:
    """Calculate baseline HR rates for each park."""
    if 'stadium' not in df.columns or 'hit_hr' not in df.columns:
        return {}
    
    park_rates = {}
    for stadium in df['stadium'].unique():
        stadium_data = df[df['stadium'] == stadium]
        
        if len(stadium_data) >= min_games:
            hr_rate = stadium_data['hit_hr'].mean()
            park_rates[stadium] = hr_rate
        
    return park_rates

def analyze_park_effects(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze various park effects for validation."""
    if 'stadium' not in df.columns:
        return {}
    
    analysis = {}
    
    for stadium in df['stadium'].unique():
        stadium_data = df[df['stadium'] == stadium]
        
        if len(stadium_data) < 10:
            continue
        
        stats = {}
        
        # Basic stats
        if 'hit_hr' in df.columns:
            stats['hr_rate'] = stadium_data['hit_hr'].mean()
            stats['games'] = len(stadium_data)
        
        # Distance factors if available
        if 'park_left_field_distance' in df.columns:
            stats['avg_distance'] = (
                stadium_data['park_left_field_distance'].iloc[0] +
                stadium_data['park_center_field_distance'].iloc[0] +
                stadium_data['park_right_field_distance'].iloc[0]
            ) / 3
        
        # Environmental factors
        if 'park_elevation' in df.columns:
            stats['elevation'] = stadium_data['park_elevation'].iloc[0]
        
        if 'park_is_dome' in df.columns:
            stats['dome'] = bool(stadium_data['park_is_dome'].iloc[0])
        
        analysis[stadium] = stats
    
    return analysis