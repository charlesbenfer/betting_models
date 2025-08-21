"""
Weather Impact Features
======================

Features that capture weather conditions and their impact on home run probability.
Includes wind speed/direction, temperature, humidity, barometric pressure, and derived metrics.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import math

from data_utils import DataValidator

logger = logging.getLogger(__name__)

class WeatherFeatureCalculator:
    """Calculate weather-based features for home run prediction."""
    
    def __init__(self):
        self.validator = DataValidator()
        
        # Physical constants for calculations
        self.EARTH_RADIUS_FT = 20925525  # Earth radius in feet
        self.GRAVITY = 32.174  # ft/s²
        self.AIR_DENSITY_SEA_LEVEL = 0.075  # lb/ft³ at 59°F, 29.92 inHg
    
    def calculate_weather_features(self, batter_games_df: pd.DataFrame, 
                                 weather_data: Optional[pd.DataFrame] = None,
                                 use_real_weather: bool = True,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate all weather impact features.
        
        Args:
            batter_games_df: Batter games to add features to
            weather_data: Optional weather data (if None, will attempt to fetch real data)
            use_real_weather: Whether to attempt fetching real weather data
            start_date: Start date for weather data fetching
            end_date: End date for weather data fetching
        
        Returns:
            DataFrame with weather features added
        """
        logger.info("Calculating weather impact features...")
        
        result_df = batter_games_df.copy()
        
        # If no weather data provided, try to get real weather data first
        if weather_data is None:
            if use_real_weather:
                weather_data = self._fetch_real_weather_data(result_df, start_date, end_date)
            else:
                weather_data = self._create_enhanced_synthetic_weather(result_df)
        
        # Merge weather data
        result_df = self._merge_weather_data(result_df, weather_data)
        
        # Add basic weather impact features
        result_df = self._add_basic_weather_features(result_df)
        
        # Add atmospheric physics features
        result_df = self._add_atmospheric_features(result_df)
        
        # Add wind impact features
        result_df = self._add_wind_impact_features(result_df)
        
        # Add composite weather indices
        result_df = self._add_weather_indices(result_df)
        
        # Add ballpark-specific weather adjustments
        result_df = self._add_ballpark_weather_adjustments(result_df)
        
        logger.info("Weather feature calculation complete")
        return result_df
    
    def _create_enhanced_synthetic_weather(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Create realistic synthetic weather data with seasonal and regional patterns."""
        logger.info("Creating enhanced synthetic weather data...")
        
        # Get unique game dates and stadiums
        unique_games = games_df[['date', 'stadium']].drop_duplicates()
        
        weather_records = []
        
        for _, game in unique_games.iterrows():
            game_date = pd.to_datetime(game['date'])
            stadium = game['stadium']
            
            # Skip if invalid date
            if pd.isna(game_date):
                continue
                
            # Generate realistic weather based on date and location
            weather = self._generate_realistic_weather(game_date, stadium)
            weather['date'] = game['date']
            weather['stadium'] = stadium
            
            weather_records.append(weather)
        
        weather_df = pd.DataFrame(weather_records)
        logger.info(f"Created synthetic weather for {len(weather_df)} games")
        
        return weather_df
    
    def _fetch_real_weather_data(self, games_df: pd.DataFrame, 
                               start_date: Optional[str] = None, 
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """Attempt to fetch real weather data using the weather scraper."""
        try:
            from weather_scraper import WeatherDataScraper
            
            scraper = WeatherDataScraper()
            
            # Validate API keys
            api_status = scraper.validate_api_keys()
            if not any(api_status.values()):
                logger.warning("No weather API keys found - using synthetic weather")
                return self._create_enhanced_synthetic_weather(games_df)
            
            # Determine date range
            if start_date is None:
                start_date = games_df['date'].min().strftime('%Y-%m-%d')
            if end_date is None:
                end_date = games_df['date'].max().strftime('%Y-%m-%d')
            
            logger.info(f"Attempting to fetch real weather data for {start_date} to {end_date}")
            
            # Fetch real weather data
            weather_data = scraper.get_historical_weather(start_date, end_date, games_df)
            
            # Check if we got reasonable coverage
            weather_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']
            coverage = weather_data[weather_cols].notna().all(axis=1).mean()
            
            if coverage > 0.5:  # At least 50% real weather coverage
                logger.info(f"Successfully fetched real weather data with {coverage:.1%} coverage")
                return weather_data
            else:
                logger.warning(f"Low real weather coverage ({coverage:.1%}) - using synthetic fallback")
                return self._create_enhanced_synthetic_weather(games_df)
                
        except ImportError:
            logger.error("Weather scraper module not available - using synthetic weather")
            return self._create_enhanced_synthetic_weather(games_df)
        except Exception as e:
            logger.warning(f"Failed to fetch real weather data: {e} - using synthetic fallback")
            return self._create_enhanced_synthetic_weather(games_df)
    
    def _generate_realistic_weather(self, game_date: pd.Timestamp, 
                                  stadium: str) -> Dict[str, float]:
        """Generate realistic weather for a specific date and stadium."""
        
        # Handle invalid dates
        if pd.isna(game_date):
            month = 6  # Default to June
        else:
            month = game_date.month
        
        # Regional climate patterns (simplified)
        climate_zones = {
            'hot_humid': ['MIA', 'HOU', 'TB', 'ATL'],
            'hot_dry': ['ARI', 'TEX', 'LAA', 'OAK'],
            'moderate': ['SF', 'SEA', 'SD'],
            'cold': ['MIN', 'DET', 'CLE', 'CHC', 'MIL'],
            'variable': ['NYY', 'BOS', 'PHI', 'WSN', 'BAL']
        }
        
        # Determine climate zone  
        if pd.isna(stadium) or stadium is None:
            stadium_abbrev = 'NYC'  # Default
        else:
            stadium_abbrev = str(stadium)[:3].upper()
        climate = 'variable'  # default
        for zone, stadiums in climate_zones.items():
            if any(abbrev in stadium_abbrev for abbrev in stadiums):
                climate = zone
                break
        
        # Base temperature by season and climate
        temp_base = {
            'hot_humid': [75, 78, 85, 88, 86, 82, 79, 76],    # Mar-Oct
            'hot_dry': [70, 75, 85, 95, 98, 90, 85, 75],
            'moderate': [60, 63, 68, 72, 68, 65, 62, 58],
            'cold': [45, 55, 70, 78, 76, 68, 58, 48],
            'variable': [50, 60, 72, 80, 78, 70, 60, 52]
        }
        
        # Get base temperature for this month (Mar=0, Oct=7)
        month_idx = max(0, min(7, month - 3))
        base_temp = temp_base[climate][month_idx]
        
        # Add realistic random variation
        if pd.isna(game_date):
            date_part = "2024-06-01"
        else:
            date_part = str(game_date.date())
        
        if pd.isna(stadium) or stadium is None:
            stadium_part = "default"
        else:
            stadium_part = str(stadium)
            
        seed_str = f"{date_part}_{stadium_part}"
        np.random.seed(hash(seed_str) % 2**32)
        
        temperature = np.random.normal(base_temp, 8)
        temperature = np.clip(temperature, 40, 105)  # Reasonable bounds
        
        # Wind speed (mph) - typically 0-20 mph, occasionally higher
        wind_speed = np.random.exponential(5)
        wind_speed = np.clip(wind_speed, 0, 35)
        
        # Wind direction (degrees) - uniform distribution
        wind_direction = np.random.uniform(0, 360)
        
        # Humidity based on climate and temperature
        humidity_base = {
            'hot_humid': 70, 'hot_dry': 30, 'moderate': 50,
            'cold': 60, 'variable': 55
        }
        humidity = np.random.normal(humidity_base[climate], 15)
        humidity = np.clip(humidity, 20, 95)
        
        # Barometric pressure (inHg) - normal around 29.92
        pressure = np.random.normal(29.92, 0.3)
        pressure = np.clip(pressure, 29.0, 31.0)
        
        return {
            'temperature': temperature,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'humidity': humidity,
            'pressure': pressure
        }
    
    def _merge_weather_data(self, games_df: pd.DataFrame, 
                          weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather data with games data."""
        
        # Merge on date and stadium
        if 'stadium' in weather_df.columns:
            merge_cols = ['date', 'stadium']
        else:
            merge_cols = ['date']
        
        result_df = games_df.merge(
            weather_df,
            on=merge_cols,
            how='left'
        )
        
        # Fill missing weather with neutral values
        weather_cols = ['temperature', 'wind_speed', 'wind_direction', 'humidity', 'pressure']
        defaults = {'temperature': 72, 'wind_speed': 5, 'wind_direction': 0, 'humidity': 50, 'pressure': 29.92}
        
        for col in weather_cols:
            if col in result_df.columns:
                result_df[col] = result_df[col].fillna(defaults[col])
            else:
                result_df[col] = defaults[col]
        
        return result_df
    
    def _add_basic_weather_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add basic weather impact features."""
        logger.info("Adding basic weather features...")
        
        df = games_df.copy()
        
        # Temperature effects
        df['temp_hr_factor'] = self._calculate_temperature_factor(df['temperature'])
        df['temp_category'] = pd.cut(df['temperature'], 
                                   bins=[0, 60, 75, 85, 120], 
                                   labels=['cold', 'cool', 'warm', 'hot'])
        
        # Wind effects
        df['wind_hr_factor'] = self._calculate_wind_factor(df['wind_speed'])
        df['wind_category'] = pd.cut(df['wind_speed'],
                                   bins=[0, 5, 10, 15, 50],
                                   labels=['calm', 'light', 'moderate', 'strong'])
        
        # Humidity effects  
        df['humidity_hr_factor'] = self._calculate_humidity_factor(df['humidity'])
        
        # Pressure effects
        df['pressure_hr_factor'] = self._calculate_pressure_factor(df['pressure'])
        
        return df
    
    def _calculate_temperature_factor(self, temperature: pd.Series) -> pd.Series:
        """Calculate temperature's impact on HR probability."""
        # Warmer air is less dense, ball travels farther
        # Optimal around 80-85°F, diminishing returns after 90°F
        
        temp_factor = np.where(
            temperature < 70,
            0.85 + (temperature - 50) * 0.0075,  # Cold penalty
            np.where(
                temperature < 85,
                1.0 + (temperature - 70) * 0.02,  # Warm boost
                1.3 - (temperature - 85) * 0.01   # Hot diminishing returns
            )
        )
        
        return np.clip(temp_factor, 0.7, 1.4)
    
    def _calculate_wind_factor(self, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind's impact on HR probability (directional effect added later)."""
        # More wind generally helps carry, but too much creates turbulence
        
        wind_factor = np.where(
            wind_speed < 15,
            1.0 + wind_speed * 0.02,  # Linear boost up to 15 mph
            1.3 - (wind_speed - 15) * 0.01  # Diminishing returns after 15 mph
        )
        
        return np.clip(wind_factor, 0.8, 1.5)
    
    def _calculate_humidity_factor(self, humidity: pd.Series) -> pd.Series:
        """Calculate humidity's impact on HR probability."""
        # Lower humidity = less air resistance = more carry
        # Optimal around 30-40% humidity
        
        humidity_factor = np.where(
            humidity < 50,
            1.1 - humidity * 0.002,  # Dry air helps
            1.1 - (humidity - 50) * 0.004  # Humid air hurts more
        )
        
        return np.clip(humidity_factor, 0.85, 1.15)
    
    def _calculate_pressure_factor(self, pressure: pd.Series) -> pd.Series:
        """Calculate barometric pressure's impact on HR probability."""
        # Lower pressure = less air density = more carry
        # 29.92 inHg is standard sea level
        
        pressure_factor = 1.0 + (29.92 - pressure) * 0.05
        
        return np.clip(pressure_factor, 0.9, 1.1)
    
    def _add_atmospheric_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add atmospheric physics-based features."""
        logger.info("Adding atmospheric physics features...")
        
        df = games_df.copy()
        
        # Air density calculation
        df['air_density'] = self._calculate_air_density(
            df['temperature'], df['pressure'], df['humidity']
        )
        
        # Air density relative to standard conditions
        df['air_density_ratio'] = df['air_density'] / self.AIR_DENSITY_SEA_LEVEL
        
        # Ball flight distance adjustment
        df['flight_distance_factor'] = 1.0 / df['air_density_ratio']
        
        # Drag coefficient adjustment
        df['drag_factor'] = df['air_density_ratio']
        
        return df
    
    def _calculate_air_density(self, temp_f: pd.Series, pressure_inhg: pd.Series, 
                             humidity_pct: pd.Series) -> pd.Series:
        """Calculate air density using ideal gas law with humidity correction."""
        
        # Convert to SI units
        temp_k = (temp_f - 32) * 5/9 + 273.15  # Kelvin
        pressure_pa = pressure_inhg * 3386.39  # Pascals
        
        # Saturation vapor pressure (simplified approximation)
        vapor_pressure = humidity_pct / 100 * 610.7 * np.exp(
            17.27 * (temp_k - 273.15) / (temp_k - 35.85)
        )
        
        # Dry air pressure
        dry_pressure = pressure_pa - vapor_pressure
        
        # Air density (kg/m³)
        R_dry = 287.05  # J/(kg·K) for dry air
        R_vapor = 461.5  # J/(kg·K) for water vapor
        
        density_si = (dry_pressure / (R_dry * temp_k) + 
                     vapor_pressure / (R_vapor * temp_k))
        
        # Convert to lb/ft³
        density_imperial = density_si * 0.062428
        
        return density_imperial
    
    def _add_wind_impact_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add wind direction and stadium-specific wind features."""
        logger.info("Adding wind impact features...")
        
        df = games_df.copy()
        
        # Wind direction categories
        df['wind_direction_category'] = self._categorize_wind_direction(df['wind_direction'])
        
        # Stadium-specific wind effects (simplified)
        df = self._add_stadium_wind_effects(df)
        
        # Effective wind speed for HR (considering direction)
        df['effective_wind_speed'] = self._calculate_effective_wind_speed(
            df['wind_speed'], df['wind_direction'], df.get('stadium', 'Generic')
        )
        
        # Wind assistance factor
        df['wind_assistance_factor'] = self._calculate_wind_assistance(
            df['effective_wind_speed']
        )
        
        return df
    
    def _categorize_wind_direction(self, wind_direction: pd.Series) -> pd.Series:
        """Categorize wind direction relative to home run hitting."""
        
        # Simplified: assume most HRs go to center/left-center (roughly 45-135 degrees)
        # Favorable wind: blowing out to center field (45-135 degrees)
        # Unfavorable: blowing in (225-315 degrees)
        # Cross wind: other directions
        
        categories = []
        for direction in wind_direction:
            if 45 <= direction <= 135:
                categories.append('favorable')  # Blowing out
            elif 225 <= direction <= 315:
                categories.append('unfavorable')  # Blowing in
            else:
                categories.append('neutral')  # Cross wind
        
        return pd.Series(categories, index=wind_direction.index)
    
    def _add_stadium_wind_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add stadium-specific wind pattern effects."""
        
        # Stadium wind characteristics (simplified)
        stadium_wind_patterns = {
            'Wrigley Field': {'swirling': True, 'wind_amplification': 1.3},
            'Coors Field': {'altitude_effect': True, 'wind_amplification': 1.1},
            'Minute Maid Park': {'dome_effect': True, 'wind_amplification': 0.3},
            'Tropicana Field': {'dome_effect': True, 'wind_amplification': 0.1},
            'default': {'wind_amplification': 1.0}
        }
        
        df['stadium_wind_factor'] = 1.0
        
        for stadium in df['stadium'].unique():
            if pd.isna(stadium):
                continue
                
            pattern = stadium_wind_patterns.get(stadium, stadium_wind_patterns['default'])
            mask = df['stadium'] == stadium
            
            df.loc[mask, 'stadium_wind_factor'] = pattern['wind_amplification']
        
        return df
    
    def _calculate_effective_wind_speed(self, wind_speed: pd.Series, 
                                      wind_direction: pd.Series,
                                      stadium: pd.Series) -> pd.Series:
        """Calculate effective wind speed for home run assistance."""
        
        # Project wind speed onto home run trajectory
        # Assume average HR direction is 90 degrees (center field)
        hr_direction = 90
        
        wind_component = wind_speed * np.cos(np.radians(wind_direction - hr_direction))
        
        return wind_component
    
    def _calculate_wind_assistance(self, effective_wind_speed: pd.Series) -> pd.Series:
        """Calculate wind assistance factor for HRs."""
        
        # Positive wind = tailwind = helps HRs
        # Negative wind = headwind = hurts HRs
        
        assistance_factor = 1.0 + effective_wind_speed * 0.02
        
        return np.clip(assistance_factor, 0.7, 1.4)
    
    def _add_weather_indices(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add composite weather indices."""
        logger.info("Adding composite weather indices...")
        
        df = games_df.copy()
        
        # Overall weather favorability index
        df['weather_favorability_index'] = (
            df['temp_hr_factor'] * 0.3 +
            df['wind_hr_factor'] * 0.25 +
            df['humidity_hr_factor'] * 0.2 +
            df['pressure_hr_factor'] * 0.15 +
            df['wind_assistance_factor'] * 0.1
        )
        
        # Atmospheric carry index (physics-based)
        df['atmospheric_carry_index'] = (
            df['flight_distance_factor'] * 0.6 +
            df['wind_assistance_factor'] * 0.4
        )
        
        # Weather category
        df['weather_category'] = pd.cut(
            df['weather_favorability_index'],
            bins=[0, 0.9, 1.1, 2.0],
            labels=['unfavorable', 'neutral', 'favorable']
        )
        
        return df
    
    def _add_ballpark_weather_adjustments(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add ballpark-specific weather adjustments."""
        logger.info("Adding ballpark weather adjustments...")
        
        df = games_df.copy()
        
        # Ballpark elevation effects
        elevation_effects = {
            'Coors Field': 1.15,  # Mile high
            'Chase Field': 1.05,   # Phoenix elevation
            'Kauffman Stadium': 1.03,  # Kansas City elevation
            'default': 1.0
        }
        
        df['elevation_factor'] = 1.0
        for stadium in df['stadium'].unique():
            if pd.isna(stadium):
                continue
            factor = elevation_effects.get(stadium, elevation_effects['default'])
            df.loc[df['stadium'] == stadium, 'elevation_factor'] = factor
        
        # Combined ballpark-weather factor
        df['ballpark_weather_factor'] = (
            df['weather_favorability_index'] * df['elevation_factor']
        )
        
        return df

# Export the main class
__all__ = ['WeatherFeatureCalculator']