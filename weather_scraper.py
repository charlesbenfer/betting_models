"""
Weather Data Scraper
===================

Scrapes real historical weather data from multiple sources to enhance home run predictions.
Supports OpenWeatherMap API, Visual Crossing API, and NOAA data.
"""

import pandas as pd
import numpy as np
import logging
import requests
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

from data_utils import DataValidator, CacheManager

logger = logging.getLogger(__name__)

class WeatherDataScraper:
    """Scrape and manage real historical weather data."""
    
    def __init__(self):
        self.validator = DataValidator()
        self.cache_manager = CacheManager()
        
        # API keys from environment
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
        self.visualcrossing_api_key = os.getenv("VISUALCROSSING_API_KEY", "").strip()
        
        # Rate limiting and fallback control
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
        self.rate_limit_hit = False  # Track if we've hit rate limits
        self.use_synthetic_fallback = False  # Global flag to switch to synthetic
        
        # Stadium coordinates for weather lookups
        self.stadium_coordinates = {
            'Chase Field': (33.4453, -112.0667),
            'Truist Park': (33.8906, -84.4677),
            'Oriole Park at Camden Yards': (39.2840, -76.6218),
            'Fenway Park': (42.3467, -71.0972),
            'Wrigley Field': (41.9484, -87.6553),
            'Great American Ball Park': (39.0975, -84.5068),
            'Progressive Field': (41.4962, -81.6852),
            'Coors Field': (39.7559, -104.9942),
            'Guaranteed Rate Field': (41.8300, -87.6338),
            'Comerica Park': (42.3390, -83.0485),
            'Minute Maid Park': (29.7572, -95.3552),
            'Kauffman Stadium': (39.0517, -94.4803),
            'Angel Stadium': (33.8003, -117.8827),
            'Dodger Stadium': (34.0739, -118.2400),
            'loanDepot park': (25.7781, -80.2197),
            'American Family Field': (43.0280, -87.9712),
            'Target Field': (44.9817, -93.2777),
            'Citi Field': (40.7571, -73.8458),
            'Yankee Stadium': (40.8296, -73.9262),
            'Oakland Coliseum': (37.7516, -122.2008),
            'Citizens Bank Park': (39.9061, -75.1665),
            'PNC Park': (40.4469, -80.0057),
            'Petco Park': (32.7073, -117.1566),
            'T-Mobile Park': (47.5914, -122.3326),
            'Oracle Park': (37.7786, -122.3893),
            'Busch Stadium': (38.6226, -90.1928),
            'Tropicana Field': (27.7682, -82.6534),
            'Globe Life Field': (32.7471, -97.0825),
            'Rogers Centre': (43.6414, -79.3894),
            'Nationals Park': (38.8730, -77.0074)
        }
        
    def get_historical_weather(self, start_date: str, end_date: str, 
                             games_df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """
        Get real historical weather data for all games in the dataset.
        
        Args:
            start_date: Start date for weather data
            end_date: End date for weather data  
            games_df: DataFrame with game dates and stadiums
            use_cache: Whether to use cached weather data
            
        Returns:
            DataFrame with weather data for each game
        """
        logger.info(f"Fetching real historical weather data: {start_date} to {end_date}")
        
        # Check cache first
        cache_id = f"weather_data_{start_date}_{end_date}"
        if use_cache:
            cached_weather = self.cache_manager.load_cache(cache_id, "parquet")
            if cached_weather is not None:
                logger.info(f"Loaded weather data from cache: {len(cached_weather)} records")
                return self._merge_weather_with_games(cached_weather, games_df)
        
        # Get unique game dates and stadiums
        unique_games = games_df[['date', 'stadium']].drop_duplicates()
        unique_games = unique_games.dropna(subset=['stadium'])
        
        logger.info(f"Fetching weather for {len(unique_games)} unique game-stadium combinations")
        
        weather_records = []
        failed_requests = 0
        synthetic_count = 0
        
        for idx, (_, game) in enumerate(unique_games.iterrows()):
            if idx % 50 == 0:
                logger.info(f"Weather progress: {idx}/{len(unique_games)} ({idx/len(unique_games)*100:.1f}%)")
            
            try:
                game_date = pd.to_datetime(game['date']).date()
                stadium = game['stadium']
                
                # Check if we should use synthetic weather due to rate limits
                if self.use_synthetic_fallback:
                    weather_data = self._generate_synthetic_weather_for_game(game_date, stadium)
                    synthetic_count += 1
                else:
                    weather_data = self._fetch_weather_for_game(game_date, stadium)
                    
                    # If rate limit was hit during this request, switch to synthetic for remaining requests
                    if self.rate_limit_hit:
                        logger.warning(f"Rate limit detected! Switching to synthetic weather for remaining {len(unique_games) - idx - 1} requests")
                        self.use_synthetic_fallback = True
                        # Generate synthetic weather for this failed request
                        if not weather_data:
                            weather_data = self._generate_synthetic_weather_for_game(game_date, stadium)
                            synthetic_count += 1
                
                if weather_data:
                    weather_data['date'] = game['date']
                    weather_data['stadium'] = stadium
                    weather_records.append(weather_data)
                else:
                    failed_requests += 1
                    
            except Exception as e:
                logger.warning(f"Failed to fetch weather for {game['date']} at {game['stadium']}: {e}")
                failed_requests += 1
                
            # Rate limiting (only if not using synthetic fallback)
            if not self.use_synthetic_fallback:
                self._rate_limit()
        
        if not weather_records:
            logger.error("No weather data retrieved! Using synthetic weather as fallback.")
            return self._create_fallback_weather(games_df)
        
        weather_df = pd.DataFrame(weather_records)
        real_data_count = len(weather_df) - synthetic_count
        logger.info(f"Successfully fetched {len(weather_df)} weather records:")
        logger.info(f"  Real weather data: {real_data_count}")
        logger.info(f"  Synthetic weather data: {synthetic_count}")
        logger.info(f"  Failed requests: {failed_requests}")
        
        if self.rate_limit_hit:
            logger.info("Rate limits were encountered and synthetic fallback was used")
        
        # Cache the results
        self.cache_manager.save_cache(weather_df, cache_id, "parquet")
        
        return self._merge_weather_with_games(weather_df, games_df)
    
    def _fetch_weather_for_game(self, game_date: datetime.date, stadium: str) -> Optional[Dict[str, float]]:
        """Fetch weather data for a specific game date and stadium."""
        
        if stadium not in self.stadium_coordinates:
            logger.warning(f"No coordinates found for stadium: {stadium}")
            return None
        
        lat, lon = self.stadium_coordinates[stadium]
        
        # Try Visual Crossing first (more reliable for historical data)
        if self.visualcrossing_api_key:
            weather = self._fetch_visual_crossing_weather(game_date, lat, lon)
            if weather:
                return weather
        
        # Fallback to OpenWeatherMap
        if self.openweather_api_key:
            weather = self._fetch_openweather_historical(game_date, lat, lon)
            if weather:
                return weather
        
        # If both fail, return None (will use synthetic data)
        logger.warning(f"Could not fetch weather for {game_date} at {stadium}")
        return None
    
    def _fetch_visual_crossing_weather(self, game_date: datetime.date, lat: float, lon: float) -> Optional[Dict[str, float]]:
        """Fetch weather from Visual Crossing Weather API."""
        try:
            base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
            date_str = game_date.strftime("%Y-%m-%d")
            location_str = f"{lat},{lon}"
            
            params = {
                'key': self.visualcrossing_api_key,
                'unitGroup': 'us',
                'include': 'hours',
                'elements': 'temp,humidity,windspeed,winddir,pressure'
            }
            
            # URL format: /timeline/location/date
            url = f"{base_url}/{location_str}/{date_str}"
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'days' in data and len(data['days']) > 0:
                    day_data = data['days'][0]
                    
                    # Use evening hours if available, otherwise daily average
                    if 'hours' in day_data and len(day_data['hours']) > 0:
                        # Average evening hours (7-10 PM typical game time)
                        evening_hours = []
                        for h in day_data['hours']:
                            try:
                                if 'datetime' in h:
                                    hour = int(h['datetime'].split(':')[0])
                                    if 19 <= hour <= 22:
                                        evening_hours.append(h)
                            except (ValueError, KeyError, AttributeError):
                                continue
                        
                        if evening_hours:
                            avg_data = {
                                'temp': np.mean([h.get('temp', 72) for h in evening_hours]),
                                'humidity': np.mean([h.get('humidity', 50) for h in evening_hours]),
                                'windspeed': np.mean([h.get('windspeed', 5) for h in evening_hours]),
                                'winddir': np.mean([h.get('winddir', 0) for h in evening_hours]),
                                'pressure': np.mean([h.get('pressure', 29.92) for h in evening_hours])
                            }
                        else:
                            # Use daily average
                            avg_data = day_data
                    else:
                        avg_data = day_data
                    
                    return {
                        'temperature': float(avg_data.get('temp', 72)),
                        'humidity': float(avg_data.get('humidity', 50)),
                        'wind_speed': float(avg_data.get('windspeed', 5)),
                        'wind_direction': float(avg_data.get('winddir', 0)),
                        'pressure': float(avg_data.get('pressure', 29.92))
                    }
            
            elif response.status_code == 429:
                logger.warning("Visual Crossing API rate limit exceeded")
                self.rate_limit_hit = True  # Set flag to switch to synthetic
                time.sleep(5)  # Wait longer for rate limits
            else:
                logger.warning(f"Visual Crossing API error: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Visual Crossing API request failed: {e}")
        
        return None
    
    def _fetch_openweather_historical(self, game_date: datetime.date, lat: float, lon: float) -> Optional[Dict[str, float]]:
        """Fetch historical weather from OpenWeatherMap API."""
        try:
            # OpenWeatherMap historical data requires timestamp
            game_datetime = datetime.combine(game_date, datetime.min.time().replace(hour=20))  # 8 PM game time
            timestamp = int(game_datetime.timestamp())
            
            url = "http://api.openweathermap.org/data/3.0/onecall/timemachine"
            params = {
                'lat': lat,
                'lon': lon,
                'dt': timestamp,
                'appid': self.openweather_api_key,
                'units': 'imperial'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    weather = data['data'][0]
                    
                    return {
                        'temperature': float(weather.get('temp', 72)),
                        'humidity': float(weather.get('humidity', 50)),
                        'wind_speed': float(weather.get('wind_speed', 5) * 2.237),  # Convert m/s to mph
                        'wind_direction': float(weather.get('wind_deg', 0)),
                        'pressure': float(weather.get('pressure', 1013) * 0.02953)  # Convert hPa to inHg
                    }
            
            elif response.status_code == 429:
                logger.warning("OpenWeatherMap API rate limit exceeded")
                self.rate_limit_hit = True  # Set flag to switch to synthetic
                time.sleep(2)
            else:
                logger.warning(f"OpenWeatherMap API error: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"OpenWeatherMap API request failed: {e}")
        
        return None
    
    def _rate_limit(self):
        """Implement rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _merge_weather_with_games(self, weather_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather data with games, filling missing values with synthetic data."""
        
        # Merge on date and stadium
        result_df = games_df.merge(
            weather_df,
            on=['date', 'stadium'],
            how='left'
        )
        
        # Count missing weather data
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']
        missing_weather = result_df[weather_cols].isna().any(axis=1).sum()
        
        if missing_weather > 0:
            logger.info(f"Missing weather data for {missing_weather} games, using synthetic fallback")
            result_df = self._fill_missing_weather_synthetic(result_df)
        
        logger.info(f"Weather data coverage: {(len(result_df) - missing_weather)/len(result_df)*100:.1f}%")
        
        return result_df
    
    def _fill_missing_weather_synthetic(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing weather data with synthetic values."""
        from weather_features import WeatherFeatureCalculator
        
        weather_calc = WeatherFeatureCalculator()
        
        # Find rows with missing weather
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']
        missing_mask = games_df[weather_cols].isna().any(axis=1)
        
        if missing_mask.sum() == 0:
            return games_df
        
        # Generate synthetic weather for missing rows
        missing_games = games_df[missing_mask].copy()
        synthetic_weather = weather_calc._create_enhanced_synthetic_weather(missing_games)
        
        # Merge synthetic weather back
        result_df = games_df.copy()
        
        for col in weather_cols:
            if col in synthetic_weather.columns:
                # Fill missing values with synthetic data
                synthetic_merge = missing_games[['date', 'stadium']].merge(
                    synthetic_weather[['date', 'stadium', col]],
                    on=['date', 'stadium'],
                    how='left'
                )
                
                for idx in missing_games.index:
                    if pd.isna(result_df.loc[idx, col]):
                        synthetic_idx = synthetic_merge.index[synthetic_merge.index.isin([idx])][0] if len(synthetic_merge.index[synthetic_merge.index.isin([idx])]) > 0 else 0
                        result_df.loc[idx, col] = synthetic_merge.loc[synthetic_idx, col] if not pd.isna(synthetic_merge.loc[synthetic_idx, col]) else 72.0
        
        return result_df
    
    def _generate_synthetic_weather_for_game(self, game_date: datetime.date, stadium: str) -> Dict[str, float]:
        """Generate synthetic weather for a single game."""
        from weather_features import WeatherFeatureCalculator
        weather_calc = WeatherFeatureCalculator()
        
        # Create a minimal DataFrame for this game
        game_df = pd.DataFrame({
            'date': [pd.to_datetime(game_date)],
            'stadium': [stadium]
        })
        
        # Generate synthetic weather and extract for this game
        synthetic_weather_df = weather_calc._create_enhanced_synthetic_weather(game_df)
        
        if len(synthetic_weather_df) > 0:
            weather_row = synthetic_weather_df.iloc[0]
            return {
                'temperature': float(weather_row.get('temperature', 72)),
                'humidity': float(weather_row.get('humidity', 50)),
                'wind_speed': float(weather_row.get('wind_speed', 5)),
                'wind_direction': float(weather_row.get('wind_direction', 0)),
                'pressure': float(weather_row.get('pressure', 29.92))
            }
        else:
            # Fallback to basic synthetic weather
            return {
                'temperature': 72.0,
                'humidity': 50.0,
                'wind_speed': 5.0,
                'wind_direction': 0.0,
                'pressure': 29.92
            }
    
    def _create_fallback_weather(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic weather as complete fallback."""
        logger.warning("Using complete synthetic weather fallback")
        
        from weather_features import WeatherFeatureCalculator
        weather_calc = WeatherFeatureCalculator()
        return weather_calc._create_enhanced_synthetic_weather(games_df)
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate available API keys."""
        results = {
            'visual_crossing': bool(self.visualcrossing_api_key),
            'openweathermap': bool(self.openweather_api_key)
        }
        
        logger.info(f"API key availability: {results}")
        return results
    
    def reset_rate_limit_flags(self):
        """Reset rate limit flags for a new session."""
        self.rate_limit_hit = False
        self.use_synthetic_fallback = False
        logger.info("Rate limit flags reset for new session")
    
    def test_api_connection(self) -> Dict[str, bool]:
        """Test API connections with simple requests."""
        results = {}
        
        # Test Visual Crossing
        if self.visualcrossing_api_key:
            try:
                url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
                params = {
                    'key': self.visualcrossing_api_key,
                    'location': '40.7128,-74.0060',  # NYC
                    'startDateTime': '2024-01-01',
                    'endDateTime': '2024-01-01',
                    'unitGroup': 'us'
                }
                response = requests.get(f"{url}/2024-01-01", params=params, timeout=5)
                results['visual_crossing'] = response.status_code in [200, 429]  # 429 means API key works but rate limited
            except:
                results['visual_crossing'] = False
        else:
            results['visual_crossing'] = False
        
        # Test OpenWeatherMap
        if self.openweather_api_key:
            try:
                url = "http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': 40.7128,
                    'lon': -74.0060,
                    'appid': self.openweather_api_key
                }
                response = requests.get(url, params=params, timeout=5)
                results['openweathermap'] = response.status_code in [200, 429]
            except:
                results['openweathermap'] = False
        else:
            results['openweathermap'] = False
        
        logger.info(f"API connection test results: {results}")
        return results

# Export the main class
__all__ = ['WeatherDataScraper']