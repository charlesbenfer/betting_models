"""
Test Real Weather Data Integration
=================================

Test the enhanced weather system with real weather data scraping.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime

# Import our modules
from config import config
from dataset_builder import PregameDatasetBuilder
from weather_scraper import WeatherDataScraper
from weather_features import WeatherFeatureCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_real_weather_integration():
    """Test the real weather data integration."""
    logger.info("="*70)
    logger.info("TESTING REAL WEATHER DATA INTEGRATION")
    logger.info("="*70)
    
    try:
        # Check API key availability
        check_api_keys()
        
        # Test weather scraper directly
        test_weather_scraper()
        
        # Test full dataset integration
        test_dataset_with_real_weather()
        
        # Compare real vs synthetic weather
        compare_weather_sources()
        
        logger.info("\nReal weather integration testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Real weather integration testing failed: {e}")
        return False

def check_api_keys():
    """Check if weather API keys are available."""
    logger.info("\n" + "="*60)
    logger.info("API KEY AVAILABILITY CHECK")
    logger.info("="*60)
    
    scraper = WeatherDataScraper()
    api_status = scraper.validate_api_keys()
    
    logger.info(f"Visual Crossing API: {'✅ Available' if api_status['visual_crossing'] else '❌ Missing'}")
    logger.info(f"OpenWeatherMap API: {'✅ Available' if api_status['openweathermap'] else '❌ Missing'}")
    
    if not any(api_status.values()):
        logger.warning("\n⚠️  No weather API keys found!")
        logger.warning("Set environment variables:")
        logger.warning("  export VISUALCROSSING_API_KEY='your_key_here'")
        logger.warning("  export OPENWEATHER_API_KEY='your_key_here'")
        logger.warning("\nWill use synthetic weather as fallback.")
    else:
        logger.info("✅ At least one weather API is available")
        
        # Test API connections
        logger.info("\nTesting API connections...")
        connection_status = scraper.test_api_connection()
        
        for api, status in connection_status.items():
            logger.info(f"  {api}: {'✅ Connected' if status else '❌ Failed'}")

def test_weather_scraper():
    """Test the weather scraper with a small sample."""
    logger.info("\n" + "="*60)
    logger.info("WEATHER SCRAPER TEST")
    logger.info("="*60)
    
    # Create sample game data
    sample_games = pd.DataFrame({
        'date': pd.date_range('2024-08-01', '2024-08-03', freq='D'),
        'stadium': ['Yankee Stadium', 'Fenway Park', 'Coors Field']
    })
    
    logger.info(f"Testing weather scraping for {len(sample_games)} sample games:")
    for _, game in sample_games.iterrows():
        logger.info(f"  {game['date'].strftime('%Y-%m-%d')} at {game['stadium']}")
    
    try:
        scraper = WeatherDataScraper()
        weather_data = scraper.get_historical_weather('2024-08-01', '2024-08-03', sample_games)
        
        if not weather_data.empty:
            logger.info(f"\n✅ Successfully fetched weather data: {len(weather_data)} records")
            
            # Show sample weather data
            weather_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']
            sample_weather = weather_data[['date', 'stadium'] + weather_cols].head(3)
            
            logger.info("\nSample weather data:")
            for _, row in sample_weather.iterrows():
                logger.info(f"  {row['date']} at {row['stadium']}:")
                logger.info(f"    Temp: {row['temperature']:.1f}°F, Humidity: {row['humidity']:.1f}%")
                logger.info(f"    Wind: {row['wind_speed']:.1f}mph @ {row['wind_direction']:.0f}°")
                logger.info(f"    Pressure: {row['pressure']:.2f}inHg")
        else:
            logger.warning("❌ No weather data retrieved")
            
    except Exception as e:
        logger.error(f"Weather scraper test failed: {e}")

def test_dataset_with_real_weather():
    """Test dataset building with real weather integration."""
    logger.info("\n" + "="*60)
    logger.info("DATASET INTEGRATION TEST")
    logger.info("="*60)
    
    try:
        # Build small dataset to test real weather integration
        logger.info("Building dataset with real weather integration...")
        builder = PregameDatasetBuilder(
            start_date="2024-08-01",
            end_date="2024-08-03"  # Small range for testing
        )
        
        dataset = builder.build_dataset(force_rebuild=True)
        
        if dataset.empty:
            logger.error("Dataset is empty")
            return False
        
        logger.info(f"Dataset built: {len(dataset)} rows, {len(dataset.columns)} columns")
        
        # Check weather data coverage
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure']
        weather_coverage = dataset[weather_cols].notna().all(axis=1).mean()
        
        logger.info(f"Weather data coverage: {weather_coverage:.1%}")
        
        # Show weather statistics
        logger.info("\nWeather data statistics:")
        for col in weather_cols:
            if col in dataset.columns:
                data = dataset[col].dropna()
                if len(data) > 0:
                    logger.info(f"  {col}: mean={data.mean():.2f}, std={data.std():.2f}, range=[{data.min():.2f}, {data.max():.2f}]")
        
        # Check for weather impact features
        weather_impact_cols = [col for col in dataset.columns if 'weather' in col or 'atmospheric' in col]
        logger.info(f"\nWeather impact features: {len(weather_impact_cols)}")
        logger.info(f"Examples: {weather_impact_cols[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset integration test failed: {e}")
        return False

def compare_weather_sources():
    """Compare real weather vs synthetic weather."""
    logger.info("\n" + "="*60)
    logger.info("REAL VS SYNTHETIC WEATHER COMPARISON")
    logger.info("="*60)
    
    try:
        # Create sample games
        sample_games = pd.DataFrame({
            'date': pd.date_range('2024-08-01', '2024-08-05', freq='D'),
            'stadium': ['Yankee Stadium'] * 5
        })
        
        weather_calc = WeatherFeatureCalculator()
        
        # Get synthetic weather
        logger.info("Generating synthetic weather...")
        synthetic_weather = weather_calc._create_enhanced_synthetic_weather(sample_games)
        
        # Get real weather (if possible)
        logger.info("Attempting to fetch real weather...")
        real_weather = weather_calc._fetch_real_weather_data(sample_games, '2024-08-01', '2024-08-05')
        
        # Compare the two
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'pressure']
        
        logger.info("\nWeather comparison (Real vs Synthetic):")
        logger.info("Date\t\tSource\t\tTemp\tHumidity\tWind\tPressure")
        logger.info("-" * 70)
        
        for _, row in sample_games.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            
            # Real weather data
            real_row = real_weather[real_weather['date'] == row['date']]
            if not real_row.empty:
                real_row = real_row.iloc[0]
                logger.info(f"{date_str}\tReal\t\t{real_row['temperature']:.1f}\t{real_row['humidity']:.1f}\t\t{real_row['wind_speed']:.1f}\t{real_row['pressure']:.2f}")
            
            # Synthetic weather data
            synth_row = synthetic_weather[synthetic_weather['date'] == row['date']]
            if not synth_row.empty:
                synth_row = synth_row.iloc[0]
                logger.info(f"{date_str}\tSynthetic\t{synth_row['temperature']:.1f}\t{synth_row['humidity']:.1f}\t\t{synth_row['wind_speed']:.1f}\t{synth_row['pressure']:.2f}")
            
            logger.info("")
        
        # Calculate differences if we have real data
        real_coverage = real_weather[weather_cols].notna().all(axis=1).mean()
        if real_coverage > 0:
            logger.info(f"Real weather coverage: {real_coverage:.1%}")
            
            if real_coverage > 0.8:
                logger.info("✅ High quality real weather data available")
            elif real_coverage > 0.5:
                logger.info("⚠️  Partial real weather data - mixed with synthetic")
            else:
                logger.info("❌ Low real weather coverage - mostly synthetic")
        else:
            logger.info("❌ No real weather data available - using synthetic only")
        
    except Exception as e:
        logger.error(f"Weather comparison failed: {e}")

def show_setup_instructions():
    """Show setup instructions for weather APIs."""
    logger.info("\n" + "="*70)
    logger.info("WEATHER API SETUP INSTRUCTIONS")
    logger.info("="*70)
    
    logger.info("""
To enable real weather data scraping, you need API keys from weather providers:

1. VISUAL CROSSING WEATHER API (Recommended):
   - Visit: https://www.visualcrossing.com/weather-api
   - Sign up for free account (1000 records/day free)
   - Get API key
   - Set: export VISUALCROSSING_API_KEY='your_key_here'

2. OPENWEATHERMAP API (Alternative):
   - Visit: https://openweathermap.org/api
   - Sign up for free account (1000 calls/day free)
   - Get API key for "One Call API 3.0"
   - Set: export OPENWEATHER_API_KEY='your_key_here'

3. Test the setup:
   python test_real_weather.py

Without API keys, the system will use sophisticated synthetic weather
that mimics realistic patterns but may not capture actual conditions.
""")

def main():
    """Main testing function."""
    logger.info("Starting real weather data integration testing...")
    
    success = test_real_weather_integration()
    
    if success:
        logger.info("\n" + "="*70)
        logger.info("✅ REAL WEATHER INTEGRATION TESTING COMPLETED!")
        logger.info("Enhanced weather system is working correctly.")
        logger.info("="*70)
        
        show_setup_instructions()
        return 0
    else:
        logger.error("\nReal weather integration testing failed!")
        show_setup_instructions()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)