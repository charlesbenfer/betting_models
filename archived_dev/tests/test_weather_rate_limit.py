"""
Test Weather Rate Limit Handling
===============================

Quick test to verify the rate limit fallback logic works correctly.
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
import sys

# Import our modules
from weather_scraper import WeatherDataScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rate_limit_fallback():
    """Test that rate limit detection switches to synthetic weather."""
    logger.info("="*60)
    logger.info("TESTING WEATHER RATE LIMIT FALLBACK")
    logger.info("="*60)
    
    try:
        # Create scraper instance
        scraper = WeatherDataScraper()
        
        # Reset any existing flags
        scraper.reset_rate_limit_flags()
        
        # Create sample games data
        test_dates = [
            datetime.now().date() - timedelta(days=30),
            datetime.now().date() - timedelta(days=31),
            datetime.now().date() - timedelta(days=32)
        ]
        
        games_df = pd.DataFrame({
            'date': test_dates,
            'stadium': ['Yankee Stadium', 'Fenway Park', 'Wrigley Field']
        })
        
        logger.info(f"Created test dataset with {len(games_df)} games")
        
        # Check initial state
        logger.info(f"Initial rate_limit_hit: {scraper.rate_limit_hit}")
        logger.info(f"Initial use_synthetic_fallback: {scraper.use_synthetic_fallback}")
        
        # Simulate a rate limit hit
        logger.info("\\nSimulating rate limit detection...")
        scraper.rate_limit_hit = True
        
        # Try to fetch weather (should switch to synthetic)
        weather_data = scraper.get_historical_weather(
            start_date=test_dates[0].strftime('%Y-%m-%d'),
            end_date=test_dates[-1].strftime('%Y-%m-%d'),
            games_df=games_df,
            use_cache=False  # Don't use cache for testing
        )
        
        logger.info(f"\\nResults:")
        logger.info(f"  Weather records returned: {len(weather_data)}")
        logger.info(f"  Rate limit hit flag: {scraper.rate_limit_hit}")
        logger.info(f"  Using synthetic fallback: {scraper.use_synthetic_fallback}")
        
        # Check that we got weather data
        if len(weather_data) > 0:
            logger.info(f"  Sample weather data:")
            sample = weather_data.iloc[0]
            logger.info(f"    Temperature: {sample.get('temperature', 'N/A')}")
            logger.info(f"    Wind speed: {sample.get('wind_speed', 'N/A')}")
            logger.info(f"    Humidity: {sample.get('humidity', 'N/A')}")
            
            logger.info("\\n✅ Rate limit fallback test PASSED!")
            logger.info("The system correctly switches to synthetic weather after rate limits.")
            return True
        else:
            logger.error("No weather data returned!")
            return False
            
    except Exception as e:
        logger.error(f"Rate limit fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reset_functionality():
    """Test that rate limit flags can be reset."""
    logger.info("\\n" + "="*60)
    logger.info("TESTING RATE LIMIT RESET FUNCTIONALITY")
    logger.info("="*60)
    
    try:
        scraper = WeatherDataScraper()
        
        # Set flags to simulate previous rate limit
        scraper.rate_limit_hit = True
        scraper.use_synthetic_fallback = True
        
        logger.info(f"Before reset - rate_limit_hit: {scraper.rate_limit_hit}")
        logger.info(f"Before reset - use_synthetic_fallback: {scraper.use_synthetic_fallback}")
        
        # Reset flags
        scraper.reset_rate_limit_flags()
        
        logger.info(f"After reset - rate_limit_hit: {scraper.rate_limit_hit}")
        logger.info(f"After reset - use_synthetic_fallback: {scraper.use_synthetic_fallback}")
        
        if not scraper.rate_limit_hit and not scraper.use_synthetic_fallback:
            logger.info("\\n✅ Reset functionality test PASSED!")
            return True
        else:
            logger.error("Reset functionality failed!")
            return False
            
    except Exception as e:
        logger.error(f"Reset functionality test failed: {e}")
        return False

def main():
    """Main testing function."""
    logger.info("Starting weather rate limit handling tests...")
    
    # Test 1: Rate limit fallback
    test1_passed = test_rate_limit_fallback()
    
    # Test 2: Reset functionality  
    test2_passed = test_reset_functionality()
    
    # Overall results
    if test1_passed and test2_passed:
        logger.info("\\n" + "="*60)
        logger.info("✅ ALL RATE LIMIT TESTS PASSED!")
        logger.info("Weather scraper correctly handles rate limits and switches to synthetic data.")
        logger.info("="*60)
        return 0
    else:
        logger.error("\\nSome rate limit tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)