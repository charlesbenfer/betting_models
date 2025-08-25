#!/usr/bin/env python3
"""
Fetch Recent MLB Data for Live Predictions
==========================================

This script fetches the last 45 days of MLB data needed for making
predictions on today's games.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from dataset_builder import PregameDatasetBuilder
from config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_recent_data_for_predictions(days_back: int = 45):
    """
    Fetch recent MLB data for live predictions.
    
    Args:
        days_back: Number of days to fetch (default 45)
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"🔄 Fetching MLB data from {start_str} to {end_str}")
        
        # Initialize dataset builder with recent date range
        builder = PregameDatasetBuilder(
            start_date=start_str,
            end_date=end_str
        )
        
        # Build the dataset (this will fetch from pybaseball)
        logger.info("📊 Building pregame dataset with all features...")
        df = builder.build_dataset(force_rebuild=True)  # Force fresh data
        
        if df.empty:
            logger.error("❌ No data retrieved")
            return None
            
        logger.info(f"✅ Retrieved {len(df)} records")
        logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"   Unique players: {df['batter'].nunique()}")
        logger.info(f"   Features: {len(df.columns)}")
        
        # Save to cache for predictions
        cache_dir = Path(config.DATA_DIR) / "processed"
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        filename = f"pregame_dataset_recent_{end_str}.parquet"
        filepath = cache_dir / filename
        
        df.to_parquet(filepath, compression='snappy')
        logger.info(f"💾 Saved to {filepath}")
        
        # Also save a "latest" version for easy access
        latest_path = cache_dir / "pregame_dataset_latest.parquet"
        df.to_parquet(latest_path, compression='snappy')
        logger.info(f"💾 Saved latest version to {latest_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch recent data: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch recent MLB data for predictions')
    parser.add_argument('--days', type=int, default=45,
                       help='Number of days to fetch (default: 45)')
    parser.add_argument('--test', action='store_true',
                       help='Test with just 7 days of data')
    
    args = parser.parse_args()
    
    days = 7 if args.test else args.days
    
    print(f"🚀 Fetching last {days} days of MLB data...")
    print("=" * 60)
    
    df = fetch_recent_data_for_predictions(days_back=days)
    
    if df is not None:
        print("\n✅ Data fetch complete!")
        print(f"   Total records: {len(df):,}")
        print(f"   Ready for live predictions")
    else:
        print("\n❌ Data fetch failed")
        print("   Check logs for details")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())