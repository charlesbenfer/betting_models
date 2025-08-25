"""Debug script to see what's in the odds data."""

from api_client import SafeAPIClient
from datetime import datetime

def debug_odds():
    api_client = SafeAPIClient()
    
    target_date = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching odds for {target_date}...")
    
    odds_df = api_client.get_todays_odds(target_date)
    
    if odds_df.empty:
        print("No odds data found")
        return
    
    print(f"Found {len(odds_df)} rows of odds data")
    print(f"Columns: {list(odds_df.columns)}")
    
    # Show first few rows
    print("\nFirst 5 rows:")
    print(odds_df.head().to_string())
    
    # Check for player names
    player_cols = [col for col in odds_df.columns if 'player' in col.lower() or 'name' in col.lower()]
    print(f"\nPlayer-related columns: {player_cols}")
    
    for col in player_cols:
        unique_vals = odds_df[col].dropna().unique()[:10]
        print(f"  {col}: {len(odds_df[col].dropna())} non-null, examples: {unique_vals}")

if __name__ == "__main__":
    debug_odds()