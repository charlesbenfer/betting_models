"""
Prediction Data Builder
======================

Builds prediction datasets from historical data for live predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging

from config import config
from dataset_builder import PregameDatasetBuilder

logger = logging.getLogger(__name__)

class PredictionDataBuilder:
    """
    Builds prediction-ready datasets from historical data.
    """
    
    def __init__(self):
        self.data_dir = Path(config.DATA_DIR) / "processed"
        self.pregame_builder = PregameDatasetBuilder()
    
    def get_recent_prediction_data(self, days_back: int = 45, end_date: str = None) -> pd.DataFrame:
        """
        Get recent historical data formatted for predictions.
        
        Args:
            days_back: Number of days to look back
            end_date: End date (defaults to most recent available date)
            
        Returns:
            DataFrame ready for model predictions
        """
        try:
            # Find all available pregame datasets
            pregame_files = list(self.data_dir.glob("pregame_dataset_*.parquet"))
            
            if not pregame_files:
                logger.warning("No pregame dataset files found")
                return pd.DataFrame()
            
            logger.info(f"Found {len(pregame_files)} pregame dataset files")
            
            # Load and combine all data
            all_data = []
            for file_path in pregame_files:
                try:
                    df = pd.read_parquet(file_path)
                    all_data.append(df)
                    logger.info(f"Loaded {len(df)} records from {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path.name}: {e}")
            
            if not all_data:
                logger.error("No data could be loaded")
                return pd.DataFrame()
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined dataset: {len(combined_df)} total records")
            
            # Convert date column to datetime
            if 'date' in combined_df.columns:
                combined_df['date'] = pd.to_datetime(combined_df['date'])
            else:
                logger.error("No 'date' column found in data")
                return pd.DataFrame()
            
            # Determine date range
            if end_date is None:
                end_date = combined_df['date'].max()
            else:
                end_date = pd.to_datetime(end_date)
            
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Filtering data from {start_date.date()} to {end_date.date()}")
            
            # Filter to recent data
            recent_data = combined_df[
                (combined_df['date'] >= start_date) & 
                (combined_df['date'] <= end_date)
            ].copy()
            
            logger.info(f"Recent dataset: {len(recent_data)} records")
            
            if recent_data.empty:
                logger.warning("No recent data found in specified range")
                return pd.DataFrame()
            
            # Sort by date (most recent first)
            recent_data = recent_data.sort_values('date', ascending=False)
            
            # Add some derived features that might be missing
            recent_data = self._add_prediction_features(recent_data)
            
            return recent_data
            
        except Exception as e:
            logger.error(f"Failed to build recent prediction data: {e}")
            return pd.DataFrame()
    
    def _add_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional features useful for predictions."""
        try:
            # Ensure we have the core features
            feature_columns = [
                'recent_batting_avg', 'recent_on_base_pct', 'recent_slugging_pct',
                'recent_home_runs', 'recent_at_bats', 'recent_hits',
                'pitcher_era', 'pitcher_whip', 'pitcher_k9', 'pitcher_bb9', 'pitcher_hr9',
                'temperature', 'humidity', 'wind_speed'
            ]
            
            # Fill missing values with reasonable defaults
            for col in feature_columns:
                if col in df.columns:
                    if col in ['recent_batting_avg', 'recent_on_base_pct', 'recent_slugging_pct']:
                        df[col] = df[col].fillna(0.250)  # League average-ish
                    elif col.startswith('pitcher_'):
                        if col == 'pitcher_era':
                            df[col] = df[col].fillna(4.50)  # League average ERA
                        elif col == 'pitcher_whip':
                            df[col] = df[col].fillna(1.30)  # League average WHIP
                        else:
                            df[col] = df[col].fillna(df[col].median())
                    elif col in ['temperature', 'humidity', 'wind_speed']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(0)
                else:
                    # Create missing columns with default values
                    if col in ['recent_batting_avg', 'recent_on_base_pct']:
                        df[col] = 0.250
                    elif col == 'recent_slugging_pct':
                        df[col] = 0.400
                    else:
                        df[col] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to add prediction features: {e}")
            return df
    
    def get_player_recent_form(self, player_name: str, days_back: int = 30) -> Dict[str, float]:
        """Get a specific player's recent form data."""
        try:
            recent_data = self.get_recent_prediction_data(days_back)
            
            if recent_data.empty:
                return {}
            
            # Filter to specific player
            player_data = recent_data[
                recent_data['batter_name'].str.contains(player_name, case=False, na=False)
            ]
            
            if player_data.empty:
                logger.warning(f"No recent data found for player: {player_name}")
                return {}
            
            # Calculate recent averages
            recent_form = {
                'games_played': len(player_data),
                'avg_batting_avg': player_data['recent_batting_avg'].mean(),
                'avg_on_base_pct': player_data['recent_on_base_pct'].mean(),
                'avg_slugging_pct': player_data['recent_slugging_pct'].mean(),
                'total_home_runs': player_data['recent_home_runs'].sum(),
                'total_at_bats': player_data['recent_at_bats'].sum(),
                'home_run_rate': (player_data['recent_home_runs'].sum() / 
                                max(player_data['recent_at_bats'].sum(), 1))
            }
            
            return recent_form
            
        except Exception as e:
            logger.error(f"Failed to get player recent form: {e}")
            return {}
    
    def build_todays_prediction_dataset(self, target_players: List[str], 
                                      target_date: str = None) -> pd.DataFrame:
        """
        Build a prediction dataset for specific players based on recent form.
        
        Args:
            target_players: List of player names to predict for
            target_date: Target date for predictions
            
        Returns:
            DataFrame ready for model predictions
        """
        try:
            # Get recent historical data for context
            recent_data = self.get_recent_prediction_data(days_back=45)
            
            if recent_data.empty:
                logger.error("No recent data available for building predictions")
                return pd.DataFrame()
            
            prediction_rows = []
            
            for player_name in target_players:
                try:
                    # Get player's most recent form
                    player_recent = recent_data[
                        recent_data['batter_name'].str.contains(player_name, case=False, na=False)
                    ]
                    
                    if player_recent.empty:
                        logger.warning(f"No recent data for {player_name}")
                        continue
                    
                    # Use most recent game as template
                    template_row = player_recent.iloc[0].copy()
                    
                    # Update date to target date
                    if target_date:
                        template_row['date'] = pd.to_datetime(target_date)
                    else:
                        template_row['date'] = pd.to_datetime('today')
                    
                    # Calculate recent averages for key features
                    recent_games = player_recent.head(10)  # Last 10 games
                    
                    template_row['recent_batting_avg'] = recent_games['recent_batting_avg'].mean()
                    template_row['recent_on_base_pct'] = recent_games['recent_on_base_pct'].mean()
                    template_row['recent_slugging_pct'] = recent_games['recent_slugging_pct'].mean()
                    template_row['recent_home_runs'] = recent_games['recent_home_runs'].mean()
                    template_row['recent_at_bats'] = recent_games['recent_at_bats'].mean()
                    
                    prediction_rows.append(template_row)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {player_name}: {e}")
                    continue
            
            if not prediction_rows:
                logger.error("No prediction rows could be created")
                return pd.DataFrame()
            
            prediction_df = pd.DataFrame(prediction_rows)
            logger.info(f"Built prediction dataset: {len(prediction_df)} players")
            
            return prediction_df
            
        except Exception as e:
            logger.error(f"Failed to build today's prediction dataset: {e}")
            return pd.DataFrame()


def test_prediction_data_builder():
    """Test the prediction data builder."""
    try:
        print("🧪 Testing Prediction Data Builder...")
        
        builder = PredictionDataBuilder()
        
        # Test getting recent data
        print("📊 Getting recent prediction data...")
        recent_data = builder.get_recent_prediction_data(days_back=30)
        
        if not recent_data.empty:
            print(f"✅ Recent data: {len(recent_data)} records")
            print(f"   Date range: {recent_data['date'].min()} to {recent_data['date'].max()}")
            print(f"   Unique players: {recent_data['batter_name'].nunique()}")
            print(f"   Unique teams: {recent_data['home_team'].nunique()}")
            
            # Test player form
            sample_players = recent_data['batter_name'].dropna().unique()[:5]
            print(f"\n👤 Testing player form for {len(sample_players)} players...")
            
            for player in sample_players:
                form = builder.get_player_recent_form(player, days_back=30)
                if form:
                    print(f"   {player}: {form['games_played']} games, "
                          f"{form['avg_batting_avg']:.3f} BA, {form['total_home_runs']} HRs")
            
            # Test building prediction dataset
            print(f"\n🔮 Building prediction dataset...")
            pred_data = builder.build_todays_prediction_dataset(sample_players[:3])
            
            if not pred_data.empty:
                print(f"✅ Prediction dataset: {len(pred_data)} players ready for prediction")
            else:
                print("⚠️  No prediction data created")
        else:
            print("❌ No recent data found")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_prediction_data_builder()