"""
Production Inference Feature Calculator
======================================

Fast feature calculation for real-time inference using pre-computed matchup database.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from matchup_database import MatchupDatabase
from data_utils import DataValidator

logger = logging.getLogger(__name__)

class InferenceFeatureCalculator:
    """Production-ready feature calculator for fast inference."""
    
    def __init__(self, matchup_db_path: str = "data/matchup_database.db"):
        self.matchup_db = MatchupDatabase(matchup_db_path)
        self.validator = DataValidator()
        logger.info("Initialized inference feature calculator")
    
    def get_game_features(self, batter_id: int, pitcher_id: int, 
                         game_date: str, additional_features: Dict = None) -> Dict[str, float]:
        """
        Get all features for a single batter-pitcher matchup for inference.
        
        Args:
            batter_id: Batter ID
            pitcher_id: Pitcher ID  
            game_date: Game date (for any date-dependent features)
            additional_features: Any additional features to include
        
        Returns:
            Dictionary of all features for this matchup
        """
        # Get matchup features from database
        matchup_features = self.matchup_db.get_matchup_features(batter_id, pitcher_id)
        
        # Add any additional features
        if additional_features:
            matchup_features.update(additional_features)
        
        logger.debug(f"Generated features for batter {batter_id} vs pitcher {pitcher_id}")
        return matchup_features
    
    def get_batch_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get features for a batch of games efficiently.
        
        Args:
            games_df: DataFrame with columns ['batter', 'pitcher', 'date']
        
        Returns:
            DataFrame with original columns plus all matchup features
        """
        logger.info(f"Calculating features for {len(games_df)} games")
        
        # Validate input
        required_cols = ['batter', 'pitcher', 'date']
        self.validator.validate_required_columns(games_df, required_cols, "Inference games")
        
        result_df = games_df.copy()
        
        # Get features for each game
        for idx, row in result_df.iterrows():
            matchup_features = self.matchup_db.get_matchup_features(
                int(row['batter']), 
                int(row['pitcher'])
            )
            
            # Add features to this row
            for feature, value in matchup_features.items():
                result_df.loc[idx, feature] = value
        
        logger.info(f"Added {len(matchup_features)} matchup features to {len(games_df)} games")
        return result_df
    
    def get_todays_features(self, batter_pitcher_pairs: List[Tuple[int, int]]) -> pd.DataFrame:
        """
        Get features for today's games given batter-pitcher pairs.
        
        Args:
            batter_pitcher_pairs: List of (batter_id, pitcher_id) tuples
        
        Returns:
            DataFrame ready for model prediction
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        games_data = []
        for batter_id, pitcher_id in batter_pitcher_pairs:
            features = self.get_game_features(batter_id, pitcher_id, today)
            
            # Add basic game info
            game_row = {
                'batter': batter_id,
                'pitcher': pitcher_id,
                'date': today,
                **features
            }
            games_data.append(game_row)
        
        games_df = pd.DataFrame(games_data)
        logger.info(f"Generated features for {len(games_df)} today's matchups")
        
        return games_df

class MatchupDatabaseBuilder:
    """Builder for creating and updating the matchup database."""
    
    def __init__(self, matchup_db_path: str = "data/matchup_database.db"):
        self.matchup_db = MatchupDatabase(matchup_db_path)
        self.validator = DataValidator()
    
    def build_initial_database(self, start_date: str = "2020-01-01", 
                             end_date: Optional[str] = None) -> Dict[str, int]:
        """
        Build the initial matchup database from historical Statcast data.
        
        Args:
            start_date: Start date for historical data
            end_date: End date (defaults to today)
        
        Returns:
            Statistics about the build process
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Building initial matchup database: {start_date} to {end_date}")
        logger.warning("This will take several minutes for multiple years of data...")
        
        # Import here to avoid circular dependency
        from feature_engineering import StatcastDataProcessor
        from data_utils import CacheManager
        
        # Fetch historical data
        cache_manager = CacheManager()
        processor = StatcastDataProcessor(cache_manager)
        
        logger.info("Fetching historical Statcast data...")
        statcast_data = processor.fetch_statcast_data(start_date, end_date, use_cache=True)
        
        # Build database
        stats = self.matchup_db.bulk_update_matchups(statcast_data)
        
        logger.info(f"Initial database build complete: {stats}")
        return stats
    
    def update_database_incremental(self, new_data_start: str) -> Dict[str, int]:
        """
        Update database with new data since last update.
        
        Args:
            new_data_start: Date to start fetching new data from
        
        Returns:
            Update statistics
        """
        logger.info(f"Updating matchup database with data since {new_data_start}")
        
        # Import here to avoid circular dependency
        from feature_engineering import StatcastDataProcessor
        from data_utils import CacheManager
        
        # Fetch new data
        cache_manager = CacheManager()
        processor = StatcastDataProcessor(cache_manager)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        new_data = processor.fetch_statcast_data(new_data_start, end_date, use_cache=False)
        
        # Update database
        stats = self.matchup_db.bulk_update_matchups(new_data, cutoff_date=new_data_start)
        
        logger.info(f"Database update complete: {stats}")
        return stats
    
    def get_database_info(self) -> Dict:
        """Get information about the current database."""
        return self.matchup_db.get_database_stats()

class ProductionInferenceExample:
    """Example of how to use the system for production inference."""
    
    def __init__(self):
        self.feature_calc = InferenceFeatureCalculator()
        self.db_builder = MatchupDatabaseBuilder()
    
    def predict_todays_games(self, model, batter_pitcher_pairs: List[Tuple[int, int]]) -> pd.DataFrame:
        """
        Complete pipeline for predicting today's games.
        
        Args:
            model: Trained model object
            batter_pitcher_pairs: List of (batter_id, pitcher_id) for today
        
        Returns:
            DataFrame with predictions
        """
        # Get features fast from database
        features_df = self.feature_calc.get_todays_features(batter_pitcher_pairs)
        
        # Make predictions (assuming model has predict_proba method)
        try:
            predictions = model.predict_proba(features_df)
            features_df['hr_probability'] = predictions[:, 1]  # Probability of HR
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            features_df['hr_probability'] = 0.0
        
        return features_df[['batter', 'pitcher', 'hr_probability'] + 
                          [col for col in features_df.columns if 'matchup' in col or 'vs_similar' in col]]
    
    def daily_update_workflow(self):
        """Daily workflow to update database with yesterday's games."""
        yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info("Starting daily database update...")
        
        try:
            # Update database with yesterday's games
            stats = self.db_builder.update_database_incremental(yesterday)
            
            # Log status
            db_info = self.db_builder.get_database_info()
            logger.info(f"Daily update complete. Database now has: {db_info}")
            
            return True
        except Exception as e:
            logger.error(f"Daily update failed: {e}")
            return False

# Export main classes
__all__ = [
    'InferenceFeatureCalculator', 
    'MatchupDatabaseBuilder', 
    'ProductionInferenceExample'
]