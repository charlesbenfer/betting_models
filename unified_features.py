"""
Unified Feature System
=====================

Integrates the fast matchup database with the training pipeline for unified
feature calculation across training and inference.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

from enhanced_features import BatterPitcherMatchupCalculator
from inference_features import InferenceFeatureCalculator, MatchupDatabaseBuilder
from data_utils import DataValidator

logger = logging.getLogger(__name__)

class UnifiedMatchupCalculator:
    """
    Unified matchup calculator that can use either:
    1. Fast database lookups (for inference and recent training data)
    2. Full rebuild (for historical training data or when database is missing)
    """
    
    def __init__(self, db_path: str = "data/matchup_database.db", 
                 auto_build_db: bool = True):
        self.db_path = Path(db_path)
        self.validator = DataValidator()
        self.auto_build_db = auto_build_db
        
        # Initialize both systems
        self.inference_calc = None
        self.legacy_calc = BatterPitcherMatchupCalculator()
        
        # Check if database exists
        if self.db_path.exists():
            try:
                self.inference_calc = InferenceFeatureCalculator(str(db_path))
                logger.info("Using fast database for matchup features")
            except Exception as e:
                logger.warning(f"Database exists but failed to initialize: {e}")
                self.inference_calc = None
        else:
            logger.info("No matchup database found")
            if auto_build_db:
                logger.info("Auto-build is enabled - will create database when needed")
    
    def calculate_matchup_features(self, statcast_df: pd.DataFrame, 
                                 batter_games_df: pd.DataFrame,
                                 use_database: Optional[bool] = None,
                                 force_rebuild_db: bool = False) -> pd.DataFrame:
        """
        Calculate matchup features using the best available method.
        
        Args:
            statcast_df: Historical Statcast data
            batter_games_df: Games to calculate features for
            use_database: Force use of database (True) or legacy (False). 
                         If None, auto-decide.
            force_rebuild_db: Force rebuild of database from statcast_df
        
        Returns:
            DataFrame with matchup features added
        """
        # Decide which method to use
        should_use_db = self._should_use_database(
            statcast_df, batter_games_df, use_database, force_rebuild_db
        )
        
        if should_use_db:
            return self._calculate_with_database(statcast_df, batter_games_df, force_rebuild_db)
        else:
            return self._calculate_with_legacy(statcast_df, batter_games_df)
    
    def _should_use_database(self, statcast_df: pd.DataFrame, 
                           batter_games_df: pd.DataFrame,
                           use_database: Optional[bool],
                           force_rebuild_db: bool) -> bool:
        """Decide whether to use database or legacy calculation."""
        
        # If explicitly specified, use that
        if use_database is not None:
            return use_database
        
        # If forcing rebuild, use database approach
        if force_rebuild_db:
            return True
        
        # If no database exists and auto-build is disabled, use legacy
        if not self.db_path.exists() and not self.auto_build_db:
            logger.info("No database and auto-build disabled - using legacy calculation")
            return False
        
        # For small datasets, legacy is fine
        if len(statcast_df) < 100000:  # Less than 100k rows
            logger.info("Small dataset - using legacy calculation for simplicity")
            return False
        
        # For large datasets or if database exists, use database
        logger.info("Large dataset or database available - using database approach")
        return True
    
    def _calculate_with_database(self, statcast_df: pd.DataFrame, 
                               batter_games_df: pd.DataFrame,
                               force_rebuild: bool) -> pd.DataFrame:
        """Calculate features using the database approach."""
        logger.info("Calculating matchup features using database approach")
        
        # Build/update database if needed
        if force_rebuild or not self.db_path.exists():
            logger.info("Building/updating matchup database...")
            builder = MatchupDatabaseBuilder(str(self.db_path))
            stats = builder.matchup_db.bulk_update_matchups(statcast_df)
            logger.info(f"Database update complete: {stats}")
        
        # Initialize inference calculator if not already done
        if self.inference_calc is None:
            self.inference_calc = InferenceFeatureCalculator(str(self.db_path))
        
        # Use fast database lookups
        result_df = batter_games_df.copy()
        
        # Get required columns
        required_cols = ['batter', 'opp_starter', 'date']
        self.validator.validate_required_columns(result_df, required_cols, "Games for database lookup")
        
        # Process each game
        for idx, row in result_df.iterrows():
            batter_id = row['batter']
            pitcher_id = row['opp_starter']
            
            if pd.isna(pitcher_id):
                # No pitcher - use defaults
                features = self._get_default_features()
            else:
                # Fast database lookup
                features = self.inference_calc.matchup_db.get_matchup_features(
                    int(batter_id), int(pitcher_id)
                )
            
            # Add features to row
            for feature, value in features.items():
                result_df.loc[idx, feature] = value
        
        logger.info("Database-based feature calculation complete")
        return result_df
    
    def _calculate_with_legacy(self, statcast_df: pd.DataFrame, 
                             batter_games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features using the legacy approach."""
        logger.info("Calculating matchup features using legacy approach")
        
        # Use the original calculator
        result_df = self.legacy_calc.calculate_matchup_features(statcast_df, batter_games_df)
        
        # Also calculate similarity features
        result_df = self.legacy_calc.calculate_pitcher_similarity_features(statcast_df, result_df)
        
        logger.info("Legacy feature calculation complete")
        return result_df
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default features when no data is available."""
        return {
            'matchup_pa_career': 0.0,
            'matchup_hr_career': 0.0,
            'matchup_hr_rate_career': 0.0,
            'matchup_avg_ev_career': 0.0,
            'matchup_avg_la_career': 0.0,
            'matchup_pa_recent': 0.0,
            'matchup_hr_recent': 0.0,
            'matchup_hr_rate_recent': 0.0,
            'matchup_days_since_last': 999.0,
            'matchup_encounters_last_year': 0.0,
            'matchup_familiarity_score': 0.0,
            'vs_similar_hand_pa': 0.0,
            'vs_similar_hand_hr': 0.0,
            'vs_similar_hand_hr_rate': 0.0,
            'vs_similar_velocity_pa': 0.0,
            'vs_similar_velocity_hr': 0.0,
            'vs_similar_velocity_hr_rate': 0.0
        }
    
    def ensure_database_ready(self, start_date: str = "2020-01-01") -> Dict[str, int]:
        """
        Ensure the matchup database is ready for production use.
        
        Args:
            start_date: Start date for historical data to build database
        
        Returns:
            Database build/update statistics
        """
        if self.db_path.exists():
            # Database exists - check if it needs updating
            builder = MatchupDatabaseBuilder(str(self.db_path))
            db_info = builder.get_database_info()
            
            logger.info(f"Existing database found: {db_info}")
            
            # Check if database is recent enough (last week)
            if db_info.get('last_update'):
                import dateutil.parser
                last_update = dateutil.parser.parse(db_info['last_update'])
                days_old = (pd.Timestamp.now() - last_update).days
                
                if days_old > 7:
                    logger.info(f"Database is {days_old} days old - updating...")
                    return builder.update_database_incremental(
                        (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                    )
                else:
                    logger.info("Database is recent - no update needed")
                    return db_info
            else:
                logger.warning("Database has no update metadata - consider rebuilding")
                return db_info
        else:
            # No database - build from scratch
            logger.info("No database found - building from scratch...")
            builder = MatchupDatabaseBuilder(str(self.db_path))
            return builder.build_initial_database(start_date)
    
    def get_inference_calculator(self) -> InferenceFeatureCalculator:
        """Get the inference calculator for production use."""
        if self.inference_calc is None:
            if not self.db_path.exists():
                raise ValueError(
                    "No matchup database available. "
                    "Call ensure_database_ready() first or enable auto_build_db."
                )
            self.inference_calc = InferenceFeatureCalculator(str(self.db_path))
        
        return self.inference_calc

class ProductionDeploymentHelper:
    """Helper class for deploying the unified system to production."""
    
    @staticmethod
    def setup_production_environment(db_path: str = "data/matchup_database.db",
                                   historical_start: str = "2020-01-01") -> Dict[str, str]:
        """
        Set up a production environment with the matchup database.
        
        Args:
            db_path: Path for the matchup database
            historical_start: Start date for historical data
        
        Returns:
            Setup status and information
        """
        logger.info("Setting up production environment...")
        
        # Initialize unified calculator
        calc = UnifiedMatchupCalculator(db_path, auto_build_db=True)
        
        # Ensure database is ready
        try:
            stats = calc.ensure_database_ready(historical_start)
            
            # Test inference
            inference_calc = calc.get_inference_calculator()
            test_features = inference_calc.get_game_features(123456, 654321, "2024-01-01")
            
            return {
                'status': 'success',
                'database_path': db_path,
                'database_stats': str(stats),
                'inference_test': 'passed',
                'ready_for_production': 'true'
            }
            
        except Exception as e:
            logger.error(f"Production setup failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'ready_for_production': 'false'
            }
    
    @staticmethod
    def daily_maintenance_workflow(db_path: str = "data/matchup_database.db") -> bool:
        """
        Daily maintenance workflow to update the database.
        
        Args:
            db_path: Path to the matchup database
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Running daily maintenance workflow...")
        
        try:
            # Update database with recent data
            builder = MatchupDatabaseBuilder(db_path)
            yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            stats = builder.update_database_incremental(yesterday)
            logger.info(f"Daily update complete: {stats}")
            
            # Verify database health
            db_info = builder.get_database_info()
            logger.info(f"Database health check: {db_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"Daily maintenance failed: {e}")
            return False

# Export main classes
__all__ = [
    'UnifiedMatchupCalculator',
    'ProductionDeploymentHelper'
]