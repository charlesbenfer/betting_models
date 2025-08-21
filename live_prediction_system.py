"""
Live Prediction System
======================

Efficient system for generating live predictions and finding betting opportunities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from config import config
from api_client import SafeAPIClient
from dataset_builder import EfficientPredictionBuilder
from modeling import DualModelSystem
from betting_utils import BettingAnalyzer, BettingOpportunity
from data_utils import DataValidator, CacheManager

logger = logging.getLogger(__name__)

class LivePredictionSystem:
    """
    Comprehensive system for live home run predictions and betting analysis.
    """
    
    def __init__(self, model_dir: str = None, api_key: str = None):
        self.model_dir = model_dir or str(config.MODEL_DIR)
        
        # Initialize components
        self.model_system = DualModelSystem(self.model_dir)
        self.api_client = SafeAPIClient(api_key)
        self.betting_analyzer = BettingAnalyzer(
            min_ev=config.MIN_EV_THRESHOLD,
            min_probability=config.MIN_PROB_THRESHOLD
        )
        self.cache_manager = CacheManager()
        self.validator = DataValidator()
        
        # State tracking
        self.is_model_loaded = False
        self.last_prediction_date = None
        self.recent_data_cache = None
        
        logger.info("Live prediction system initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the system by loading models and validating APIs.
        
        Returns:
            True if initialization successful
        """
        try:
            # Load trained models
            logger.info("Loading trained models...")
            self.model_system.load()
            self.is_model_loaded = True
            
            # Log model information
            model_info = self.model_system.get_model_info()
            logger.info(f"Model system loaded: {model_info}")
            
            # Check API availability
            odds_available = self.api_client.is_odds_available()
            logger.info(f"Odds API available: {odds_available}")
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def get_todays_predictions(self, target_date: str = None, 
                              force_rebuild: bool = False, 
                              model_name: str = 'rf') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate predictions for a specific date using recent player performance.
        
        Args:
            target_date: Date for predictions (YYYY-MM-DD), defaults to today
            force_rebuild: Force rebuild of dataset
            model_name: Model to use ('rf', 'logistic', 'xgb')
        
        Returns:
            Tuple of (feature_dataframe, probabilities)
        """
        if not self.is_model_loaded:
            raise RuntimeError("Models not loaded. Call initialize() first.")
        
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Generating predictions for {target_date} using recent performance")
        
        try:
            # First, try to get today's odds to see which players we need predictions for
            odds_df = self.api_client.get_todays_odds(target_date)
            
            if odds_df.empty:
                logger.warning(f"No odds available for {target_date}")
                return pd.DataFrame(), np.array([])
            
            logger.info(f"Found odds for {len(odds_df)} players - generating predictions")
            
            # Load recent performance data
            recent_data = self._load_recent_player_performance()
            
            if recent_data.empty:
                logger.warning("No recent performance data available")
                return pd.DataFrame(), np.array([])
            
            # Match today's players with their recent performance
            feature_df = self._match_players_with_recent_performance(odds_df, recent_data)
            
            if feature_df.empty:
                logger.warning(f"No player matches found for {target_date}")
                return pd.DataFrame(), np.array([])
            
            # Generate predictions with specified model
            logger.info(f"Making predictions for {len(feature_df)} players using {model_name}")
            probabilities = self.model_system.predict_proba(
                feature_df, model_name=model_name, prefer_enhanced=True
            )
            
            # Apply calibration if available
            calibration_factor = self._load_calibration_factor()
            if calibration_factor != 1.0:
                probabilities = probabilities * calibration_factor
                logger.info(f"Applied calibration factor: {calibration_factor:.3f}")
            
            # Validate predictions
            if not self.validator.validate_predictions(probabilities):
                if model_name == 'xgb':
                    logger.warning("XGBoost predictions failed, falling back to Random Forest")
                    probabilities = self.model_system.predict_proba(
                        feature_df, model_name='rf', prefer_enhanced=True
                    )
                    if calibration_factor != 1.0:
                        probabilities = probabilities * calibration_factor
                    if not self.validator.validate_predictions(probabilities):
                        raise ValueError("All prediction methods failed validation")
                else:
                    raise ValueError("Generated predictions failed validation")
            
            # Add odds data back to feature_df for merging
            feature_df['odds_hr_yes'] = feature_df.get('odds_hr_yes', np.nan)
            feature_df['odds_hr_no'] = feature_df.get('odds_hr_no', np.nan)
            feature_df['bookmaker'] = feature_df.get('bookmaker', 'Unknown')
            
            logger.info(f"Predictions generated successfully: {len(probabilities)} players")
            return feature_df, probabilities
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            raise
    
    def _load_recent_player_performance(self) -> pd.DataFrame:
        """Load recent player performance data from cache."""
        # Try to load the most recent dataset
        cache_files = [
            "data/processed/pregame_dataset_2025-07-02_2025-08-16.parquet",
            "data/processed/pregame_dataset_2024-01-01_2024-08-01.parquet", 
            "data/processed/pregame_dataset_2023-03-01_2024-10-01.parquet"
        ]
        
        for cache_file in cache_files:
            cache_path = Path(cache_file)
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                
                # Get the most recent performance per player (last 7 games)
                recent_df = (
                    df.sort_values(['batter_name', 'date'])
                    .groupby('batter_name')
                    .tail(7)  # Last 7 games per player
                    .reset_index(drop=True)
                )
                
                logger.info(f"Loaded recent performance for {recent_df['batter_name'].nunique()} players from {cache_file}")
                return recent_df
        
        logger.error("No recent performance data found")
        return pd.DataFrame()
    
    def _match_players_with_recent_performance(self, odds_df: pd.DataFrame, 
                                             recent_data: pd.DataFrame) -> pd.DataFrame:
        """Match today's players with their recent performance."""
        
        # Standardize names for matching
        odds_df['player_std'] = odds_df['batter_name'].apply(self.validator.standardize_name)
        recent_data['player_std'] = recent_data['batter_name'].apply(self.validator.standardize_name)
        
        # For each player with odds today, get their most recent performance
        matched_players = []
        
        for _, odds_row in odds_df.iterrows():
            player_name = odds_row['player_std']
            
            # Find this player's recent performance
            player_recent = recent_data[recent_data['player_std'] == player_name]
            
            if not player_recent.empty:
                # Use most recent game's features
                latest_performance = player_recent.iloc[-1].copy()
                
                # Add odds information
                latest_performance['odds_hr_yes'] = odds_row['odds_hr_yes']
                latest_performance['odds_hr_no'] = odds_row.get('odds_hr_no', np.nan)
                latest_performance['bookmaker'] = odds_row.get('bookmaker', 'Unknown')
                
                matched_players.append(latest_performance)
            else:
                logger.debug(f"No recent performance found for {player_name}")
        
        if matched_players:
            result_df = pd.DataFrame(matched_players)
            logger.info(f"Matched {len(result_df)} players with recent performance")
            return result_df
        else:
            logger.warning("No players matched between odds and recent performance")
            return pd.DataFrame()
    
    def _load_calibration_factor(self) -> float:
        """Load calibration factor from saved data."""
        try:
            import json
            from pathlib import Path
            
            calibration_path = Path("saved_models_pregame/calibration_data.json")
            if calibration_path.exists():
                with open(calibration_path, 'r') as f:
                    data = json.load(f)
                return data.get('calibration_factor', 1.0)
        except Exception as e:
            logger.debug(f"Could not load calibration factor: {e}")
        
        return 1.0  # No calibration if file not found
    
    def get_todays_betting_opportunities(self, target_date: str = None,
                                       min_ev: float = None,
                                       min_prob: float = None) -> List[BettingOpportunity]:
        """
        Get today's betting opportunities with full analysis.
        
        Args:
            target_date: Date for analysis (YYYY-MM-DD)
            min_ev: Minimum expected value threshold
            min_prob: Minimum probability threshold
        
        Returns:
            List of betting opportunities sorted by EV
        """
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        # Update analyzer thresholds if provided
        if min_ev is not None:
            self.betting_analyzer.min_ev = min_ev
        if min_prob is not None:
            self.betting_analyzer.min_probability = min_prob
        
        logger.info(f"Analyzing betting opportunities for {target_date}")
        
        try:
            # Get predictions
            feature_df, probabilities = self.get_todays_predictions(target_date)
            
            if feature_df.empty:
                logger.warning("No prediction data available")
                return []
            
            # Get odds data
            odds_df = self.api_client.get_todays_odds(target_date)
            
            if odds_df.empty:
                logger.warning("No odds data available")
                return []
            
            logger.info(f"Found odds for {len(odds_df)} players")
            
            # Merge predictions with odds
            merged_df = self.betting_analyzer.merge_predictions_with_odds(
                feature_df, probabilities, odds_df
            )
            
            # Identify opportunities
            opportunities = self.betting_analyzer.identify_betting_opportunities(merged_df)
            
            logger.info(f"Found {len(opportunities)} betting opportunities")
            
            # Log summary
            if opportunities:
                best_ev = max(opp.expected_value for opp in opportunities)
                avg_ev = np.mean([opp.expected_value for opp in opportunities])
                logger.info(f"Best EV: {best_ev:.3f}, Average EV: {avg_ev:.3f}")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Betting opportunity analysis failed: {e}")
            return []
    
    def run_full_analysis(self, target_date: str = None, 
                         print_results: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis including predictions, opportunities, and near-misses.
        
        Args:
            target_date: Date for analysis
            print_results: Whether to print formatted results
        
        Returns:
            Dictionary with all analysis results
        """
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Running full analysis for {target_date}")
        
        results = {
            'date': target_date,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'opportunities': [],
            'near_misses': pd.DataFrame(),
            'summary_stats': {},
            'errors': []
        }
        
        try:
            # Get betting opportunities
            opportunities = self.get_todays_betting_opportunities(target_date)
            results['opportunities'] = opportunities
            
            # Get prediction data for near-miss analysis
            feature_df, probabilities = self.get_todays_predictions(target_date)
            odds_df = self.api_client.get_todays_odds(target_date)
            
            if not feature_df.empty and not odds_df.empty:
                # Analyze near-misses
                merged_df = self.betting_analyzer.merge_predictions_with_odds(
                    feature_df, probabilities, odds_df
                )
                
                near_misses = self.betting_analyzer.generate_near_miss_analysis(merged_df)
                results['near_misses'] = near_misses
                
                # Calculate summary statistics
                results['summary_stats'] = self._calculate_summary_stats(
                    merged_df, opportunities
                )
            
            results['success'] = True
            
            # Print results if requested
            if print_results:
                self._print_analysis_results(results)
            
        except Exception as e:
            error_msg = f"Full analysis failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    def _should_rebuild_data(self, target_date: str) -> bool:
        """Check if data should be rebuilt."""
        # Always rebuild if no previous prediction date
        if self.last_prediction_date is None:
            return True
        
        # Rebuild if target date is different
        if self.last_prediction_date != target_date:
            return True
        
        # Rebuild if cache is older than 1 hour
        cache_id = f"predictions_{target_date}"
        if not self.cache_manager.cache_exists(cache_id):
            return True
        
        return False
    
    def _get_cached_prediction_data(self, target_date: str) -> Optional[pd.DataFrame]:
        """Get cached prediction data if available."""
        cache_id = f"predictions_{target_date}"
        return self.cache_manager.load_cache(cache_id)
    
    def _cache_prediction_data(self, target_date: str, data: pd.DataFrame) -> None:
        """Cache prediction data."""
        cache_id = f"predictions_{target_date}"
        self.cache_manager.save_cache(data, cache_id)
    
    def _calculate_summary_stats(self, merged_df: pd.DataFrame, 
                                opportunities: List[BettingOpportunity]) -> Dict[str, Any]:
        """Calculate summary statistics for analysis."""
        has_odds = merged_df['odds_hr_yes'].notna()
        odds_data = merged_df[has_odds]
        
        if odds_data.empty:
            return {}
        
        # Basic stats
        stats = {
            'total_players_with_odds': len(odds_data),
            'total_opportunities': len(opportunities),
            'opportunity_rate': len(opportunities) / len(odds_data) if len(odds_data) > 0 else 0,
            'avg_model_probability': odds_data['model_probability'].mean(),
            'avg_expected_value': odds_data['expected_value'].mean(),
            'positive_ev_count': (odds_data['expected_value'] > 0).sum()
        }
        
        # Opportunity stats
        if opportunities:
            stats.update({
                'best_ev': max(opp.expected_value for opp in opportunities),
                'avg_opportunity_ev': np.mean([opp.expected_value for opp in opportunities]),
                'total_kelly_allocation': sum(opp.kelly_fraction for opp in opportunities),
                'avg_confidence': np.mean([opp.confidence_score for opp in opportunities])
            })
        
        return stats
    
    def _print_analysis_results(self, results: Dict[str, Any]) -> None:
        """Print formatted analysis results."""
        logger.info(f"\n{'='*80}")
        logger.info(f"LIVE PREDICTION ANALYSIS - {results['date']}")
        logger.info(f"{'='*80}")
        
        # Print summary stats
        if results['summary_stats']:
            stats = results['summary_stats']
            logger.info(f"\nSUMMARY STATISTICS:")
            logger.info(f"  Players with odds: {stats.get('total_players_with_odds', 0)}")
            logger.info(f"  Betting opportunities: {stats.get('total_opportunities', 0)}")
            logger.info(f"  Opportunity rate: {stats.get('opportunity_rate', 0):.1%}")
            logger.info(f"  Average model probability: {stats.get('avg_model_probability', 0):.1%}")
            logger.info(f"  Positive EV opportunities: {stats.get('positive_ev_count', 0)}")
        
        # Print opportunities
        if results['opportunities']:
            self.betting_analyzer.print_opportunities_table(results['opportunities'])
        else:
            logger.info("\nNo betting opportunities found meeting criteria")
        
        # Print near-misses
        if not results['near_misses'].empty:
            self.betting_analyzer.print_near_miss_table(results['near_misses'])
        
        # Print any errors
        if results['errors']:
            logger.warning(f"\nERRORS ENCOUNTERED:")
            for error in results['errors']:
                logger.warning(f"  - {error}")

class ScheduledPredictionRunner:
    """
    Automated runner for scheduled predictions and analysis.
    """
    
    def __init__(self, live_system: LivePredictionSystem, 
                 run_times: List[str] = None):
        self.live_system = live_system
        self.run_times = run_times or ["08:00", "12:00", "16:00"]  # Default run times
        self.last_run_date = None
        
    def should_run_now(self) -> bool:
        """Check if a scheduled run should happen now."""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_date = now.strftime("%Y-%m-%d")
        
        # Check if it's a run time and we haven't run today
        if current_time in self.run_times and current_date != self.last_run_date:
            return True
        
        return False
    
    def run_scheduled_analysis(self) -> Dict[str, Any]:
        """Run scheduled analysis if conditions are met."""
        if not self.should_run_now():
            return {'skipped': True, 'reason': 'Not scheduled time or already run today'}
        
        try:
            # Run analysis
            results = self.live_system.run_full_analysis(print_results=True)
            
            # Update last run date
            self.last_run_date = datetime.now().strftime("%Y-%m-%d")
            
            # Add scheduling info
            results['scheduled_run'] = True
            results['run_time'] = datetime.now().strftime("%H:%M")
            
            return results
            
        except Exception as e:
            logger.error(f"Scheduled analysis failed: {e}")
            return {
                'scheduled_run': True,
                'success': False,
                'error': str(e)
            }

# Convenience function for quick setup
def create_live_system(api_key: str = None, model_dir: str = None) -> LivePredictionSystem:
    """
    Create and initialize a live prediction system.
    
    Args:
        api_key: The Odds API key
        model_dir: Directory containing trained models
    
    Returns:
        Initialized LivePredictionSystem
    """
    # Use environment variable if api_key not provided
    if api_key is None:
        api_key = config.THEODDS_API_KEY
    
    # Create system
    system = LivePredictionSystem(model_dir=model_dir, api_key=api_key)
    
    # Initialize
    if system.initialize():
        logger.info("Live prediction system ready")
        return system
    else:
        raise RuntimeError("Failed to initialize live prediction system")

# Export main classes
__all__ = [
    'LivePredictionSystem', 'ScheduledPredictionRunner', 'create_live_system'
]