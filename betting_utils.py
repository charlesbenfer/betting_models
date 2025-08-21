"""
Betting and Odds Utilities - FIXED VERSION
==========================================

Utilities for odds conversion, EV calculation, and betting analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

from data_utils import DataValidator

logger = logging.getLogger(__name__)

@dataclass
class BettingOpportunity:
    """Represents a single betting opportunity."""
    date: pd.Timestamp
    player_name: str
    model_probability: float
    odds_yes: float  # Keep this as odds_yes to match your existing code
    odds_no: Optional[float]
    expected_value: float
    kelly_fraction: float
    bookmaker: str
    fair_odds: float
    confidence_score: float = 0.0

class OddsConverter:
    """Utilities for converting between different odds formats."""
    
    @staticmethod
    def american_to_probability(odds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert American odds to implied probability."""
        odds = np.asarray(odds, dtype=float)
        
        positive_mask = odds >= 0
        probability = np.where(
            positive_mask,
            100.0 / (odds + 100.0),
            -odds / (-odds + 100.0)
        )
        
        return probability if isinstance(odds, np.ndarray) else float(probability)
    
    @staticmethod
    def probability_to_american(prob: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert probability to fair American odds."""
        prob = np.asarray(prob, dtype=float)
        prob = np.clip(prob, 1e-6, 1-1e-6)  # Avoid division by zero
        
        high_prob_mask = prob >= 0.5
        odds = np.where(
            high_prob_mask,
            -np.round((prob / (1.0 - prob)) * 100.0),
            np.round(((1.0 - prob) / prob) * 100.0)
        )
        
        return odds if isinstance(prob, np.ndarray) else float(odds)
    
    @staticmethod
    def decimal_to_american(decimal_odds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert decimal odds to American odds."""
        decimal_odds = np.asarray(decimal_odds, dtype=float)
        
        high_odds_mask = decimal_odds >= 2.0
        american_odds = np.where(
            high_odds_mask,
            np.round((decimal_odds - 1.0) * 100.0),
            np.round(-100.0 / (decimal_odds - 1.0))
        )
        
        return american_odds if isinstance(decimal_odds, np.ndarray) else float(american_odds)
    
    @staticmethod
    def remove_vig_two_way(yes_odds: float, no_odds: float) -> Optional[float]:
        """
        Remove vig from two-way market to get true probability.
        
        Args:
            yes_odds: American odds for "Yes" outcome
            no_odds: American odds for "No" outcome
        
        Returns:
            No-vig probability of "Yes" outcome
        """
        try:
            p_yes = OddsConverter.american_to_probability(yes_odds)
            p_no = OddsConverter.american_to_probability(no_odds)
            
            total_prob = p_yes + p_no
            if total_prob <= 0:
                return None
            
            return p_yes / total_prob
            
        except (ValueError, ZeroDivisionError):
            return None

class EVCalculator:
    """Expected Value calculations for betting."""
    
    @staticmethod
    def calculate_ev(true_probability: float, american_odds: float) -> float:
        """
        Calculate expected value per $1 bet.
        
        Args:
            true_probability: Model's estimated probability of outcome
            american_odds: Bookmaker's American odds
        
        Returns:
            Expected profit per $1 wagered
        """
        true_probability = float(np.clip(true_probability, 1e-6, 1-1e-6))
        american_odds = float(american_odds)
        
        # Calculate payout multiplier
        if american_odds >= 0:
            payout_multiplier = american_odds / 100.0
        else:
            payout_multiplier = 100.0 / (-american_odds)
        
        # EV = (probability of win * payout) - (probability of loss * stake)
        prob_loss = 1.0 - true_probability
        ev = true_probability * payout_multiplier - prob_loss
        
        return ev
    
    @staticmethod
    def calculate_kelly_fraction(true_probability: float, american_odds: float) -> float:
        """
        Calculate Kelly criterion fraction for optimal bet sizing.
        
        Args:
            true_probability: Model's estimated probability
            american_odds: Bookmaker's American odds
        
        Returns:
            Optimal fraction of bankroll to bet (0-1)
        """
        true_probability = float(np.clip(true_probability, 1e-6, 1-1e-6))
        american_odds = float(american_odds)
        
        try:
            # Calculate payout multiplier
            if american_odds >= 0:
                b = american_odds / 100.0
            else:
                b = 100.0 / (-american_odds)
            
            # Kelly formula: f = (bp - q) / b
            # where p = probability of win, q = probability of loss, b = payout multiplier
            q = 1.0 - true_probability
            kelly_fraction = (b * true_probability - q) / b
            
            # Clamp to reasonable bounds
            return float(np.clip(kelly_fraction, 0.0, 1.0))
            
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def find_breakeven_odds(true_probability: float) -> float:
        """Find the American odds needed to break even."""
        return OddsConverter.probability_to_american(true_probability)
    
    @staticmethod
    def calculate_odds_gap(true_probability: float, current_odds: float, 
                          target_ev: float = 0.0) -> Tuple[float, float]:
        """
        Calculate how much odds need to improve to reach target EV.
        
        Returns:
            (gap_in_american_points, needed_odds)
        """
        needed_odds = EVCalculator.find_breakeven_odds(true_probability)
        
        if target_ev != 0:
            # Adjust for non-zero target EV
            # This is an approximation - exact calculation would require iteration
            adjustment = target_ev * 100  # Simple linear adjustment
            needed_odds += adjustment
        
        gap = needed_odds - current_odds
        return gap, needed_odds

class BettingAnalyzer:
    """Analyze betting opportunities and generate recommendations."""
    
    def __init__(self, min_ev: float = 0.05, min_probability: float = 0.06, 
                 max_kelly: float = 0.25):
        self.min_ev = min_ev
        self.min_probability = min_probability
        self.max_kelly = max_kelly
        self.validator = DataValidator()
    
    def standardize_name(self, name):
        """Standardize player names for consistent matching."""
        return DataValidator.standardize_name(name)
    
    def merge_predictions_with_odds(self, predictions_df: pd.DataFrame, 
                                   probabilities: np.ndarray,
                                   odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge model predictions with bookmaker odds.
        
        Args:
            predictions_df: DataFrame with player features and identifiers
            probabilities: Model probability predictions
            odds_df: DataFrame with betting odds
        
        Returns:
            Merged DataFrame with EV calculations
        """
        try:
            logger.info(f"Merging predictions (shape: {predictions_df.shape}) with odds (shape: {odds_df.shape})")
            
            # Check if odds are already in predictions_df (inline merge case)
            if 'odds_hr_yes' in predictions_df.columns and not predictions_df['odds_hr_yes'].isna().all():
                logger.info("Odds already present in predictions DataFrame")
                merged = predictions_df.copy()
                merged['model_probability'] = probabilities
                
                # Calculate EV and other metrics
                has_odds = merged['odds_hr_yes'].notna()
                if has_odds.any():
                    merged = self._calculate_betting_metrics(merged, has_odds)
                
                return merged
            
            # Prepare predictions DataFrame
            pred_df = predictions_df.copy()
            
            # Ensure date column exists and is properly formatted
            if 'date' not in pred_df.columns:
                # If no date column, use today's date
                pred_df['date'] = pd.Timestamp.now().normalize()
            else:
                pred_df['date'] = pd.to_datetime(pred_df['date']).dt.normalize()
            
            pred_df['batter_name'] = pred_df['batter_name'].apply(self.standardize_name)
            pred_df['model_probability'] = probabilities
            
            # Prepare odds DataFrame if provided and not empty
            if odds_df.empty:
                logger.warning("Empty odds DataFrame provided")
                # Return predictions with empty odds columns
                merged = pred_df.copy()
                merged['odds_hr_yes'] = np.nan
                merged['odds_hr_no'] = np.nan
                merged['bookmaker'] = 'Unknown'
                return merged
            
            odds_clean = odds_df.copy()
            
            # Ensure required columns exist in odds_df
            required_odds_cols = ['batter_name']
            missing_cols = [col for col in required_odds_cols if col not in odds_clean.columns]
            if missing_cols:
                logger.error(f"Missing required columns in odds_df: {missing_cols}")
                logger.info(f"Available odds columns: {list(odds_clean.columns)}")
                # Return predictions with empty odds
                merged = pred_df.copy()
                merged['odds_hr_yes'] = np.nan
                merged['odds_hr_no'] = np.nan
                merged['bookmaker'] = 'Unknown'
                return merged
            
            # Standardize odds DataFrame
            if 'date' not in odds_clean.columns:
                odds_clean['date'] = pd.Timestamp.now().normalize()
            else:
                odds_clean['date'] = pd.to_datetime(odds_clean['date']).dt.normalize()
            
            odds_clean['batter_name'] = odds_clean['batter_name'].apply(self.standardize_name)
            
            # Merge on date and player name
            merged = pred_df.merge(
                odds_clean, 
                on=['date', 'batter_name'], 
                how='left',
                suffixes=('', '_odds')
            )
            
            # Handle column name conflicts
            if 'odds_hr_yes_odds' in merged.columns:
                merged['odds_hr_yes'] = merged['odds_hr_yes_odds']
                merged = merged.drop(columns=['odds_hr_yes_odds'])
            
            logger.info(f"Merged DataFrame shape: {merged.shape}")
            
            # Calculate betting metrics where odds are available
            has_odds = merged['odds_hr_yes'].notna()
            logger.info(f"Players with odds: {has_odds.sum()}")
            
            if has_odds.any():
                merged = self._calculate_betting_metrics(merged, has_odds)
            
            return merged
            
        except Exception as e:
            logger.error(f"Failed to merge predictions with odds: {e}")
            logger.info(f"Predictions columns: {list(predictions_df.columns)}")
            logger.info(f"Odds columns: {list(odds_df.columns) if not odds_df.empty else 'Empty odds DataFrame'}")
            
            # Return basic merged DataFrame on error
            merged = predictions_df.copy()
            merged['model_probability'] = probabilities
            if 'odds_hr_yes' not in merged.columns:
                merged['odds_hr_yes'] = np.nan
            if 'odds_hr_no' not in merged.columns:
                merged['odds_hr_no'] = np.nan
            if 'bookmaker' not in merged.columns:
                merged['bookmaker'] = 'Unknown'
            return merged
    
    def _calculate_betting_metrics(self, merged_df: pd.DataFrame, has_odds_mask: pd.Series) -> pd.DataFrame:
        """Calculate betting metrics for rows with valid odds."""
        try:
            # Calculate expected value
            merged_df.loc[has_odds_mask, 'expected_value'] = merged_df.loc[has_odds_mask].apply(
                lambda row: EVCalculator.calculate_ev(row['model_probability'], row['odds_hr_yes']),
                axis=1
            )
            
            # Calculate Kelly fraction
            merged_df.loc[has_odds_mask, 'kelly_fraction'] = merged_df.loc[has_odds_mask].apply(
                lambda row: EVCalculator.calculate_kelly_fraction(row['model_probability'], row['odds_hr_yes']),
                axis=1
            )
            
            # Calculate fair odds
            merged_df.loc[has_odds_mask, 'fair_odds'] = merged_df.loc[has_odds_mask, 'model_probability'].apply(
                OddsConverter.probability_to_american
            )
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Failed to calculate betting metrics: {e}")
            return merged_df
    
    def identify_betting_opportunities(self, merged_df: pd.DataFrame) -> List[BettingOpportunity]:
        """
        Identify positive EV betting opportunities.
        
        Args:
            merged_df: DataFrame with predictions and odds
        
        Returns:
            List of betting opportunities sorted by EV
        """
        try:
            logger.info(f"Identifying betting opportunities from {len(merged_df)} rows")
            
            # Validate required columns
            required_cols = ['model_probability']
            missing_cols = [col for col in required_cols if col not in merged_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Available columns: {list(merged_df.columns)}")
                return []
            
            # Check for odds column with flexible naming
            odds_column = None
            odds_candidates = ['odds_hr_yes', 'odds_yes', 'home_run_yes', 'hr_yes_odds']
            
            for candidate in odds_candidates:
                if candidate in merged_df.columns:
                    odds_column = candidate
                    break
            
            if odds_column is None:
                logger.warning(f"No odds column found. Available columns: {list(merged_df.columns)}")
                return []
            
            logger.info(f"Using odds column: {odds_column}")
            
            # Ensure expected_value column exists
            if 'expected_value' not in merged_df.columns:
                logger.info("Calculating expected value on-the-fly")
                merged_df = merged_df.copy()
                has_odds = merged_df[odds_column].notna()
                merged_df.loc[has_odds, 'expected_value'] = merged_df.loc[has_odds].apply(
                    lambda row: EVCalculator.calculate_ev(row['model_probability'], row[odds_column]),
                    axis=1
                )
            
            # Ensure kelly_fraction column exists
            if 'kelly_fraction' not in merged_df.columns:
                logger.info("Calculating Kelly fraction on-the-fly")
                has_odds = merged_df[odds_column].notna()
                merged_df.loc[has_odds, 'kelly_fraction'] = merged_df.loc[has_odds].apply(
                    lambda row: EVCalculator.calculate_kelly_fraction(row['model_probability'], row[odds_column]),
                    axis=1
                )
            
            # Ensure fair_odds column exists
            if 'fair_odds' not in merged_df.columns:
                logger.info("Calculating fair odds on-the-fly")
                merged_df['fair_odds'] = merged_df['model_probability'].apply(
                    OddsConverter.probability_to_american
                )
            
            # Filter for opportunities meeting criteria
            candidates = merged_df[
                (merged_df[odds_column].notna()) &
                (merged_df['model_probability'] >= self.min_probability) &
                (merged_df['expected_value'] >= self.min_ev)
            ].copy()
            
            logger.info(f"Found {len(candidates)} candidates meeting criteria")
            
            opportunities = []
            
            for _, row in candidates.iterrows():
                try:
                    # Calculate confidence score based on various factors
                    confidence = self._calculate_confidence_score(row)
                    
                    # Cap Kelly fraction at maximum
                    kelly_capped = min(row['kelly_fraction'], self.max_kelly)
                    
                    # Get date - handle missing date column
                    if 'date' in row and pd.notna(row['date']):
                        date = pd.to_datetime(row['date'])
                    else:
                        date = pd.Timestamp.now().normalize()
                    
                    opportunity = BettingOpportunity(
                        date=date,
                        player_name=row['batter_name'],
                        model_probability=row['model_probability'],
                        odds_yes=row[odds_column],  # Use the found odds column
                        odds_no=row.get('odds_hr_no'),
                        expected_value=row['expected_value'],
                        kelly_fraction=kelly_capped,
                        bookmaker=row.get('bookmaker', 'Unknown'),
                        fair_odds=row['fair_odds'],
                        confidence_score=confidence
                    )
                    
                    opportunities.append(opportunity)
                    
                except Exception as e:
                    logger.warning(f"Failed to create opportunity for {row.get('batter_name', 'Unknown')}: {e}")
                    continue
            
            # Sort by EV descending
            opportunities.sort(key=lambda x: x.expected_value, reverse=True)
            
            logger.info(f"Created {len(opportunities)} betting opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to identify betting opportunities: {e}")
            logger.info(f"DataFrame shape: {merged_df.shape}")
            logger.info(f"DataFrame columns: {list(merged_df.columns)}")
            if not merged_df.empty:
                logger.info(f"Sample data:\n{merged_df.head(2)}")
            return []
    
    def _calculate_confidence_score(self, row: pd.Series) -> float:
        """Calculate confidence score for a betting opportunity."""
        score = 0.0
        
        # Base score from EV magnitude
        score += min(row['expected_value'] * 10, 5.0)  # Cap at 5 points
        
        # Bonus for reasonable probability range
        if 0.08 <= row['model_probability'] <= 0.20:
            score += 2.0
        
        # Bonus for recent player performance (if available)
        if row.get('roll10_hr_rate', 0) > 0.10:
            score += 1.0
        
        # Bonus for favorable park factor
        if row.get('park_factor', 1.0) > 1.05:
            score += 1.0
        
        # Penalty for very high Kelly fraction (risky)
        if row['kelly_fraction'] > 0.15:
            score -= 1.0
        
        return max(0.0, min(score, 10.0))  # Clamp to 0-10 range
    
    def generate_near_miss_analysis(self, merged_df: pd.DataFrame, 
                                   top_n: int = 15) -> pd.DataFrame:
        """
        Analyze opportunities that are close to being +EV.
        
        Args:
            merged_df: DataFrame with predictions and odds
            top_n: Number of near-misses to return
        
        Returns:
            DataFrame with near-miss analysis
        """
        try:
            # Check for odds column
            odds_column = None
            odds_candidates = ['odds_hr_yes', 'odds_yes', 'home_run_yes', 'hr_yes_odds']
            
            for candidate in odds_candidates:
                if candidate in merged_df.columns:
                    odds_column = candidate
                    break
            
            if odds_column is None:
                logger.warning("No odds column found for near-miss analysis")
                return pd.DataFrame()
            
            # Get opportunities with odds but not meeting EV threshold
            near_misses = merged_df[
                (merged_df[odds_column].notna()) &
                (merged_df['model_probability'] >= self.min_probability) &
                (merged_df['expected_value'] < self.min_ev)
            ].copy()
            
            if near_misses.empty:
                return pd.DataFrame()
            
            # Calculate what odds are needed for different EV targets
            targets = [0.00, 0.02, 0.05]
            
            for target in targets:
                gap_col = f'gap_ev{int(target*100)}'
                need_col = f'need_ev{int(target*100)}'
                
                results = near_misses.apply(
                    lambda row: EVCalculator.calculate_odds_gap(
                        row['model_probability'], row[odds_column], target
                    ), axis=1
                )
                
                near_misses[gap_col] = [r[0] for r in results]
                near_misses[need_col] = [r[1] for r in results]
            
            # Sort by smallest gap to positive EV
            near_misses = near_misses.sort_values('gap_ev0').head(top_n)
            
            return near_misses
            
        except Exception as e:
            logger.error(f"Near-miss analysis failed: {e}")
            return pd.DataFrame()
    
    def print_opportunities_table(self, opportunities: List[BettingOpportunity], 
                                 max_rows: int = 25) -> None:
        """Print formatted table of betting opportunities."""
        if not opportunities:
            logger.info("No betting opportunities found")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP BETTING OPPORTUNITIES (EV >= {self.min_ev:.1%})")
        logger.info(f"{'='*80}")
        
        # Create DataFrame for pretty printing
        rows = []
        for opp in opportunities[:max_rows]:
            rows.append({
                'Date': opp.date.strftime('%m/%d'),
                'Player': opp.player_name,
                'Model%': f"{opp.model_probability:.1%}",
                'Odds': f"{opp.odds_yes:+.0f}",
                'EV': f"{opp.expected_value:+.3f}",
                'Kelly%': f"{opp.kelly_fraction:.1%}",
                'Fair': f"{opp.fair_odds:+.0f}",
                'Book': opp.bookmaker,
                'Conf': f"{opp.confidence_score:.1f}"
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"\n{df.to_string(index=False)}")
    
    def print_near_miss_table(self, near_misses_df: pd.DataFrame) -> None:
        """Print formatted table of near-miss opportunities."""
        if near_misses_df.empty:
            logger.info("No near-miss opportunities found")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info("NEAR-MISS ANALYSIS (Closest to +EV)")
        logger.info(f"{'='*80}")
        
        # Select columns for display
        display_cols = [
            'date', 'batter_name', 'model_probability', 'odds_hr_yes', 
            'fair_odds', 'need_ev0', 'gap_ev0', 'need_ev2', 'gap_ev2'
        ]
        
        display_df = near_misses_df[
            [col for col in display_cols if col in near_misses_df.columns]
        ].copy()
        
        # Format for display
        if not display_df.empty:
            if 'date' in display_df.columns:
                display_df['date'] = display_df['date'].dt.strftime('%m/%d')
            display_df['model_probability'] = display_df['model_probability'].apply(lambda x: f"{x:.1%}")
            
            # Rename columns for clarity
            display_df = display_df.rename(columns={
                'date': 'Date',
                'batter_name': 'Player',
                'model_probability': 'Model%',
                'odds_hr_yes': 'Current',
                'fair_odds': 'Fair',
                'need_ev0': 'Need(0%)',
                'gap_ev0': 'Gap(0%)',
                'need_ev2': 'Need(2%)',
                'gap_ev2': 'Gap(2%)'
            })
            
            logger.info(f"\n{display_df.to_string(index=False)}")

# Keep your other classes unchanged
class PortfolioManager:
    """Manage a portfolio of betting opportunities."""
    
    def __init__(self, bankroll: float = 1000.0, max_single_bet: float = 0.05,
                 max_daily_risk: float = 0.10):
        self.bankroll = bankroll
        self.max_single_bet = max_single_bet
        self.max_daily_risk = max_daily_risk
        self.bets = []
    
    def evaluate_bet_sizing(self, opportunities: List[BettingOpportunity]) -> pd.DataFrame:
        """Determine optimal bet sizes for a set of opportunities."""
        results = []
        daily_risk = 0.0
        
        for opp in opportunities:
            kelly_bet = opp.kelly_fraction * self.bankroll
            max_bet = self.max_single_bet * self.bankroll
            recommended_bet = min(kelly_bet, max_bet)
            
            bet_risk = recommended_bet
            if daily_risk + bet_risk > self.max_daily_risk * self.bankroll:
                available_risk = self.max_daily_risk * self.bankroll - daily_risk
                recommended_bet = max(0, available_risk)
            
            daily_risk += recommended_bet
            expected_profit = recommended_bet * opp.expected_value
            
            results.append({
                'player': opp.player_name,
                'date': opp.date,
                'model_prob': opp.model_probability,
                'odds': opp.odds_yes,
                'ev': opp.expected_value,
                'kelly_fraction': opp.kelly_fraction,
                'kelly_bet': kelly_bet,
                'recommended_bet': recommended_bet,
                'expected_profit': expected_profit,
                'confidence': opp.confidence_score
            })
            
            if daily_risk >= self.max_daily_risk * self.bankroll:
                break
        
        return pd.DataFrame(results)

class MarketAnalyzer:
    """Analyze betting markets for inefficiencies."""
    
    def __init__(self):
        self.validator = DataValidator()

# Keep all your utility functions and exports as they were
def american_to_prob(odds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Legacy function for American odds to probability conversion."""
    return OddsConverter.american_to_probability(odds)

def prob_to_american(prob: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Legacy function for probability to American odds conversion."""
    return OddsConverter.probability_to_american(prob)

def ev_of_bet(prob: Union[float, np.ndarray], american_odds: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Legacy function for EV calculation."""
    return EVCalculator.calculate_ev(prob, american_odds)

def kelly_fraction(prob: float, american_odds: float) -> float:
    """Legacy function for Kelly fraction calculation."""
    return EVCalculator.calculate_kelly_fraction(prob, american_odds)

def two_way_no_vig_prob(yes_odds: float, no_odds: float) -> Optional[float]:
    """Legacy function for no-vig probability calculation."""
    return OddsConverter.remove_vig_two_way(yes_odds, no_odds)

# Export main classes and functions
__all__ = [
    'BettingOpportunity', 'OddsConverter', 'EVCalculator', 'BettingAnalyzer',
    'PortfolioManager', 'MarketAnalyzer',
    # Legacy functions
    'american_to_prob', 'prob_to_american', 'ev_of_bet', 'kelly_fraction', 'two_way_no_vig_prob'
]