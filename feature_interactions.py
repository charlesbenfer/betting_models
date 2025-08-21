"""
Feature Interaction Terms (Step 8)
==================================

Advanced feature interactions and combinations including:
- Multiplicative interactions between key features
- Conditional feature effects
- Composite performance indices
- Cross-domain feature synergies
- Non-linear transformations
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class InteractionGroup:
    """Definition of a feature interaction group."""
    name: str
    features: List[str]
    interaction_type: str  # 'multiply', 'add', 'ratio', 'composite'
    description: str

class FeatureInteractionCalculator:
    """Calculate feature interaction terms and composite indices."""
    
    def __init__(self):
        self.interaction_groups = self._define_interaction_groups()
        
    def calculate_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive feature interactions.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with added interaction features
        """
        logger.info("Calculating feature interaction terms...")
        
        df_copy = df.copy()
        
        # Calculate different types of interactions
        df_copy = self._calculate_multiplicative_interactions(df_copy)
        df_copy = self._calculate_conditional_interactions(df_copy)
        df_copy = self._calculate_composite_indices(df_copy)
        df_copy = self._calculate_ratio_interactions(df_copy)
        df_copy = self._calculate_threshold_interactions(df_copy)
        df_copy = self._calculate_cross_domain_synergies(df_copy)
        df_copy = self._calculate_performance_multipliers(df_copy)
        
        logger.info("Feature interaction calculation completed")
        return df_copy
    
    def _define_interaction_groups(self) -> List[InteractionGroup]:
        """Define groups of features that should interact."""
        return [
            InteractionGroup(
                name="power_form_park",
                features=["power_form_hr_rate", "park_elevation_carry_boost"],
                interaction_type="multiply",
                description="Power form enhanced by ballpark carry"
            ),
            InteractionGroup(
                name="hot_streak_confidence",
                features=["hot_streak_intensity", "confidence_indicator"],
                interaction_type="multiply",
                description="Hot streak amplified by confidence"
            ),
            InteractionGroup(
                name="weather_park_synergy",
                features=["temp_hr_factor", "park_elevation"],
                interaction_type="multiply",
                description="Temperature effects amplified at altitude"
            ),
            InteractionGroup(
                name="fatigue_performance",
                features=["energy_reserves", "circadian_performance_factor"],
                interaction_type="multiply",
                description="Energy and circadian rhythm interaction"
            ),
            InteractionGroup(
                name="clutch_momentum",
                features=["clutch_hr_rate", "momentum_strength"],
                interaction_type="multiply",
                description="Clutch performance enhanced by momentum"
            )
        ]
    
    def _calculate_multiplicative_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiplicative feature interactions."""
        
        # Power × Environment interactions
        if 'power_form_hr_rate' in df.columns and 'park_elevation_carry_boost' in df.columns:
            df['power_form_altitude_boost'] = (
                df['power_form_hr_rate'] * (1 + df['park_elevation_carry_boost'])
            )
        
        if 'power_form_avg_ev' in df.columns and 'weather_favorability_index' in df.columns:
            df['power_weather_synergy'] = (
                df['power_form_avg_ev'] * df['weather_favorability_index']
            )
        
        # Streak × Confidence interactions
        if 'hot_streak_intensity' in df.columns and 'confidence_indicator' in df.columns:
            df['hot_streak_confidence_boost'] = (
                df['hot_streak_intensity'] * df['confidence_indicator']
            )
        
        if 'momentum_strength' in df.columns and 'mental_toughness' in df.columns:
            df['momentum_toughness_factor'] = (
                df['momentum_strength'] * df['mental_toughness']
            )
        
        # Fatigue × Performance interactions
        if 'energy_reserves' in df.columns and 'circadian_performance_factor' in df.columns:
            df['energy_circadian_factor'] = (
                df['energy_reserves'] * df['circadian_performance_factor']
            )
        
        if 'rest_quality_score' in df.columns and 'momentum_strength' in df.columns:
            df['rested_momentum_boost'] = (
                df['rest_quality_score'] * df['momentum_strength']
            )
        
        # Matchup × Form interactions
        if 'matchup_hr_rate_career' in df.columns and 'power_form_hr_rate' in df.columns:
            df['matchup_form_synergy'] = (
                df['matchup_hr_rate_career'] * df['power_form_hr_rate']
            )
        
        # Park × Weather interactions
        if 'park_wind_interaction' in df.columns and 'wind_hr_factor' in df.columns:
            df['park_wind_amplification'] = (
                df['park_wind_interaction'] * df['wind_hr_factor']
            )
        
        if 'park_temperature_interaction' in df.columns and 'temp_hr_factor' in df.columns:
            df['park_temp_amplification'] = (
                df['park_temperature_interaction'] * df['temp_hr_factor']
            )
        
        # Situational × Psychological interactions
        if 'clutch_hr_rate' in df.columns and 'pressure_response' in df.columns:
            df['clutch_pressure_performance'] = (
                df['clutch_hr_rate'] * df['pressure_response']
            )
        
        return df
    
    def _calculate_conditional_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate conditional feature effects (if-then relationships)."""
        
        # Hot streak conditional effects
        if 'current_hot_streak' in df.columns and 'power_form_hr_rate' in df.columns:
            hot_streak_mask = df['current_hot_streak'] >= 3
            df['hot_streak_power_boost'] = np.where(
                hot_streak_mask, 
                df['power_form_hr_rate'] * 1.15,  # 15% boost during hot streaks
                df['power_form_hr_rate']
            )
        
        # Cold streak penalties
        if 'current_cold_streak' in df.columns and 'confidence_indicator' in df.columns:
            cold_streak_mask = df['current_cold_streak'] >= 4
            df['cold_streak_confidence_penalty'] = np.where(
                cold_streak_mask,
                df['confidence_indicator'] * 0.85,  # 15% penalty during cold streaks
                df['confidence_indicator']
            )
        
        # High fatigue conditional effects
        if 'fatigue_level' in df.columns and 'power_momentum_7d' in df.columns:
            high_fatigue_mask = df['fatigue_level'] > 0.6
            df['fatigue_momentum_penalty'] = np.where(
                high_fatigue_mask,
                df['power_momentum_7d'] * 0.8,  # 20% penalty when highly fatigued
                df['power_momentum_7d']
            )
        
        # Travel fatigue effects
        if 'jet_lag_factor' in df.columns and 'circadian_performance_factor' in df.columns:
            jet_lag_mask = df['jet_lag_factor'] > 0.02
            df['jet_lag_circadian_disruption'] = np.where(
                jet_lag_mask,
                df['circadian_performance_factor'] * (1 - df['jet_lag_factor']),
                df['circadian_performance_factor']
            )
        
        # Park advantage conditional effects
        if 'batter_park_hr_rate_boost' in df.columns and 'park_pull_factor_left' in df.columns:
            # Assume left-handed batters (simplified - would need actual handedness data)
            park_advantage_mask = df['batter_park_hr_rate_boost'] > 0.05
            df['park_advantage_pull_boost'] = np.where(
                park_advantage_mask,
                df['park_pull_factor_left'] * 1.2,  # 20% boost at favorable parks
                df['park_pull_factor_left']
            )
        
        return df
    
    def _calculate_composite_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite performance indices."""
        
        # Overall power index
        power_components = [
            'power_form_hr_rate', 'power_form_avg_ev', 'power_form_hard_hit_rate'
        ]
        available_power = [col for col in power_components if col in df.columns]
        if len(available_power) >= 2:
            power_scores = []
            for col in available_power:
                # Normalize to 0-1 scale
                col_norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
                power_scores.append(col_norm)
            df['composite_power_index'] = np.mean(power_scores, axis=0)
        
        # Overall momentum index
        momentum_components = [
            'momentum_strength', 'trend_acceleration', 'breakout_velocity'
        ]
        available_momentum = [col for col in momentum_components if col in df.columns]
        if len(available_momentum) >= 2:
            momentum_scores = []
            for col in available_momentum:
                col_norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
                momentum_scores.append(col_norm)
            df['composite_momentum_index'] = np.mean(momentum_scores, axis=0)
        
        # Environmental favorability index
        env_components = [
            'park_elevation_carry_boost', 'temp_hr_factor', 'wind_hr_factor'
        ]
        available_env = [col for col in env_components if col in df.columns]
        if len(available_env) >= 2:
            env_scores = []
            for col in available_env:
                # Center around 1.0 (neutral)
                col_centered = df[col] 
                env_scores.append(col_centered)
            df['environmental_favorability_index'] = np.mean(env_scores, axis=0)
        
        # Physical condition index
        condition_components = [
            'energy_reserves', 'rest_quality_score', 'recovery_status'
        ]
        available_condition = [col for col in condition_components if col in df.columns]
        if len(available_condition) >= 2:
            condition_scores = []
            for col in available_condition:
                condition_scores.append(df[col])
            df['physical_condition_index'] = np.mean(condition_scores, axis=0)
        
        # Psychological state index
        psych_components = [
            'confidence_indicator', 'mental_toughness', 'pressure_response'
        ]
        available_psych = [col for col in psych_components if col in df.columns]
        if len(available_psych) >= 2:
            psych_scores = []
            for col in available_psych:
                psych_scores.append(df[col])
            df['psychological_state_index'] = np.mean(psych_scores, axis=0)
        
        return df
    
    def _calculate_ratio_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ratio-based feature interactions."""
        
        # Power-to-contact ratio
        if 'power_form_hr_rate' in df.columns and 'contact_form_consistency' in df.columns:
            df['power_contact_ratio'] = (
                df['power_form_hr_rate'] / (df['contact_form_consistency'] + 0.01)
            )
        
        # Momentum-to-fatigue ratio
        if 'momentum_strength' in df.columns and 'fatigue_level' in df.columns:
            df['momentum_fatigue_ratio'] = (
                df['momentum_strength'] / (df['fatigue_level'] + 0.01)
            )
        
        # Rest-to-workload ratio
        if 'rest_quality_score' in df.columns and 'schedule_intensity' in df.columns:
            df['rest_workload_ratio'] = (
                df['rest_quality_score'] / (df['schedule_intensity'] + 0.01)
            )
        
        # Performance-to-pressure ratio
        if 'power_form_hr_rate' in df.columns and 'pressure_performance_index' in df.columns:
            df['performance_pressure_ratio'] = (
                df['power_form_hr_rate'] / (1 / (df['pressure_performance_index'] + 0.01))
            )
        
        # Hot-to-cold streak ratio
        if 'hot_streak_intensity' in df.columns and 'cold_streak_depth' in df.columns:
            df['hot_cold_balance'] = (
                df['hot_streak_intensity'] / (df['cold_streak_depth'] + 0.01)
            )
        
        return df
    
    def _calculate_threshold_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate threshold-based interactions (step functions)."""
        
        # Elite power threshold
        if 'power_form_hr_rate' in df.columns:
            elite_power_threshold = df['power_form_hr_rate'].quantile(0.8)
            df['elite_power_indicator'] = (df['power_form_hr_rate'] > elite_power_threshold).astype(int)
        
        # High momentum threshold
        if 'momentum_strength' in df.columns:
            high_momentum_threshold = df['momentum_strength'].quantile(0.75)
            df['high_momentum_indicator'] = (df['momentum_strength'] > high_momentum_threshold).astype(int)
        
        # Extreme fatigue threshold
        if 'fatigue_level' in df.columns:
            extreme_fatigue_threshold = df['fatigue_level'].quantile(0.9)
            df['extreme_fatigue_indicator'] = (df['fatigue_level'] > extreme_fatigue_threshold).astype(int)
        
        # Optimal conditions threshold
        conditions = []
        if 'energy_reserves' in df.columns:
            conditions.append(df['energy_reserves'] > 0.8)
        if 'confidence_indicator' in df.columns:
            conditions.append(df['confidence_indicator'] > 1.1)
        if 'environmental_favorability_index' in df.columns:
            conditions.append(df['environmental_favorability_index'] > 1.05)
        
        if conditions:
            df['optimal_conditions_indicator'] = np.sum(conditions, axis=0)
        
        # Combined elite performance threshold
        elite_conditions = []
        if 'elite_power_indicator' in df.columns:
            elite_conditions.append(df['elite_power_indicator'])
        if 'high_momentum_indicator' in df.columns:
            elite_conditions.append(df['high_momentum_indicator'])
        if 'optimal_conditions_indicator' in df.columns:
            elite_conditions.append(df['optimal_conditions_indicator'] >= 2)
        
        if elite_conditions:
            df['elite_performance_convergence'] = np.sum(elite_conditions, axis=0)
        
        return df
    
    def _calculate_cross_domain_synergies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate synergies between different feature domains."""
        
        # Physical × Mental synergy
        if 'physical_condition_index' in df.columns and 'psychological_state_index' in df.columns:
            df['mind_body_synergy'] = (
                df['physical_condition_index'] * df['psychological_state_index']
            )
        
        # Form × Environment synergy
        if 'composite_power_index' in df.columns and 'environmental_favorability_index' in df.columns:
            df['form_environment_synergy'] = (
                df['composite_power_index'] * df['environmental_favorability_index']
            )
        
        # Momentum × Opportunity synergy
        if 'composite_momentum_index' in df.columns:
            opportunity_factors = []
            if 'clutch_pa_percentage' in df.columns:
                opportunity_factors.append(df['clutch_pa_percentage'])
            if 'risp_percentage' in df.columns:
                opportunity_factors.append(df['risp_percentage'])
            
            if opportunity_factors:
                opportunity_index = np.mean(opportunity_factors, axis=0)
                df['momentum_opportunity_synergy'] = (
                    df['composite_momentum_index'] * opportunity_index
                )
        
        # Experience × Pressure synergy
        if 'matchup_familiarity_score' in df.columns and 'pressure_performance_index' in df.columns:
            df['experience_pressure_synergy'] = (
                df['matchup_familiarity_score'] * df['pressure_performance_index']
            )
        
        # Rest × Performance synergy
        if 'physical_condition_index' in df.columns and 'power_form_hr_rate' in df.columns:
            df['rest_performance_synergy'] = (
                df['physical_condition_index'] * df['power_form_hr_rate']
            )
        
        return df
    
    def _calculate_performance_multipliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate overall performance multiplier effects."""
        
        # Base performance multiplier
        multiplier_components = []
        
        if 'circadian_performance_factor' in df.columns:
            multiplier_components.append(df['circadian_performance_factor'])
        
        if 'energy_reserves' in df.columns:
            multiplier_components.append(df['energy_reserves'])
        
        if 'confidence_indicator' in df.columns:
            # Scale confidence to be around 1.0
            conf_scaled = df['confidence_indicator'] / df['confidence_indicator'].mean()
            multiplier_components.append(conf_scaled)
        
        if 'park_weather_hr_multiplier' in df.columns:
            multiplier_components.append(df['park_weather_hr_multiplier'])
        
        if multiplier_components:
            df['overall_performance_multiplier'] = np.prod(multiplier_components, axis=0)
        
        # Situation-specific multipliers
        if 'clutch_hr_rate' in df.columns and 'overall_performance_multiplier' in df.columns:
            # Clutch situations get different multiplier
            clutch_situations = df['clutch_hr_rate'] > df['clutch_hr_rate'].mean()
            df['clutch_performance_multiplier'] = np.where(
                clutch_situations,
                df['overall_performance_multiplier'] * 1.1,  # 10% boost in clutch
                df['overall_performance_multiplier']
            )
        
        # Hot streak multiplier
        if 'hot_streak_intensity' in df.columns and 'overall_performance_multiplier' in df.columns:
            hot_streak_boost = 1 + (df['hot_streak_intensity'] * 0.2)  # Up to 20% boost
            df['hot_streak_performance_multiplier'] = (
                df['overall_performance_multiplier'] * hot_streak_boost
            )
        
        # Fatigue-adjusted multiplier
        if 'fatigue_level' in df.columns and 'overall_performance_multiplier' in df.columns:
            fatigue_penalty = 1 - (df['fatigue_level'] * 0.3)  # Up to 30% penalty
            df['fatigue_adjusted_multiplier'] = (
                df['overall_performance_multiplier'] * fatigue_penalty
            )
        
        return df

def calculate_interaction_importance(df: pd.DataFrame, target_col: str = 'hit_hr') -> Dict[str, float]:
    """Calculate importance scores for interaction features."""
    if target_col not in df.columns:
        return {}
    
    interaction_features = [col for col in df.columns 
                          if any(term in col for term in ['synergy', 'boost', 'multiplier', 'ratio', 'index', 'interaction'])]
    
    importance_scores = {}
    for feature in interaction_features:
        if feature in df.columns:
            corr = df[feature].corr(df[target_col])
            if not pd.isna(corr):
                importance_scores[feature] = abs(corr)
    
    return importance_scores

def analyze_interaction_effects(df: pd.DataFrame) -> Dict[str, Dict]:
    """Analyze the effects of feature interactions."""
    analysis = {}
    
    # Analyze multiplicative effects
    multiplicative_features = [col for col in df.columns if 'boost' in col or 'amplification' in col]
    if multiplicative_features:
        analysis['multiplicative'] = {}
        for feature in multiplicative_features:
            data = df[feature].dropna()
            if len(data) > 0:
                analysis['multiplicative'][feature] = {
                    'mean_effect': data.mean(),
                    'max_effect': data.max(),
                    'positive_effects': (data > 1.0).sum() / len(data) if 'multiplier' in feature else (data > 0).sum() / len(data)
                }
    
    # Analyze composite indices
    index_features = [col for col in df.columns if 'index' in col]
    if index_features:
        analysis['composite_indices'] = {}
        for feature in index_features:
            data = df[feature].dropna()
            if len(data) > 0:
                analysis['composite_indices'][feature] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'range': data.max() - data.min()
                }
    
    # Analyze threshold effects
    threshold_features = [col for col in df.columns if 'indicator' in col or 'threshold' in col]
    if threshold_features:
        analysis['thresholds'] = {}
        for feature in threshold_features:
            data = df[feature].dropna()
            if len(data) > 0 and data.nunique() <= 10:  # Categorical/threshold features
                analysis['thresholds'][feature] = {
                    'activation_rate': data.mean(),
                    'unique_values': sorted(data.unique())
                }
    
    return analysis

def validate_interaction_logic(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate the logic of interaction calculations."""
    validation = {}
    
    # Check multiplicative interactions make sense
    if 'energy_circadian_factor' in df.columns:
        energy_circ = df['energy_circadian_factor'].dropna()
        if len(energy_circ) > 0:
            # Should be <= both input components
            if 'energy_reserves' in df.columns and 'circadian_performance_factor' in df.columns:
                energy = df['energy_reserves'].dropna()
                circadian = df['circadian_performance_factor'].dropna()
                # Product should generally be <= max of components (with some tolerance)
                max_product_reasonable = (energy_circ <= (energy.max() * circadian.max() * 1.1)).all()
                validation['multiplicative_bounds_reasonable'] = max_product_reasonable
    
    # Check ratio interactions
    if 'momentum_fatigue_ratio' in df.columns:
        ratio = df['momentum_fatigue_ratio'].dropna()
        if len(ratio) > 0:
            # Ratios should be positive and not extremely large
            positive_ratios = (ratio >= 0).all()
            reasonable_magnitude = (ratio <= 100).all()  # Arbitrary reasonable upper bound
            validation['ratios_reasonable'] = positive_ratios and reasonable_magnitude
    
    # Check composite indices
    composite_features = [col for col in df.columns if 'composite' in col or 'index' in col]
    if composite_features:
        indices_in_range = True
        for feature in composite_features:
            data = df[feature].dropna()
            if len(data) > 0:
                # Most composite indices should be roughly 0-2 range
                if not (data.min() >= -1 and data.max() <= 5):  # Generous bounds
                    indices_in_range = False
                    break
        validation['composite_indices_in_range'] = indices_in_range
    
    return validation