"""
Streak and Momentum Features (Step 5)
====================================

Advanced streak detection and momentum analysis for home run prediction.
Builds on recent form features with sophisticated pattern recognition.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StreakParameters:
    """Configuration for streak detection."""
    min_streak_length: int = 3
    max_lookback_days: int = 21
    hr_threshold: float = 0.15  # HR rate to consider "hot"
    slump_threshold: float = 0.05  # HR rate to consider "cold"
    momentum_window: int = 7  # Days for momentum calculation
    velocity_window: int = 5  # Days for velocity calculation

class StreakMomentumCalculator:
    """Calculate advanced streak and momentum features."""
    
    def __init__(self, params: StreakParameters = None):
        self.params = params or StreakParameters()
        
    def calculate_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive streak and momentum features.
        
        Args:
            df: DataFrame with batting data
            
        Returns:
            DataFrame with added streak/momentum features
        """
        logger.info("Calculating streak and momentum features...")
        
        # Ensure required columns
        required_cols = ['batter', 'date', 'hit_hr']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return df
            
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        # Sort by batter and date
        df_copy = df_copy.sort_values(['batter', 'date']).reset_index(drop=True)
        
        # Calculate individual streak features
        df_copy = self._calculate_hot_streaks(df_copy)
        df_copy = self._calculate_cold_streaks(df_copy)
        df_copy = self._calculate_momentum_features(df_copy)
        df_copy = self._calculate_velocity_features(df_copy)
        df_copy = self._calculate_pattern_features(df_copy)
        df_copy = self._calculate_psychological_features(df_copy)
        
        logger.info("Streak and momentum features calculation completed")
        return df_copy
    
    def _calculate_hot_streaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate hot streak indicators and intensity."""
        
        def calculate_batter_hot_streaks(batter_data):
            """Calculate hot streaks for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            # Initialize features
            data['current_hot_streak'] = 0
            data['max_hot_streak_21d'] = 0
            data['hot_streak_intensity'] = 0.0
            data['days_since_hot_streak'] = np.inf
            data['hot_streak_frequency'] = 0.0
            
            for i in range(len(data)):
                current_date = data.iloc[i]['date']
                
                # Look back for recent games
                lookback_data = data[
                    (data['date'] <= current_date) & 
                    (data['date'] >= current_date - timedelta(days=self.params.max_lookback_days))
                ].iloc[:-1]  # Exclude current game
                
                if len(lookback_data) < 3:
                    continue
                
                # Calculate current hot streak
                current_streak = self._get_current_streak(lookback_data, 'hot')
                data.iloc[i, data.columns.get_loc('current_hot_streak')] = current_streak
                
                # Calculate max hot streak in window
                max_streak = self._get_max_streak_in_window(lookback_data, 'hot')
                data.iloc[i, data.columns.get_loc('max_hot_streak_21d')] = max_streak
                
                # Calculate hot streak intensity (weighted by recency)
                intensity = self._calculate_streak_intensity(lookback_data, 'hot')
                data.iloc[i, data.columns.get_loc('hot_streak_intensity')] = intensity
                
                # Days since last hot streak
                days_since = self._days_since_last_streak(lookback_data, 'hot', current_date)
                data.iloc[i, data.columns.get_loc('days_since_hot_streak')] = days_since
                
                # Hot streak frequency (streaks per week)
                frequency = self._calculate_streak_frequency(lookback_data, 'hot')
                data.iloc[i, data.columns.get_loc('hot_streak_frequency')] = frequency
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_hot_streaks)
    
    def _calculate_cold_streaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cold streak indicators and recovery patterns."""
        
        def calculate_batter_cold_streaks(batter_data):
            """Calculate cold streaks for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            # Initialize features
            data['current_cold_streak'] = 0
            data['max_cold_streak_21d'] = 0
            data['cold_streak_depth'] = 0.0
            data['days_since_cold_streak'] = np.inf
            data['recovery_momentum'] = 0.0
            data['slump_risk_indicator'] = 0.0
            
            for i in range(len(data)):
                current_date = data.iloc[i]['date']
                
                # Look back for recent games
                lookback_data = data[
                    (data['date'] <= current_date) & 
                    (data['date'] >= current_date - timedelta(days=self.params.max_lookback_days))
                ].iloc[:-1]  # Exclude current game
                
                if len(lookback_data) < 3:
                    continue
                
                # Calculate current cold streak
                current_streak = self._get_current_streak(lookback_data, 'cold')
                data.iloc[i, data.columns.get_loc('current_cold_streak')] = current_streak
                
                # Calculate max cold streak in window
                max_streak = self._get_max_streak_in_window(lookback_data, 'cold')
                data.iloc[i, data.columns.get_loc('max_cold_streak_21d')] = max_streak
                
                # Calculate cold streak depth (how bad the slump is)
                depth = self._calculate_streak_depth(lookback_data, 'cold')
                data.iloc[i, data.columns.get_loc('cold_streak_depth')] = depth
                
                # Days since last cold streak
                days_since = self._days_since_last_streak(lookback_data, 'cold', current_date)
                data.iloc[i, data.columns.get_loc('days_since_cold_streak')] = days_since
                
                # Recovery momentum (bounce back from slumps)
                recovery = self._calculate_recovery_momentum(lookback_data)
                data.iloc[i, data.columns.get_loc('recovery_momentum')] = recovery
                
                # Slump risk indicator
                risk = self._calculate_slump_risk(lookback_data)
                data.iloc[i, data.columns.get_loc('slump_risk_indicator')] = risk
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_cold_streaks)
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum and trend features."""
        
        def calculate_batter_momentum(batter_data):
            """Calculate momentum features for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            # Initialize features
            data['power_momentum_7d'] = 0.0
            data['consistency_momentum'] = 0.0
            data['trend_acceleration'] = 0.0
            data['momentum_direction'] = 0  # -1, 0, 1
            data['momentum_strength'] = 0.0
            data['momentum_sustainability'] = 0.0
            
            for i in range(len(data)):
                current_date = data.iloc[i]['date']
                
                # Get momentum window data
                momentum_data = data[
                    (data['date'] <= current_date) & 
                    (data['date'] >= current_date - timedelta(days=self.params.momentum_window))
                ].iloc[:-1]  # Exclude current game
                
                if len(momentum_data) < 3:
                    continue
                
                # Power momentum (weighted recent performance)
                power_momentum = self._calculate_power_momentum(momentum_data)
                data.iloc[i, data.columns.get_loc('power_momentum_7d')] = power_momentum
                
                # Consistency momentum (how steady the trend is)
                consistency = self._calculate_consistency_momentum(momentum_data)
                data.iloc[i, data.columns.get_loc('consistency_momentum')] = consistency
                
                # Trend acceleration (second derivative)
                acceleration = self._calculate_trend_acceleration(momentum_data)
                data.iloc[i, data.columns.get_loc('trend_acceleration')] = acceleration
                
                # Momentum direction and strength
                direction, strength = self._calculate_momentum_vector(momentum_data)
                data.iloc[i, data.columns.get_loc('momentum_direction')] = direction
                data.iloc[i, data.columns.get_loc('momentum_strength')] = strength
                
                # Momentum sustainability
                sustainability = self._calculate_momentum_sustainability(momentum_data)
                data.iloc[i, data.columns.get_loc('momentum_sustainability')] = sustainability
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_momentum)
    
    def _calculate_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate velocity and rate-of-change features."""
        
        def calculate_batter_velocity(batter_data):
            """Calculate velocity features for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            # Initialize features
            data['hr_rate_velocity'] = 0.0
            data['performance_acceleration'] = 0.0
            data['velocity_consistency'] = 0.0
            data['breakout_velocity'] = 0.0
            
            for i in range(len(data)):
                current_date = data.iloc[i]['date']
                
                # Get velocity window data
                velocity_data = data[
                    (data['date'] <= current_date) & 
                    (data['date'] >= current_date - timedelta(days=self.params.velocity_window))
                ].iloc[:-1]  # Exclude current game
                
                if len(velocity_data) < 2:
                    continue
                
                # HR rate velocity (first derivative)
                hr_velocity = self._calculate_hr_rate_velocity(velocity_data)
                data.iloc[i, data.columns.get_loc('hr_rate_velocity')] = hr_velocity
                
                # Performance acceleration (second derivative)
                acceleration = self._calculate_performance_acceleration(velocity_data)
                data.iloc[i, data.columns.get_loc('performance_acceleration')] = acceleration
                
                # Velocity consistency
                consistency = self._calculate_velocity_consistency(velocity_data)
                data.iloc[i, data.columns.get_loc('velocity_consistency')] = consistency
                
                # Breakout velocity (rapid improvement indicator)
                breakout = self._calculate_breakout_velocity(velocity_data)
                data.iloc[i, data.columns.get_loc('breakout_velocity')] = breakout
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_velocity)
    
    def _calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern recognition and cyclical features."""
        
        def calculate_batter_patterns(batter_data):
            """Calculate pattern features for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            # Initialize features
            data['hot_cold_cycle_position'] = 0.0
            data['pattern_stability'] = 0.0
            data['rhythm_indicator'] = 0.0
            data['cycle_prediction'] = 0.0
            
            for i in range(len(data)):
                current_date = data.iloc[i]['date']
                
                # Get pattern window data
                pattern_data = data[
                    (data['date'] <= current_date) & 
                    (data['date'] >= current_date - timedelta(days=self.params.max_lookback_days))
                ].iloc[:-1]  # Exclude current game
                
                if len(pattern_data) < 5:
                    continue
                
                # Hot/cold cycle position
                cycle_pos = self._calculate_cycle_position(pattern_data)
                data.iloc[i, data.columns.get_loc('hot_cold_cycle_position')] = cycle_pos
                
                # Pattern stability
                stability = self._calculate_pattern_stability(pattern_data)
                data.iloc[i, data.columns.get_loc('pattern_stability')] = stability
                
                # Rhythm indicator
                rhythm = self._calculate_rhythm_indicator(pattern_data)
                data.iloc[i, data.columns.get_loc('rhythm_indicator')] = rhythm
                
                # Cycle-based prediction
                prediction = self._calculate_cycle_prediction(pattern_data)
                data.iloc[i, data.columns.get_loc('cycle_prediction')] = prediction
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_patterns)
    
    def _calculate_psychological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate psychological and confidence-based features."""
        
        def calculate_batter_psychology(batter_data):
            """Calculate psychological features for a single batter."""
            data = batter_data.copy().sort_values('date')
            
            # Initialize features
            data['confidence_indicator'] = 0.0
            data['pressure_response'] = 0.0
            data['clutch_momentum'] = 0.0
            data['mental_toughness'] = 0.0
            
            for i in range(len(data)):
                current_date = data.iloc[i]['date']
                
                # Get psychological window data
                psych_data = data[
                    (data['date'] <= current_date) & 
                    (data['date'] >= current_date - timedelta(days=self.params.max_lookback_days))
                ].iloc[:-1]  # Exclude current game
                
                if len(psych_data) < 3:
                    continue
                
                # Confidence indicator (based on recent success patterns)
                confidence = self._calculate_confidence_indicator(psych_data)
                data.iloc[i, data.columns.get_loc('confidence_indicator')] = confidence
                
                # Pressure response (performance in high-leverage situations)
                if 'high_leverage_situation' in data.columns:
                    pressure = self._calculate_pressure_response(psych_data)
                    data.iloc[i, data.columns.get_loc('pressure_response')] = pressure
                
                # Clutch momentum (building on clutch performance)
                if 'clutch_situation' in data.columns:
                    clutch = self._calculate_clutch_momentum(psych_data)
                    data.iloc[i, data.columns.get_loc('clutch_momentum')] = clutch
                
                # Mental toughness (bouncing back from failures)
                toughness = self._calculate_mental_toughness(psych_data)
                data.iloc[i, data.columns.get_loc('mental_toughness')] = toughness
            
            return data
        
        return df.groupby('batter', group_keys=False).apply(calculate_batter_psychology)
    
    # Helper methods for streak calculations
    def _get_current_streak(self, data: pd.DataFrame, streak_type: str) -> int:
        """Get the current streak length."""
        if len(data) == 0:
            return 0
        
        # Calculate daily HR rates
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if streak_type == 'hot':
            threshold = self.params.hr_threshold
            condition = daily_rates >= threshold
        else:  # cold
            threshold = self.params.slump_threshold
            condition = daily_rates <= threshold
        
        # Count consecutive days meeting condition (from most recent backwards)
        streak = 0
        for rate in reversed(condition.values):
            if rate:
                streak += 1
            else:
                break
        
        return streak
    
    def _get_max_streak_in_window(self, data: pd.DataFrame, streak_type: str) -> int:
        """Get the maximum streak length in the window."""
        if len(data) == 0:
            return 0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if streak_type == 'hot':
            threshold = self.params.hr_threshold
            condition = daily_rates >= threshold
        else:  # cold
            threshold = self.params.slump_threshold
            condition = daily_rates <= threshold
        
        # Find maximum consecutive streak
        max_streak = 0
        current_streak = 0
        
        for meets_condition in condition.values:
            if meets_condition:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_streak_intensity(self, data: pd.DataFrame, streak_type: str) -> float:
        """Calculate the intensity of a streak (weighted by recency and magnitude)."""
        if len(data) == 0:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        dates = daily_rates.index
        
        # Calculate recency weights (exponential decay)
        max_date = dates.max()
        days_ago = (max_date - dates).days
        weights = np.exp(-days_ago / 7)  # 7-day half-life
        
        if streak_type == 'hot':
            # Intensity = weighted sum of rates above threshold
            excess_rates = np.maximum(0, daily_rates - self.params.hr_threshold)
        else:  # cold
            # Intensity = weighted sum of rates below threshold (negative values)
            excess_rates = np.minimum(0, daily_rates - self.params.slump_threshold)
        
        intensity = np.sum(weights * excess_rates) / np.sum(weights) if np.sum(weights) > 0 else 0.0
        return intensity
    
    def _days_since_last_streak(self, data: pd.DataFrame, streak_type: str, current_date: datetime) -> float:
        """Calculate days since last significant streak."""
        if len(data) == 0:
            return np.inf
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if streak_type == 'hot':
            threshold = self.params.hr_threshold
            condition = daily_rates >= threshold
        else:  # cold
            threshold = self.params.slump_threshold
            condition = daily_rates <= threshold
        
        # Find last streak of minimum length
        dates = daily_rates.index
        streak_dates = []
        current_streak = []
        
        for i, (date, meets_condition) in enumerate(zip(dates, condition.values)):
            if meets_condition:
                current_streak.append(date)
            else:
                if len(current_streak) >= self.params.min_streak_length:
                    streak_dates.extend(current_streak)
                current_streak = []
        
        # Check final streak
        if len(current_streak) >= self.params.min_streak_length:
            streak_dates.extend(current_streak)
        
        if not streak_dates:
            return np.inf
        
        last_streak_date = max(streak_dates)
        days_since = (current_date - last_streak_date).days
        return float(days_since)
    
    def _calculate_streak_frequency(self, data: pd.DataFrame, streak_type: str) -> float:
        """Calculate frequency of streaks (streaks per week)."""
        if len(data) == 0:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if streak_type == 'hot':
            threshold = self.params.hr_threshold
            condition = daily_rates >= threshold
        else:  # cold
            threshold = self.params.slump_threshold
            condition = daily_rates <= threshold
        
        # Count streaks of minimum length
        streak_count = 0
        current_streak = 0
        
        for meets_condition in condition.values:
            if meets_condition:
                current_streak += 1
            else:
                if current_streak >= self.params.min_streak_length:
                    streak_count += 1
                current_streak = 0
        
        # Check final streak
        if current_streak >= self.params.min_streak_length:
            streak_count += 1
        
        # Calculate frequency (streaks per week)
        total_days = (daily_rates.index.max() - daily_rates.index.min()).days + 1
        weeks = total_days / 7
        frequency = streak_count / max(weeks, 1)
        
        return frequency
    
    def _calculate_streak_depth(self, data: pd.DataFrame, streak_type: str) -> float:
        """Calculate the depth/severity of cold streaks."""
        if len(data) == 0 or streak_type != 'cold':
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        # Calculate how far below threshold the rates are
        below_threshold = self.params.slump_threshold - daily_rates
        below_threshold = np.maximum(0, below_threshold)  # Only negative deviations
        
        # Weight by recency
        dates = daily_rates.index
        max_date = dates.max()
        days_ago = (max_date - dates).days
        weights = np.exp(-days_ago / 7)  # 7-day half-life
        
        depth = np.sum(weights * below_threshold) / np.sum(weights) if np.sum(weights) > 0 else 0.0
        return depth
    
    def _calculate_recovery_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum coming out of slumps."""
        if len(data) < 3:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        # Look for pattern: low rates followed by improvement
        recent_rates = daily_rates.tail(5).values
        if len(recent_rates) < 3:
            return 0.0
        
        # Calculate trend in recent rates
        x = np.arange(len(recent_rates))
        trend = np.polyfit(x, recent_rates, 1)[0]  # Slope
        
        # Weight trend by how low we started
        min_rate = np.min(recent_rates[:3])  # Minimum of first 3 games
        baseline_depression = max(0, self.params.hr_threshold - min_rate)
        
        # Recovery momentum = trend * depth of hole we're climbing out of
        recovery = trend * (1 + baseline_depression * 2)
        return recovery
    
    def _calculate_slump_risk(self, data: pd.DataFrame) -> float:
        """Calculate risk of entering a slump."""
        if len(data) < 3:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        recent_rates = daily_rates.tail(5).values
        
        if len(recent_rates) < 3:
            return 0.0
        
        # Risk factors
        risk_score = 0.0
        
        # 1. Declining trend
        x = np.arange(len(recent_rates))
        trend = np.polyfit(x, recent_rates, 1)[0]
        if trend < 0:
            risk_score += abs(trend) * 2
        
        # 2. Current rate below threshold
        current_rate = recent_rates[-1]
        if current_rate < self.params.slump_threshold:
            risk_score += (self.params.slump_threshold - current_rate) * 3
        
        # 3. Consistency of poor performance
        below_threshold_count = np.sum(recent_rates < self.params.slump_threshold)
        risk_score += (below_threshold_count / len(recent_rates)) * 0.5
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    # Momentum calculation helpers
    def _calculate_power_momentum(self, data: pd.DataFrame) -> float:
        """Calculate power-weighted momentum."""
        if len(data) == 0:
            return 0.0
        
        # Use exit velocity and launch angle if available
        if 'exit_velocity' in data.columns and 'launch_angle' in data.columns:
            power_metric = data['exit_velocity'] * np.cos(np.radians(data['launch_angle'] - 25))
        else:
            # Fallback to HR rate
            power_metric = data.groupby('date')['hit_hr'].mean()
        
        # Calculate weighted momentum with exponential decay
        if hasattr(power_metric, 'index'):
            dates = power_metric.index
        else:
            dates = data['date'].unique()
            
        weights = np.exp(-np.arange(len(power_metric))[::-1] / 3)  # 3-day half-life
        momentum = np.average(power_metric, weights=weights)
        
        return momentum
    
    def _calculate_consistency_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum consistency (low variance = high consistency)."""
        if len(data) < 3:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean()
        variance = daily_rates.var()
        
        # Inverse relationship: lower variance = higher consistency
        consistency = 1 / (1 + variance * 10)  # Scale factor
        return consistency
    
    def _calculate_trend_acceleration(self, data: pd.DataFrame) -> float:
        """Calculate second derivative of performance trend."""
        if len(data) < 4:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if len(daily_rates) < 4:
            return 0.0
        
        # Calculate second derivative (acceleration)
        x = np.arange(len(daily_rates))
        poly_coeffs = np.polyfit(x, daily_rates.values, 2)
        acceleration = 2 * poly_coeffs[0]  # Second derivative coefficient
        
        return acceleration
    
    def _calculate_momentum_vector(self, data: pd.DataFrame) -> Tuple[int, float]:
        """Calculate momentum direction and strength."""
        if len(data) < 2:
            return 0, 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if len(daily_rates) < 2:
            return 0, 0.0
        
        # Calculate trend
        x = np.arange(len(daily_rates))
        trend = np.polyfit(x, daily_rates.values, 1)[0]
        
        # Direction: -1 (negative), 0 (flat), 1 (positive)
        if abs(trend) < 0.001:  # Threshold for "flat"
            direction = 0
        else:
            direction = 1 if trend > 0 else -1
        
        # Strength: magnitude of trend
        strength = abs(trend)
        
        return direction, strength
    
    def _calculate_momentum_sustainability(self, data: pd.DataFrame) -> float:
        """Calculate how sustainable the current momentum is."""
        if len(data) < 3:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        # Sustainability factors:
        # 1. Consistency (low variance)
        variance = daily_rates.var()
        consistency_factor = 1 / (1 + variance * 5)
        
        # 2. Sample size (more games = more sustainable)
        sample_size_factor = min(len(daily_rates) / 10, 1.0)
        
        # 3. Current level vs. historical baseline
        current_rate = daily_rates.tail(3).mean()
        baseline_rate = daily_rates.mean()
        level_factor = min(current_rate / max(baseline_rate, 0.01), 2.0)
        
        sustainability = consistency_factor * sample_size_factor * (level_factor / 2)
        return min(sustainability, 1.0)
    
    # Velocity calculation helpers
    def _calculate_hr_rate_velocity(self, data: pd.DataFrame) -> float:
        """Calculate first derivative of HR rate."""
        if len(data) < 2:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if len(daily_rates) < 2:
            return 0.0
        
        # Simple velocity: change over time
        rate_change = daily_rates.diff().tail(3).mean()  # Average recent change
        return rate_change
    
    def _calculate_performance_acceleration(self, data: pd.DataFrame) -> float:
        """Calculate second derivative of performance."""
        if len(data) < 3:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if len(daily_rates) < 3:
            return 0.0
        
        # Calculate acceleration from velocity changes
        velocities = daily_rates.diff()
        acceleration = velocities.diff().tail(2).mean()  # Average recent acceleration
        return acceleration
    
    def _calculate_velocity_consistency(self, data: pd.DataFrame) -> float:
        """Calculate consistency of velocity (rate of change)."""
        if len(data) < 3:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if len(daily_rates) < 3:
            return 0.0
        
        velocities = daily_rates.diff().dropna()
        if len(velocities) == 0:
            return 0.0
        
        # Low variance in velocity = high consistency
        velocity_variance = velocities.var()
        consistency = 1 / (1 + velocity_variance * 20)  # Scale factor
        
        return consistency
    
    def _calculate_breakout_velocity(self, data: pd.DataFrame) -> float:
        """Calculate breakout velocity (rapid improvement indicator)."""
        if len(data) < 3:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        if len(daily_rates) < 3:
            return 0.0
        
        # Look for rapid acceleration in performance
        recent_rates = daily_rates.tail(3).values
        
        if len(recent_rates) < 3:
            return 0.0
        
        # Calculate if we're seeing accelerating improvement
        early_avg = recent_rates[0]
        late_avg = recent_rates[-1]
        
        improvement = late_avg - early_avg
        
        # Breakout velocity is high when:
        # 1. We're improving rapidly
        # 2. From a previously low level
        baseline_factor = 1 + max(0, self.params.hr_threshold - early_avg)
        breakout = improvement * baseline_factor
        
        return max(0, breakout)
    
    # Pattern recognition helpers
    def _calculate_cycle_position(self, data: pd.DataFrame) -> float:
        """Calculate position in hot/cold cycle."""
        if len(data) < 5:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        # Simple cycle detection using moving averages
        short_ma = daily_rates.rolling(window=3).mean()
        long_ma = daily_rates.rolling(window=7).mean()
        
        current_short = short_ma.iloc[-1] if len(short_ma) > 0 else 0
        current_long = long_ma.iloc[-1] if len(long_ma) > 0 else 0
        
        # Position relative to cycle: -1 (cold), 0 (neutral), 1 (hot)
        if current_short > current_long + 0.05:
            return 1.0  # Hot phase
        elif current_short < current_long - 0.05:
            return -1.0  # Cold phase
        else:
            return 0.0  # Neutral phase
    
    def _calculate_pattern_stability(self, data: pd.DataFrame) -> float:
        """Calculate stability of performance patterns."""
        if len(data) < 5:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        # Measure autocorrelation (pattern consistency)
        if len(daily_rates) < 5:
            return 0.0
        
        # Simple pattern stability: inverse of relative standard deviation
        mean_rate = daily_rates.mean()
        std_rate = daily_rates.std()
        
        if mean_rate == 0:
            return 0.0
        
        stability = 1 / (1 + (std_rate / mean_rate))
        return stability
    
    def _calculate_rhythm_indicator(self, data: pd.DataFrame) -> float:
        """Calculate rhythm/timing indicator."""
        if len(data) < 5:
            return 0.0
        
        # Look for regular patterns in HR timing
        hr_games = data[data['hit_hr'] == 1]['date'].sort_values()
        
        if len(hr_games) < 3:
            return 0.0
        
        # Calculate intervals between HRs
        intervals = hr_games.diff().dt.days.dropna()
        
        if len(intervals) < 2:
            return 0.0
        
        # Rhythm = consistency of intervals (low variance = good rhythm)
        interval_variance = intervals.var()
        rhythm = 1 / (1 + interval_variance / 5)  # Scale factor
        
        return rhythm
    
    def _calculate_cycle_prediction(self, data: pd.DataFrame) -> float:
        """Predict where we are in the performance cycle."""
        if len(data) < 7:
            return 0.0
        
        daily_rates = data.groupby('date')['hit_hr'].mean().sort_index()
        
        # Simple trend extrapolation
        x = np.arange(len(daily_rates))
        recent_x = x[-5:]  # Last 5 days
        recent_rates = daily_rates.iloc[-5:].values
        
        if len(recent_rates) < 3:
            return 0.0
        
        # Fit trend to recent data
        trend = np.polyfit(recent_x, recent_rates, 1)[0]
        
        # Predict next value
        next_x = len(daily_rates)
        prediction = daily_rates.iloc[-1] + trend
        
        return prediction
    
    # Psychological feature helpers
    def _calculate_confidence_indicator(self, data: pd.DataFrame) -> float:
        """Calculate confidence based on recent success patterns."""
        if len(data) < 3:
            return 0.0
        
        # Recent success rate
        recent_hr_rate = data.tail(5)['hit_hr'].mean()
        
        # Success momentum (recent > historical)
        historical_rate = data['hit_hr'].mean()
        momentum_factor = recent_hr_rate / max(historical_rate, 0.01)
        
        # Consistency factor (less variance = more confidence)
        recent_variance = data.tail(5)['hit_hr'].var()
        consistency_factor = 1 / (1 + recent_variance * 5)
        
        confidence = momentum_factor * consistency_factor
        return min(confidence, 2.0)  # Cap at 2.0
    
    def _calculate_pressure_response(self, data: pd.DataFrame) -> float:
        """Calculate response to pressure situations."""
        if 'high_leverage_situation' not in data.columns:
            return 0.0
        
        high_leverage_data = data[data['high_leverage_situation'] == 1]
        regular_data = data[data['high_leverage_situation'] == 0]
        
        if len(high_leverage_data) == 0 or len(regular_data) == 0:
            return 0.0
        
        high_leverage_rate = high_leverage_data['hit_hr'].mean()
        regular_rate = regular_data['hit_hr'].mean()
        
        # Pressure response = ratio of high leverage to regular performance
        response = high_leverage_rate / max(regular_rate, 0.01)
        return response
    
    def _calculate_clutch_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum in clutch situations."""
        if 'clutch_situation' not in data.columns:
            return 0.0
        
        clutch_data = data[data['clutch_situation'] == 1].tail(5)
        
        if len(clutch_data) < 2:
            return 0.0
        
        # Recent clutch performance trend
        clutch_rates = clutch_data.groupby('date')['hit_hr'].mean().sort_index()
        
        if len(clutch_rates) < 2:
            return 0.0
        
        # Calculate trend in clutch performance
        x = np.arange(len(clutch_rates))
        trend = np.polyfit(x, clutch_rates.values, 1)[0]
        
        return trend
    
    def _calculate_mental_toughness(self, data: pd.DataFrame) -> float:
        """Calculate mental toughness (bounce-back ability)."""
        if len(data) < 5:
            return 0.0
        
        # Find sequences of failure followed by success
        hr_sequence = data.sort_values('date')['hit_hr'].values
        
        bounce_backs = 0
        opportunities = 0
        
        for i in range(2, len(hr_sequence)):
            # Look for pattern: 0, 0, 1 (failure, failure, success)
            if hr_sequence[i-2] == 0 and hr_sequence[i-1] == 0:
                opportunities += 1
                if hr_sequence[i] == 1:
                    bounce_backs += 1
        
        if opportunities == 0:
            return 0.0
        
        toughness = bounce_backs / opportunities
        return toughness