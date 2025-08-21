"""
Data Utilities Module
====================

Utilities for data processing, validation, and transformation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Union
import re
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and cleaning utilities."""
    
    @staticmethod
    def standardize_name(name: Union[str, float]) -> str:
        """Standardize player names for consistent matching."""
        if pd.isna(name):
            return ""
        
        name_str = str(name).strip()
        
        # Handle "Last, First" format
        if "," in name_str:
            parts = name_str.split(",", 1)
            if len(parts) == 2:
                return f"{parts[1].strip()} {parts[0].strip()}"
        
        return name_str
    
    @staticmethod
    def validate_date_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """Ensure date column exists and is properly formatted."""
        df = df.copy()
        
        if date_col not in df.columns:
            if 'game_date' in df.columns:
                df[date_col] = pd.to_datetime(df['game_date'])
                logger.info(f"Created '{date_col}' column from 'game_date'")
            else:
                raise KeyError(f"No '{date_col}' or 'game_date' column found")
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
        
        # Remove rows with invalid dates
        invalid_dates = df[date_col].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Removing {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=[date_col])
        
        return df
    
    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_cols: List[str], 
                                 context: str = "") -> None:
        """Validate that required columns exist in DataFrame."""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            available_cols = list(df.columns)
            logger.error(f"Missing columns {missing} in {context}")
            logger.error(f"Available columns: {available_cols}")
            raise KeyError(f"Missing required columns {missing} in {context}")
    
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame, numeric_cols: List[str] = None) -> pd.DataFrame:
        """Clean numeric data by handling inf values and filling NaNs."""
        df = df.copy()
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col in df.columns:
                # Replace inf values with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN with median (more robust than mean)
                if df[col].notna().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(0)
        
        return df
    
    @staticmethod
    def cap_outliers(df: pd.DataFrame, col: str, lower_pct: float = 0.01, 
                    upper_pct: float = 0.99) -> pd.DataFrame:
        """Cap outliers using percentile-based limits."""
        df = df.copy()
        
        if col in df.columns and df[col].notna().sum() > 0:
            lower_bound = df[col].quantile(lower_pct)
            upper_bound = df[col].quantile(upper_pct)
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    @staticmethod
    def validate_predictions(predictions: np.ndarray) -> bool:
        """Validate model predictions are reasonable."""
        if len(predictions) == 0:
            return False
        
        # Check for NaN or infinite values
        if not np.isfinite(predictions).all():
            logger.error("Predictions contain NaN or infinite values")
            return False
        
        # Check probability bounds
        if not ((predictions >= 0) & (predictions <= 1)).all():
            logger.error("Predictions outside [0,1] range")
            return False
        
        # Check for reasonable distribution
        mean_prob = predictions.mean()
        if mean_prob < 0.01 or mean_prob > 0.5:
            logger.warning(f"Unusual prediction distribution: mean={mean_prob:.4f}")
        
        return True

class FeatureValidator:
    """Feature engineering validation utilities."""
    
    @staticmethod
    def validate_rolling_features(df: pd.DataFrame, group_col: str, 
                                 feature_cols: List[str]) -> Dict[str, Any]:
        """Validate rolling feature calculations."""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        for col in feature_cols:
            if col in df.columns:
                # Check for non-negative values where expected
                if 'hr_rate' in col and (df[col] < 0).any():
                    validation_results['issues'].append(f"Negative values found in {col}")
                    validation_results['valid'] = False
                
                # Check for reasonable bounds
                if 'hr_rate' in col and (df[col] > 1.0).any():
                    validation_results['warnings'].append(f"HR rates > 100% found in {col}")
                
                # Check for sufficient data coverage
                coverage = df[col].notna().sum() / len(df)
                if coverage < 0.5:
                    validation_results['warnings'].append(
                        f"Low data coverage ({coverage:.1%}) for {col}"
                    )
        
        return validation_results

class DateUtils:
    """Date and time utilities for baseball data."""
    
    @staticmethod
    def get_season_from_date(date: pd.Timestamp) -> int:
        """Get baseball season year from date."""
        # Baseball season runs roughly March to October
        # Dates in November-February belong to previous season
        if date.month in [11, 12, 1, 2]:
            return date.year - 1 if date.month in [11, 12] else date.year - 1
        return date.year
    
    @staticmethod
    def is_valid_game_date(date: pd.Timestamp) -> bool:
        """Check if date falls within typical baseball season."""
        month = date.month
        return month in range(3, 11)  # March through October
    
    @staticmethod
    def get_date_range_for_features(target_date: pd.Timestamp, 
                                   lookback_days: int = 45) -> tuple:
        """Get optimal date range for feature calculation."""
        target_date = pd.to_datetime(target_date).normalize()
        start_date = target_date - timedelta(days=lookback_days)
        return start_date, target_date

class StatcastUtils:
    """Utilities for working with Statcast data."""
    
    @staticmethod
    def get_required_columns() -> Dict[str, List[str]]:
        """Get required and optional columns for Statcast data."""
        return {
            'required': [
                'game_date', 'game_type', 'home_team', 'away_team', 'game_pk',
                'inning', 'events', 'batter', 'pitcher', 'p_throws', 'stand',
                'inning_topbot', 'pitch_type', 'release_speed'
            ],
            'optional': [
                'launch_speed', 'launch_angle', 'release_pos_x', 'release_pos_y',
                'release_pos_z', 'zone', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
                'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot',
                'effective_speed', 'release_spin_rate', 'release_extension',
                'spin_axis', 'api_break_z_with_gravity', 'api_break_x_arm',
                'arm_angle', 'bat_speed', 'attack_angle', 'attack_direction',
                'swing_path_tilt'
            ]
        }
    
    @staticmethod
    def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add commonly used derived columns to Statcast data."""
        df = df.copy()
        
        # Basic flags
        df['is_pa'] = df['events'].notna().astype(int)
        df['is_hr'] = (df['events'] == 'home_run').astype(int)
        
        # Batting team
        if all(col in df.columns for col in ['inning_topbot', 'home_team', 'away_team']):
            df['bat_team'] = np.where(
                df['inning_topbot'] == 'Top', 
                df['away_team'], 
                df['home_team']
            )
        
        # Date normalization
        if 'game_date' in df.columns:
            df['date'] = pd.to_datetime(df['game_date']).dt.normalize()
            df['season'] = df['date'].apply(DateUtils.get_season_from_date)
        
        return df
    
    @staticmethod
    def filter_regular_season(df: pd.DataFrame) -> pd.DataFrame:
        """Filter to regular season games only."""
        if 'game_type' in df.columns:
            return df[df['game_type'] == 'R'].copy()
        return df

class CacheManager:
    """Manage data caching for performance."""
    
    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_path(self, identifier: str, format: str = "parquet") -> Path:
        """Generate cache file path."""
        return self.cache_dir / f"{identifier}.{format}"
    
    def save_cache(self, df: pd.DataFrame, identifier: str, 
                  format: str = "parquet") -> None:
        """Save DataFrame to cache."""
        cache_path = self.get_cache_path(identifier, format)
        
        try:
            if format.lower() == "parquet":
                df.to_parquet(cache_path, index=False)
            else:
                df.to_csv(cache_path, index=False)
            
            logger.info(f"Saved cache to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_cache(self, identifier: str, format: str = "parquet") -> Optional[pd.DataFrame]:
        """Load DataFrame from cache."""
        cache_path = self.get_cache_path(identifier, format)
        
        if not cache_path.exists():
            return None
        
        try:
            if format.lower() == "parquet":
                df = pd.read_parquet(cache_path)
            else:
                df = pd.read_csv(cache_path, low_memory=False)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
            
            logger.info(f"Loaded cache from {cache_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None
    
    def cache_exists(self, identifier: str, format: str = "parquet") -> bool:
        """Check if cache exists."""
        return self.get_cache_path(identifier, format).exists()

# Export commonly used functions
__all__ = [
    'DataValidator', 'FeatureValidator', 'DateUtils', 
    'StatcastUtils', 'CacheManager'
]