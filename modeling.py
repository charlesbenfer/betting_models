"""
Enhanced Modeling Module with Proper Train/Test Splits and Comprehensive Testing
==============================================================================

Improved dual-model system with time-aware splitting, cross-validation, and extensive validation.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, train_test_split
import xgboost as xgb

# Import your existing modules
from config import config
from data_utils import DataValidator

logger = logging.getLogger(__name__)

class DataSplitter:
    """Handles various data splitting strategies for time series and cross-validation."""
    
    @staticmethod
    def time_based_split(df: pd.DataFrame, date_column: str = 'date', 
                        train_ratio: float = 0.7, val_ratio: float = 0.15,
                        gap_days: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically to prevent data leakage.
        
        Args:
            df: DataFrame with date column
            date_column: Name of the date column
            train_ratio: Proportion for training (default 70%)
            val_ratio: Proportion for validation (default 15%, test gets remainder)
            gap_days: Days to skip between train/val and val/test to prevent leakage
        
        Returns:
            train_df, val_df, test_df
        """
        if date_column not in df.columns:
            logger.warning(f"Date column '{date_column}' not found. Using random split.")
            return DataSplitter.random_split(df, train_ratio, val_ratio)
        
        # Sort by date
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        df_sorted[date_column] = pd.to_datetime(df_sorted[date_column])
        
        # Calculate split points
        n_total = len(df_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Apply gaps if specified
        if gap_days > 0:
            train_end_date = df_sorted.iloc[n_train-1][date_column]
            val_start_date = train_end_date + timedelta(days=gap_days)
            
            # Find actual start index for validation
            val_start_idx = df_sorted[df_sorted[date_column] >= val_start_date].index[0]
            
            # Adjust validation end and test start
            val_end_idx = min(val_start_idx + n_val, n_total)
            test_start_date = df_sorted.iloc[val_end_idx-1][date_column] + timedelta(days=gap_days)
            test_start_idx = df_sorted[df_sorted[date_column] >= test_start_date].index
            test_start_idx = test_start_idx[0] if len(test_start_idx) > 0 else val_end_idx
            
            train_df = df_sorted.iloc[:n_train]
            val_df = df_sorted.iloc[val_start_idx:val_end_idx]
            test_df = df_sorted.iloc[test_start_idx:]
        else:
            train_df = df_sorted.iloc[:n_train]
            val_df = df_sorted.iloc[n_train:n_train + n_val]
            test_df = df_sorted.iloc[n_train + n_val:]
        
        logger.info(f"Time-based split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        logger.info(f"Date ranges - Train: {train_df[date_column].min()} to {train_df[date_column].max()}")
        logger.info(f"Date ranges - Val: {val_df[date_column].min()} to {val_df[date_column].max()}")
        logger.info(f"Date ranges - Test: {test_df[date_column].min()} to {test_df[date_column].max()}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def random_split(df: pd.DataFrame, train_ratio: float = 0.7, 
                    val_ratio: float = 0.15, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Random stratified split maintaining class balance."""
        
        # Determine target column
        target_col = None
        if 'hit_hr' in df.columns:
            target_col = 'hit_hr'
        elif 'home_runs' in df.columns:
            df = df.copy()
            df['hit_hr'] = (df['home_runs'] > 0).astype(int)
            target_col = 'hit_hr'
        
        if stratify and target_col:
            # First split: train vs (val + test)
            train_df, temp_df = train_test_split(
                df, test_size=(1 - train_ratio), 
                stratify=df[target_col], random_state=config.RANDOM_STATE
            )
            
            # Second split: val vs test
            val_size = val_ratio / (1 - train_ratio)
            val_df, test_df = train_test_split(
                temp_df, test_size=(1 - val_size), 
                stratify=temp_df[target_col], random_state=config.RANDOM_STATE
            )
        else:
            # Simple random split
            train_df, temp_df = train_test_split(
                df, test_size=(1 - train_ratio), random_state=config.RANDOM_STATE
            )
            val_size = val_ratio / (1 - train_ratio)
            val_df, test_df = train_test_split(
                temp_df, test_size=(1 - val_size), random_state=config.RANDOM_STATE
            )
        
        logger.info(f"Random split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    @staticmethod
    def seasonal_split(df: pd.DataFrame, test_seasons: List[int], 
                      val_seasons: List[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by specific seasons for realistic evaluation."""
        
        if 'season' not in df.columns:
            if 'date' in df.columns:
                df = df.copy()
                df['season'] = pd.to_datetime(df['date']).dt.year
            else:
                raise ValueError("No season or date column found")
        
        test_df = df[df['season'].isin(test_seasons)].copy()
        
        if val_seasons:
            val_df = df[df['season'].isin(val_seasons)].copy()
            train_df = df[~df['season'].isin(test_seasons + val_seasons)].copy()
        else:
            # Use one season before test for validation
            if test_seasons:
                val_season = min(test_seasons) - 1
                val_df = df[df['season'] == val_season].copy()
                train_df = df[~df['season'].isin(test_seasons + [val_season])].copy()
            else:
                raise ValueError("No test seasons specified")
        
        logger.info(f"Seasonal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        logger.info(f"Train seasons: {sorted(train_df['season'].unique())}")
        logger.info(f"Val seasons: {sorted(val_df['season'].unique())}")
        logger.info(f"Test seasons: {sorted(test_df['season'].unique())}")
        
        return train_df, val_df, test_df

class ModelValidator:
    """Enhanced validation utilities for model performance."""
    
    @staticmethod
    def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray, label: str = "") -> Dict[str, float]:
        """Comprehensive model evaluation with additional metrics."""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_prob),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'baseline_accuracy': 1 - y_true.mean(),  # Always predict negative class
                'positive_rate': y_true.mean(),
                'prediction_rate': y_pred.mean()
            }
            
            # Add precision-recall AUC
            if len(np.unique(y_true)) > 1:
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
                metrics['pr_auc'] = np.trapz(precision_curve, recall_curve)
            
            # Calculate lift at various thresholds
            for threshold in [0.1, 0.2, 0.3]:
                high_conf_mask = y_prob >= threshold
                if high_conf_mask.sum() > 0:
                    lift = y_true[high_conf_mask].mean() / y_true.mean()
                    metrics[f'lift_at_{threshold:.1f}'] = lift
            
            logger.info(f"{label} Metrics: " + 
                       ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model performance: {e}")
            return {}
    
    @staticmethod
    def validate_predictions(predictions: np.ndarray) -> bool:
        """Enhanced prediction validation."""
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
        std_prob = predictions.std()
        
        if mean_prob < 0.01 or mean_prob > 0.5:
            logger.warning(f"Unusual prediction distribution: mean={mean_prob:.4f}")
        
        if std_prob < 0.01:
            logger.warning(f"Low prediction variance: std={std_prob:.4f}")
        
        # Check for prediction concentration
        unique_preds = len(np.unique(predictions.round(3)))
        if unique_preds < len(predictions) * 0.1:
            logger.warning(f"Predictions too concentrated: {unique_preds} unique values")
        
        return True
    
    @staticmethod
    def calculate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                                    n_bins: int = 10) -> Dict[str, float]:
        """Calculate calibration metrics for probability predictions."""
        try:
            # Bin predictions
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_data = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    
                    calibration_data.append({
                        'bin_lower': bin_lower,
                        'bin_upper': bin_upper,
                        'prop_in_bin': prop_in_bin,
                        'accuracy_in_bin': accuracy_in_bin,
                        'avg_confidence': avg_confidence_in_bin,
                        'calibration_error': abs(accuracy_in_bin - avg_confidence_in_bin)
                    })
            
            if not calibration_data:
                return {'ece': np.nan, 'mce': np.nan}
            
            # Expected Calibration Error (ECE)
            ece = sum(d['prop_in_bin'] * d['calibration_error'] for d in calibration_data)
            
            # Maximum Calibration Error (MCE)
            mce = max(d['calibration_error'] for d in calibration_data)
            
            return {'ece': ece, 'mce': mce, 'calibration_data': calibration_data}
            
        except Exception as e:
            logger.error(f"Calibration calculation failed: {e}")
            return {'ece': np.nan, 'mce': np.nan}

class FeatureSelector:
    """Enhanced feature selection for dual model system."""
    
    def __init__(self):
        self.core_features = config.CORE_FEATURES.copy()
        self.bat_tracking_features = config.BAT_TRACKING_FEATURES.copy()
        self.matchup_features = config.MATCHUP_FEATURES.copy()
        self.situational_features = config.SITUATIONAL_FEATURES.copy()
        self.weather_features = config.WEATHER_FEATURES.copy()
        self.recent_form_features = config.RECENT_FORM_FEATURES.copy()
        self.streak_momentum_features = config.STREAK_MOMENTUM_FEATURES.copy()
        self.ballpark_features = config.BALLPARK_FEATURES.copy()
        self.temporal_fatigue_features = config.TEMPORAL_FATIGUE_FEATURES.copy()
        self.interaction_features = config.INTERACTION_FEATURES.copy()
    
    def identify_available_features(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify which features are available in the dataset."""
        # Get numeric columns excluding targets
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['home_runs', 'hr', 'hit_hr', 'batter', 'game_pk', 'season']
        available_numeric = [col for col in numeric_cols if col not in exclude_cols]
        
        # Core features (basic features that should always be available)
        available_core = []
        for feature in self.core_features:
            if feature in available_numeric:
                available_core.append(feature)
        
        # Add rolling features
        rolling_features = [col for col in available_numeric if 'roll' in col and 'hr_rate' in col]
        available_core.extend(rolling_features)
        
        # Add handedness features
        handedness_features = [col for col in available_numeric if any(x in col for x in ['_R', '_L']) and 'hr_rate' in col]
        available_core.extend(handedness_features)
        
        # Add pitcher features
        pitcher_features = [col for col in available_numeric if col.startswith('p_') and any(x in col for x in ['hr', 'ev', 'vel'])]
        available_core.extend(pitcher_features)
        
        # Add park factors
        if 'park_factor' in available_numeric:
            available_core.append('park_factor')
        
        # Remove duplicates while preserving order
        available_core = list(dict.fromkeys(available_core))
        
        # Bat tracking features
        available_bt = [col for col in self.bat_tracking_features if col in available_numeric]
        # Add rolling bat tracking features
        bt_rolling = [col for col in available_numeric if any(bt in col for bt in self.bat_tracking_features) and 'roll' in col]
        available_bt.extend(bt_rolling)
        available_bt = list(dict.fromkeys(available_bt))
        
        # Matchup features
        available_matchup = [col for col in self.matchup_features if col in available_numeric]
        available_matchup = list(dict.fromkeys(available_matchup))
        
        # Situational features
        available_situational = [col for col in self.situational_features if col in available_numeric]
        available_situational = list(dict.fromkeys(available_situational))
        
        # Weather features
        available_weather = [col for col in self.weather_features if col in available_numeric]
        available_weather = list(dict.fromkeys(available_weather))
        
        # Recent form features
        available_recent_form = [col for col in self.recent_form_features if col in available_numeric]
        available_recent_form = list(dict.fromkeys(available_recent_form))
        
        # Streak and momentum features
        available_streak_momentum = [col for col in self.streak_momentum_features if col in available_numeric]
        available_streak_momentum = list(dict.fromkeys(available_streak_momentum))
        
        # Ballpark features
        available_ballpark = [col for col in self.ballpark_features if col in available_numeric]
        available_ballpark = list(dict.fromkeys(available_ballpark))
        
        # Temporal fatigue features
        available_temporal_fatigue = [col for col in self.temporal_fatigue_features if col in available_numeric]
        available_temporal_fatigue = list(dict.fromkeys(available_temporal_fatigue))
        
        # Interaction features
        available_interactions = [col for col in self.interaction_features if col in available_numeric]
        available_interactions = list(dict.fromkeys(available_interactions))
        
        # Enhanced features = ALL available features (core + matchup + situational + weather + recent form + streak momentum + ballpark + temporal fatigue + interactions)
        # No longer requires bat tracking features specifically
        available_enhanced = (available_core + available_matchup + 
                            available_situational + available_weather + available_recent_form + 
                            available_streak_momentum + available_ballpark + available_temporal_fatigue +
                            available_interactions)
        
        return {
            'core': available_core,
            'bat_tracking': available_bt,
            'matchup': available_matchup,
            'situational': available_situational,
            'weather': available_weather,
            'recent_form': available_recent_form,
            'streak_momentum': available_streak_momentum,
            'ballpark': available_ballpark,
            'temporal_fatigue': available_temporal_fatigue,
            'interactions': available_interactions,
            'enhanced': available_enhanced,
            'all_numeric': available_numeric
        }
    
    def assess_enhanced_feature_coverage(self, df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, Any]:
        """Assess enhanced feature data coverage - no longer requires bat tracking."""
        feature_info = self.identify_available_features(df)
        enhanced_features = feature_info['enhanced']
        
        if not enhanced_features:
            return {
                'sufficient': False,
                'coverage': 0.0,
                'available_features': [],
                'reason': 'No enhanced features found'
            }
        
        # Check coverage across enhanced features
        enhanced_data = df[enhanced_features]
        non_null_rows = enhanced_data.notna().any(axis=1).sum()
        coverage = non_null_rows / len(df)
        
        return {
            'sufficient': coverage >= threshold,
            'coverage': coverage,
            'available_features': enhanced_features,
            'reason': f'Coverage {coverage:.2%} (threshold: {threshold:.2%})'
        }

class BaseModelComponent:
    """Enhanced base class for individual model components."""
    
    def __init__(self, feature_list: List[str]):
        self.feature_list = feature_list
        self.models = {}
        self.scaler = StandardScaler()
        self.calibrator = None
        self.feature_medians = {}
        self.feature_importance = {}
        self.is_trained = False
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for modeling with enhanced handling."""
        # Select and order features
        available_features = [f for f in self.feature_list if f in df.columns]
        if len(available_features) != len(self.feature_list):
            missing = set(self.feature_list) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        X = df[available_features].copy()
        
        # Handle missing values
        if not self.feature_medians:
            self.feature_medians = X.median(numeric_only=True).to_dict()
        
        X_clean = X.fillna(self.feature_medians)
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(self.feature_medians)
        
        return X_clean.values
    
    def _prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare target variable."""
        if 'hit_hr' in df.columns:
            return df['hit_hr'].values
        elif 'home_runs' in df.columns:
            return (df['home_runs'] > 0).astype(int).values
        else:
            raise ValueError("No target variable found (hit_hr or home_runs)")
    
    def train(self, df: pd.DataFrame) -> None:
        """Train all models in the component with enhanced configuration."""
        logger.info(f"Training model component with {len(self.feature_list)} features")
        
        X = self._prepare_features(df)
        y = self._prepare_target(df)
        
        # Check class balance
        pos_count = int(y.sum())
        neg_count = int(len(y) - pos_count)
        balance_ratio = pos_count / max(1, neg_count)
        logger.info(f"Class balance: {pos_count} positive, {neg_count} negative (ratio: {balance_ratio:.3f})")
        
        # Scale features for logistic regression
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Logistic Regression
        self.models['logistic'] = LogisticRegression(
            max_iter=1000, 
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            solver='liblinear'  # Better for small datasets
        )
        self.models['logistic'].fit(X_scaled, y)
        
        # Train Random Forest with optimized parameters
        self.models['rf'] = RandomForestClassifier(
            n_estimators=300,  # Reduced for speed
            max_depth=12,      # Limited to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.models['rf'].fit(X, y)
        
        # Store feature importance
        self.feature_importance['rf'] = dict(zip(
            [f for f in self.feature_list if f in self.scaler.feature_names_in_] if hasattr(self.scaler, 'feature_names_in_') else self.feature_list[:len(self.models['rf'].feature_importances_)],
            self.models['rf'].feature_importances_
        ))
        
        # Train XGBoost with enhanced parameters
        scale_pos_weight = max(1.0, neg_count / max(1, pos_count))
        
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective='binary:logistic',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr'
        )
        self.models['xgb'].fit(X, y)
        
        # Store XGBoost feature importance
        if hasattr(self.models['xgb'], 'feature_importances_'):
            self.feature_importance['xgb'] = dict(zip(
                self.feature_list[:len(self.models['xgb'].feature_importances_)],
                self.models['xgb'].feature_importances_
            ))
        
        self.is_trained = True
        logger.info("Model component training completed")
    
    def predict_proba(self, df: pd.DataFrame, model_name: str = 'xgb') -> np.ndarray:
        """Generate predictions using specified model."""
        if not self.is_trained:
            raise ValueError("Model component must be trained before making predictions")
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Available: {list(self.models.keys())}")
        
        X = self._prepare_features(df)
        
        # Get predictions based on model type
        if model_name == 'logistic':
            X_scaled = self.scaler.transform(X)
            probs = self.models['logistic'].predict_proba(X_scaled)[:, 1]
        elif model_name == 'rf':
            probs = self.models['rf'].predict_proba(X)[:, 1]
        else:  # xgb
            try:
                probs = self.models['xgb'].predict_proba(X)[:, 1]
            except Exception:
                probs = self.models['xgb'].predict(X)
        
        # Apply calibration if available
        if self.calibrator is not None:
            probs = self.calibrator.predict(np.clip(probs, 1e-6, 1-1e-6))
            probs = np.clip(probs, 0, 1)
        
        return probs
    
    def fit_calibration(self, df: pd.DataFrame, model_name: str = 'xgb') -> None:
        """Fit isotonic calibration."""
        if not self.is_trained:
            raise ValueError("Model must be trained before calibration")
        
        y_true = self._prepare_target(df)
        y_prob = self.predict_proba(df, model_name)
        
        # Only calibrate if we have enough diverse predictions
        if len(np.unique(y_prob.round(3))) > 10:
            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            self.calibrator.fit(np.clip(y_prob, 1e-6, 1-1e-6), y_true)
            logger.info(f"Calibration fitted for {model_name} model")
        else:
            logger.warning(f"Skipping calibration due to insufficient prediction diversity")

class EnhancedDualModelSystem:
    """Enhanced dual model system with proper data splitting and validation."""
    
    def __init__(self, model_dir: str = None, splitting_strategy: str = 'time_based'):
        self.model_dir = Path(model_dir or str(Path("saved_models_pregame")))
        self.model_dir.mkdir(exist_ok=True)
        
        # Model components
        self.core_model = None
        self.enhanced_model = None
        
        # Feature management
        self.feature_selector = FeatureSelector()
        self.available_features = {}
        
        # Data splitting
        self.splitting_strategy = splitting_strategy
        self.splitter = DataSplitter()
        
        # Validation
        self.metadata = {}
        self.validator = ModelValidator()
        self.training_history = []
        
        logger.info(f"Enhanced dual model system initialized: {self.model_dir}")
        logger.info(f"Splitting strategy: {splitting_strategy}")
    
    def fit(self, train_df: pd.DataFrame, 
            splitting_strategy: str = None,
            test_size: float = 0.2,
            val_size: float = 0.1,
            gap_days: int = 0,
            test_seasons: List[int] = None,
            cross_validate: bool = True,
            cv_folds: int = 5) -> Dict[str, Any]:
        """
        Enhanced training with proper data splitting and validation.
        
        Args:
            train_df: Full dataset for training
            splitting_strategy: 'time_based', 'random', or 'seasonal'
            test_size: Proportion for test set (time_based/random)
            val_size: Proportion for validation set
            gap_days: Days gap between splits (time_based only)
            test_seasons: Specific seasons for test (seasonal only)
            cross_validate: Whether to perform cross-validation
            cv_folds: Number of cross-validation folds
        
        Returns:
            Training results and metrics
        """
        logger.info("="*60)
        logger.info("TRAINING ENHANCED DUAL MODEL SYSTEM")
        logger.info("="*60)
        
        splitting_strategy = splitting_strategy or self.splitting_strategy
        
        # Perform data splitting
        if splitting_strategy == 'seasonal' and test_seasons:
            train_data, val_data, test_data = self.splitter.seasonal_split(
                train_df, test_seasons=test_seasons
            )
        elif splitting_strategy == 'time_based':
            train_data, val_data, test_data = self.splitter.time_based_split(
                train_df, 
                train_ratio=1-test_size-val_size,
                val_ratio=val_size/(1-test_size),
                gap_days=gap_days
            )
        else:  # random
            train_data, val_data, test_data = self.splitter.random_split(
                train_df,
                train_ratio=1-test_size-val_size,
                val_ratio=val_size/(1-test_size)
            )
        
        # Analyze available features
        self.available_features = self.feature_selector.identify_available_features(train_data)
        enhanced_assessment = self.feature_selector.assess_enhanced_feature_coverage(train_data)
        
        logger.info(f"Core features available: {len(self.available_features['core'])}")
        logger.info(f"Enhanced features available: {len(self.available_features['enhanced'])}")
        logger.info(f"Enhanced feature assessment: {enhanced_assessment['reason']}")
        
        training_results = {
            'split_strategy': splitting_strategy,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'core_features': self.available_features['core'],
            'enhanced_features': self.available_features['enhanced'],
            'enhanced_assessment': enhanced_assessment
        }
        
        # Train core model
        logger.info("\nTraining CORE model...")
        self.core_model = BaseModelComponent(self.available_features['core'])
        self.core_model.train(train_data)
        
        # Evaluate core model on validation set
        if not val_data.empty:
            core_val_metrics = self._evaluate_model_component(
                self.core_model, val_data, "Core_Validation"
            )
            training_results['core_val_metrics'] = core_val_metrics
        
        # Train enhanced model if we have more features than core
        if len(self.available_features['enhanced']) > len(self.available_features['core']):
            logger.info("\nTraining ENHANCED model...")
            self.enhanced_model = BaseModelComponent(self.available_features['enhanced'])
            self.enhanced_model.train(train_data)
            
            # Evaluate enhanced model on validation set
            if not val_data.empty:
                enhanced_val_metrics = self._evaluate_model_component(
                    self.enhanced_model, val_data, "Enhanced_Validation"
                )
                training_results['enhanced_val_metrics'] = enhanced_val_metrics
        else:
            logger.info(f"\nSkipping enhanced model: No additional features beyond core ({len(self.available_features['enhanced'])} vs {len(self.available_features['core'])})")
            self.enhanced_model = None
        
        # Fit calibration on validation data
        if not val_data.empty:
            self.fit_calibration(val_data)
        
        # Evaluate on test set
        if not test_data.empty:
            test_results = self.evaluate_comprehensive(test_data, "Test")
            training_results['test_metrics'] = test_results
        
        # Cross-validation if requested
        if cross_validate and len(train_df) > 1000:  # Only for reasonable dataset sizes
            cv_results = self._perform_cross_validation(train_df, n_splits=cv_folds)
            training_results['cv_results'] = cv_results
        
        # Store training metadata
        self.metadata = {
            'train_timestamp': datetime.utcnow().isoformat() + 'Z',
            'splitting_strategy': splitting_strategy,
            'has_core_model': self.core_model is not None,
            'has_enhanced_model': self.enhanced_model is not None,
            'training_results': training_results
        }
        
        # Store in training history
        self.training_history.append(training_results)
        
        logger.info("Enhanced dual model training completed")
        return training_results
    
    def _evaluate_model_component(self, model_component: BaseModelComponent, 
                                 df: pd.DataFrame, label: str) -> Dict[str, Any]:
        """Evaluate a single model component."""
        try:
            y_true = model_component._prepare_target(df)
            results = {}
            
            for model_name in ['xgb', 'rf', 'logistic']:
                if model_name in model_component.models:
                    y_prob = model_component.predict_proba(df, model_name)
                    y_pred = (y_prob >= 0.5).astype(int)
                    
                    metrics = self.validator.evaluate_model_performance(
                        y_true, y_pred, y_prob, f"{label}_{model_name}"
                    )
                    
                    # Add calibration metrics
                    cal_metrics = self.validator.calculate_calibration_metrics(y_true, y_prob)
                    metrics.update({f"cal_{k}": v for k, v in cal_metrics.items() if k != 'calibration_data'})
                    
                    results[model_name] = metrics
            
            return results
            
        except Exception as e:
            logger.error(f"Model component evaluation failed: {e}")
            return {}
    
    def _perform_cross_validation(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        try:
            # Use TimeSeriesSplit for time-aware CV
            if 'date' in df.columns:
                df_sorted = df.sort_values('date')
                tscv = TimeSeriesSplit(n_splits=n_splits)
                cv_indices = list(tscv.split(df_sorted))
            else:
                # Fallback to stratified CV
                target = df['hit_hr'] if 'hit_hr' in df.columns else (df['home_runs'] > 0).astype(int)
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
                cv_indices = list(skf.split(df, target))
                df_sorted = df
            
            cv_results = {'core': [], 'enhanced': []}
            
            for fold, (train_idx, val_idx) in enumerate(cv_indices):
                logger.info(f"CV Fold {fold + 1}/{n_splits}")
                
                train_fold = df_sorted.iloc[train_idx]
                val_fold = df_sorted.iloc[val_idx]
                
                # Train and evaluate core model
                temp_core = BaseModelComponent(self.available_features['core'])
                temp_core.train(train_fold)
                core_metrics = self._evaluate_model_component(temp_core, val_fold, f"CV_Core_Fold{fold}")
                cv_results['core'].append(core_metrics)
                
                # Train and evaluate enhanced model if applicable
                if self.enhanced_model is not None:
                    temp_enhanced = BaseModelComponent(self.available_features['enhanced'])
                    temp_enhanced.train(train_fold)
                    enhanced_metrics = self._evaluate_model_component(temp_enhanced, val_fold, f"CV_Enhanced_Fold{fold}")
                    cv_results['enhanced'].append(enhanced_metrics)
            
            # Aggregate CV results
            aggregated = self._aggregate_cv_results(cv_results)
            logger.info("Cross-validation completed")
            return aggregated
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {}
    
    def _aggregate_cv_results(self, cv_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        aggregated = {}
        
        for model_type, results_list in cv_results.items():
            if not results_list:
                continue
                
            aggregated[model_type] = {}
            
            for model_name in ['xgb', 'rf', 'logistic']:
                model_metrics = []
                for fold_results in results_list:
                    if model_name in fold_results:
                        model_metrics.append(fold_results[model_name])
                
                if model_metrics:
                    # Calculate mean and std for each metric
                    all_metrics = {}
                    for metric in model_metrics[0].keys():
                        values = [m[metric] for m in model_metrics if not np.isnan(m.get(metric, np.nan))]
                        if values:
                            all_metrics[f"{metric}_mean"] = np.mean(values)
                            all_metrics[f"{metric}_std"] = np.std(values)
                    
                    aggregated[model_type][model_name] = all_metrics
        
        return aggregated
    
    def predict_proba(self, df: pd.DataFrame, model_name: str = 'xgb', 
                     prefer_enhanced: bool = True) -> np.ndarray:
        """Generate predictions using the best available model."""
        if self.core_model is None:
            raise ValueError("No trained models available")
        
        # Determine which model to use
        use_enhanced = (
            prefer_enhanced and 
            self.enhanced_model is not None and
            self._can_use_enhanced_model(df)
        )
        
        if use_enhanced:
            logger.info(f"Using enhanced model ({len(self.available_features['enhanced'])} features)")
            predictions = self.enhanced_model.predict_proba(df, model_name)
        else:
            logger.info(f"Using core model ({len(self.available_features['core'])} features)")
            predictions = self.core_model.predict_proba(df, model_name)
        
        # Validate predictions
        if not self.validator.validate_predictions(predictions):
            logger.error("Invalid predictions generated")
            # Fallback to core model if enhanced model failed
            if use_enhanced and self.core_model is not None:
                logger.info("Falling back to core model")
                predictions = self.core_model.predict_proba(df, model_name)
            else:
                raise ValueError("Model predictions failed validation")
        
        return predictions
    
    def predict(self, df: pd.DataFrame, model_name: str = 'xgb', 
                prefer_enhanced: bool = True, threshold: float = 0.15) -> np.ndarray:
        """Generate binary predictions using the best available model.
        
        Args:
            df: Input features
            model_name: Model to use ('xgb', 'rf', 'lgb')
            prefer_enhanced: Whether to prefer enhanced model over core
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(df, model_name, prefer_enhanced)
        # Get probability of positive class (home run)
        if probabilities.ndim == 2:
            probabilities = probabilities[:, 1]
        return (probabilities >= threshold).astype(int)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive metrics for model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        # If probabilities not provided, assume uniform probability of 0.5 for all predictions
        if y_prob is None:
            y_prob = np.where(y_pred == 1, 0.6, 0.4)  # Simple approximation
        
        return ModelValidator.evaluate_model_performance(y_true, y_pred, y_prob)
    
    def _can_use_enhanced_model(self, df: pd.DataFrame) -> bool:
        """Check if enhanced model can be used with given data."""
        if self.enhanced_model is None:
            return False
        
        # Check if all enhanced features are present
        missing_features = [
            col for col in self.available_features['enhanced'] 
            if col not in df.columns
        ]
        
        if missing_features:
            logger.warning(f"Enhanced model requires missing features: {missing_features[:10]}...")
            return False
        
        # No longer requires bat tracking coverage - just check if features are available
        return True
    
    def evaluate_comprehensive(self, df: pd.DataFrame, label: str = "Evaluation") -> Dict[str, Any]:
        """Comprehensive evaluation with multiple metrics."""
        results = {}
        
        try:
            # Evaluate both models if available
            if self.core_model is not None:
                core_results = self._evaluate_model_component(self.core_model, df, f"{label}_Core")
                results['core'] = core_results
            
            if self.enhanced_model is not None:
                enhanced_results = self._evaluate_model_component(self.enhanced_model, df, f"{label}_Enhanced")
                results['enhanced'] = enhanced_results
            
            # System-level evaluation (using best model)
            y_true = self.core_model._prepare_target(df) if self.core_model else None
            if y_true is not None:
                system_probs = self.predict_proba(df)
                system_preds = (system_probs >= 0.5).astype(int)
                
                system_metrics = self.validator.evaluate_model_performance(
                    y_true, system_preds, system_probs, f"{label}_System"
                )
                
                # Add calibration analysis
                cal_metrics = self.validator.calculate_calibration_metrics(y_true, system_probs)
                system_metrics.update({f"cal_{k}": v for k, v in cal_metrics.items() if k != 'calibration_data'})
                
                results['system'] = system_metrics
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            return {}
    
    def fit_calibration(self, val_df: pd.DataFrame, model_name: str = 'xgb') -> None:
        """Fit calibration on validation data."""
        logger.info("Fitting model calibration...")
        
        if self.core_model is not None:
            try:
                self.core_model.fit_calibration(val_df, model_name)
                logger.info("Core model calibration completed")
            except Exception as e:
                logger.warning(f"Core model calibration failed: {e}")
        
        if self.enhanced_model is not None:
            try:
                self.enhanced_model.fit_calibration(val_df, model_name)
                logger.info("Enhanced model calibration completed")
            except Exception as e:
                logger.warning(f"Enhanced model calibration failed: {e}")
    
    def get_feature_importance(self, model_type: str = 'core', algorithm: str = 'xgb') -> Dict[str, float]:
        """Get feature importance from trained models."""
        if model_type == 'core' and self.core_model is not None:
            return self.core_model.feature_importance.get(algorithm, {})
        elif model_type == 'enhanced' and self.enhanced_model is not None:
            return self.enhanced_model.feature_importance.get(algorithm, {})
        else:
            return {}
    
    def save(self, train_years: List[int] = None) -> None:
        """Save the enhanced dual model system."""
        try:
            # Update metadata
            self.metadata.update({
                'train_years': train_years or [],
                'save_timestamp': datetime.utcnow().isoformat() + 'Z',
                'feature_counts': {
                    'core': len(self.available_features.get('core', [])),
                    'enhanced': len(self.available_features.get('enhanced', []))
                }
            })
            
            # Save core model
            if self.core_model is not None:
                core_path = self.model_dir / "core_model.joblib"
                joblib.dump(self.core_model, core_path)
                logger.info(f"Saved core model to {core_path}")
            
            # Save enhanced model
            if self.enhanced_model is not None:
                enhanced_path = self.model_dir / "enhanced_model.joblib"
                joblib.dump(self.enhanced_model, enhanced_path)
                logger.info(f"Saved enhanced model to {enhanced_path}")
            
            # Save metadata and training history
            metadata_path = self.model_dir / "enhanced_dual_model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'training_history': self.training_history,
                    'available_features': self.available_features
                }, f, indent=2, default=str)
            
            logger.info(f"Enhanced dual model system saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            raise
    
    def load(self) -> Dict[str, Any]:
        """Load the enhanced dual model system."""
        try:
            # Load metadata
            metadata_path = self.model_dir / "enhanced_dual_model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    saved_data = json.load(f)
                    self.metadata = saved_data.get('metadata', {})
                    self.training_history = saved_data.get('training_history', [])
                    self.available_features = saved_data.get('available_features', {})
            else:
                # Fallback to original metadata file
                old_metadata_path = self.model_dir / "dual_model_metadata.json"
                if old_metadata_path.exists():
                    with open(old_metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    self.available_features = {
                        'core': self.metadata.get('core_features', []),
                        'enhanced': self.metadata.get('enhanced_features', []),
                        'bat_tracking': self.metadata.get('bat_tracking_features', [])
                    }
            
            # Load core model
            core_path = self.model_dir / "core_model.joblib"
            if core_path.exists():
                self.core_model = joblib.load(core_path)
                logger.info("Loaded core model")
            else:
                logger.warning("Core model file not found")
            
            # Load enhanced model
            enhanced_path = self.model_dir / "enhanced_model.joblib"
            if enhanced_path.exists():
                self.enhanced_model = joblib.load(enhanced_path)
                logger.info("Loaded enhanced model")
            else:
                logger.info("Enhanced model file not found (this is normal if not trained)")
            
            logger.info("Enhanced dual model system loaded successfully")
            return self.metadata
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        info = {
            'core_model_loaded': self.core_model is not None,
            'enhanced_model_loaded': self.enhanced_model is not None,
            'core_features_count': len(self.available_features.get('core', [])),
            'enhanced_features_count': len(self.available_features.get('enhanced', [])),
            'metadata': self.metadata,
            'training_history': self.training_history
        }
        
        # Add model type information
        if self.enhanced_model is not None:
            info['active_model'] = 'enhanced'
            info['active_feature_count'] = info['enhanced_features_count']
        elif self.core_model is not None:
            info['active_model'] = 'core'
            info['active_feature_count'] = info['core_features_count']
        else:
            info['active_model'] = 'none'
            info['active_feature_count'] = 0
        
        return info

# Backward compatibility - replace the original DualModelSystem
DualModelSystem = EnhancedDualModelSystem

# Export enhanced classes
__all__ = [
    'EnhancedDualModelSystem', 'DualModelSystem', 'DataSplitter', 'ModelValidator',
    'BaseModelComponent', 'FeatureSelector'
]