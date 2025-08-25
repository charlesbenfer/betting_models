"""
Step 10: Comparative Analysis and Performance Evaluation
=======================================================

Comprehensive comparison of feature engineering improvements across all 8 steps.
Evaluates model performance, feature importance, and production readiness.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, log_loss
import joblib
from pathlib import Path

# Import our modules
from config import config
from dataset_builder import PregameDatasetBuilder
from modeling import EnhancedDualModelSystem

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for comparative experiments."""
    experiment_name: str
    features_to_include: List[str]
    description: str
    
@dataclass
class PerformanceMetrics:
    """Performance metrics for a single experiment."""
    experiment_name: str
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    log_loss: float
    feature_count: int
    training_time: float
    prediction_time: float

class ComparativeAnalyzer:
    """Comprehensive comparative analysis system."""
    
    def __init__(self, 
                 start_date: str = "2024-04-01", 
                 end_date: str = "2024-08-31", 
                 test_start_date: str = "2024-09-01",
                 test_end_date: str = "2024-09-30",
                 use_cache: bool = False):
        self.start_date = start_date
        self.end_date = end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.use_cache = use_cache
        self.experiments = []
        self.results = []
        self.best_model = None
        self.best_model_info = None
        
    def setup_experiments(self) -> List[ExperimentConfig]:
        """Setup all experimental configurations."""
        
        experiments = [
            # Baseline: Core features only
            ExperimentConfig(
                experiment_name="baseline_core",
                features_to_include=config.CORE_FEATURES,
                description="Baseline with core statistical features only"
            ),
            
            # Step 1: Add matchup features
            ExperimentConfig(
                experiment_name="step1_matchup", 
                features_to_include=config.CORE_FEATURES + config.MATCHUP_FEATURES,
                description="Core + Batter vs Pitcher matchup history"
            ),
            
            # Step 2: Add situational features
            ExperimentConfig(
                experiment_name="step2_situational",
                features_to_include=(config.CORE_FEATURES + config.MATCHUP_FEATURES + 
                                   config.SITUATIONAL_FEATURES),
                description="Previous + Situational context (inning, score, runners)"
            ),
            
            # Step 3: Add weather features
            ExperimentConfig(
                experiment_name="step3_weather",
                features_to_include=(config.CORE_FEATURES + config.MATCHUP_FEATURES + 
                                   config.SITUATIONAL_FEATURES + config.WEATHER_FEATURES),
                description="Previous + Weather impact features"
            ),
            
            # Step 4: Add recent form features
            ExperimentConfig(
                experiment_name="step4_recent_form",
                features_to_include=(config.CORE_FEATURES + config.MATCHUP_FEATURES + 
                                   config.SITUATIONAL_FEATURES + config.WEATHER_FEATURES +
                                   config.RECENT_FORM_FEATURES),
                description="Previous + Recent form with decay functions"
            ),
            
            # Step 5: Add streak/momentum features
            ExperimentConfig(
                experiment_name="step5_streaks",
                features_to_include=(config.CORE_FEATURES + config.MATCHUP_FEATURES + 
                                   config.SITUATIONAL_FEATURES + config.WEATHER_FEATURES +
                                   config.RECENT_FORM_FEATURES + config.STREAK_MOMENTUM_FEATURES),
                description="Previous + Streak and momentum analysis"
            ),
            
            # Step 6: Add ballpark features
            ExperimentConfig(
                experiment_name="step6_ballpark",
                features_to_include=(config.CORE_FEATURES + config.MATCHUP_FEATURES + 
                                   config.SITUATIONAL_FEATURES + config.WEATHER_FEATURES +
                                   config.RECENT_FORM_FEATURES + config.STREAK_MOMENTUM_FEATURES +
                                   config.BALLPARK_FEATURES),
                description="Previous + Ballpark-specific advanced features"
            ),
            
            # Step 7: Add temporal/fatigue features
            ExperimentConfig(
                experiment_name="step7_temporal",
                features_to_include=(config.CORE_FEATURES + config.MATCHUP_FEATURES + 
                                   config.SITUATIONAL_FEATURES + config.WEATHER_FEATURES +
                                   config.RECENT_FORM_FEATURES + config.STREAK_MOMENTUM_FEATURES +
                                   config.BALLPARK_FEATURES + config.TEMPORAL_FATIGUE_FEATURES),
                description="Previous + Time-of-day and fatigue features"
            ),
            
            # Step 8: Add interaction features
            ExperimentConfig(
                experiment_name="step8_interactions",
                features_to_include=(config.CORE_FEATURES + config.MATCHUP_FEATURES + 
                                   config.SITUATIONAL_FEATURES + config.WEATHER_FEATURES +
                                   config.RECENT_FORM_FEATURES + config.STREAK_MOMENTUM_FEATURES +
                                   config.BALLPARK_FEATURES + config.TEMPORAL_FATIGUE_FEATURES +
                                   config.INTERACTION_FEATURES),
                description="All features + Interaction terms"
            ),
            
            # Best subset analysis
            ExperimentConfig(
                experiment_name="optimized_subset",
                features_to_include=[],  # Will be determined by feature selection
                description="Optimized feature subset based on importance"
            )
        ]
        
        self.experiments = experiments
        return experiments
    
    def run_comparative_analysis(self) -> Dict[str, Any]:
        """Run complete comparative analysis."""
        logger.info("="*80)
        logger.info("COMPREHENSIVE COMPARATIVE ANALYSIS")
        logger.info("Evaluating feature engineering improvements across all steps")
        logger.info("="*80)
        
        # Setup experiments
        experiments = self.setup_experiments()
        
        # Build datasets - this will take significant time with 4 years of data
        logger.info("Building training and test datasets...")
        logger.info(f"Training period: {self.start_date} to {self.end_date} (2 months)")
        logger.info(f"Testing period: {self.test_start_date} to {self.test_end_date} (1 month)")
        logger.info("⚠️  Dataset build - should complete in 5-15 minutes")
        
        train_dataset = self._build_dataset(self.start_date, self.end_date, "training")
        test_dataset = self._build_dataset(self.test_start_date, self.test_end_date, "testing")
        
        if train_dataset.empty or test_dataset.empty:
            logger.error("Failed to build datasets")
            return {}
        
        # Run experiments with progress tracking
        results = []
        feature_importance_data = {}
        total_experiments = len(experiments) - 1  # Exclude optimized subset initially
        
        logger.info(f"\\n🚀 Starting {total_experiments} experiments with 2 months of training data")
        logger.info("Each experiment should take 2-5 minutes with smaller datasets")
        
        for i, experiment in enumerate(experiments[:-1], 1):  # Skip optimized subset for now
            logger.info(f"\\n{'='*60}")
            logger.info(f"EXPERIMENT {i}/{total_experiments}: {experiment.experiment_name.upper()}")
            logger.info(f"{'='*60}")
            logger.info(f"Description: {experiment.description}")
            logger.info(f"Progress: {i}/{total_experiments} ({i/total_experiments*100:.1f}%)")
            
            experiment_start = datetime.now()
            result = self._run_single_experiment(
                experiment, train_dataset, test_dataset
            )
            experiment_time = (datetime.now() - experiment_start).total_seconds()
            
            if result:
                results.append(result)
                feature_importance_data[experiment.experiment_name] = result.get('feature_importance', {})
                logger.info(f"✅ Experiment {i} completed in {experiment_time/60:.1f} minutes")
                
                # Estimated time remaining
                avg_time_per_exp = sum(r['total_time'] for r in results) / len(results)
                remaining_experiments = total_experiments - i
                eta_minutes = (remaining_experiments * avg_time_per_exp) / 60
                logger.info(f"⏱️  Estimated time remaining: {eta_minutes:.1f} minutes")
            else:
                logger.warning(f"❌ Experiment {i} failed")
        
        # Run optimized subset experiment
        if results:
            logger.info(f"\\n{'='*60}")
            logger.info("OPTIMIZED SUBSET EXPERIMENT")
            logger.info(f"{'='*60}")
            
            optimized_result = self._run_optimized_experiment(
                train_dataset, test_dataset, feature_importance_data
            )
            if optimized_result:
                results.append(optimized_result)
        
        # Generate comprehensive analysis
        analysis_results = self._generate_analysis_report(results, feature_importance_data)
        
        # Save results with timestamp for large analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_results(results, analysis_results, timestamp)
        
        # Save the best performing model
        self._save_best_model(results, analysis_results)
        
        logger.info("\\n" + "="*80)
        logger.info("✅ COMPREHENSIVE 4-YEAR COMPARATIVE ANALYSIS COMPLETED!")
        logger.info(f"Training data: {self.start_date} to {self.end_date} (4 years)")
        logger.info(f"Test data: {self.test_start_date} to {self.test_end_date} (2 months)")
        logger.info(f"Results saved with timestamp: {timestamp}")
        logger.info("Check comparative_analysis_results_[timestamp].json for detailed results")
        logger.info("="*80)
        
        return analysis_results
    
    def _build_dataset(self, start_date: str, end_date: str, dataset_type: str) -> pd.DataFrame:
        """Build dataset for specified date range."""
        logger.info(f"Building {dataset_type} dataset ({start_date} to {end_date})...")
        
        try:
            builder = PregameDatasetBuilder(
                start_date=start_date,
                end_date=end_date
            )
            
            dataset = builder.build_dataset(force_rebuild=not self.use_cache)
            
            if dataset.empty:
                logger.error(f"{dataset_type.title()} dataset is empty")
                return pd.DataFrame()
            
            logger.info(f"{dataset_type.title()} dataset: {len(dataset)} rows, {len(dataset.columns)} columns")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to build {dataset_type} dataset: {e}")
            return pd.DataFrame()
    
    def _train_xgboost_model(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model optimized for probability accuracy (log loss)."""
        logger.info("Training XGBoost model with log loss optimization...")
        
        # Calculate scale_pos_weight for imbalanced data
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos_weight = max(1.0, neg_count / max(1, pos_count))
        base_rate = pos_count / (pos_count + neg_count)
        logger.info(f"Class balance - Positive rate: {base_rate:.3%}, Scale weight: {scale_pos_weight:.2f}")
        
        # Create model optimized for probability accuracy
        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,  # Lower learning rate for better calibration
            max_depth=5,  # Slightly less depth to prevent overfitting
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,  # More L1 regularization
            reg_lambda=3.0,  # More L2 regularization
            gamma=1.0,  # Minimum loss reduction for split (helps calibration)
            objective='binary:logistic',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',  # Changed from aucpr to logloss
            base_score=base_rate  # Initialize with base rate for better calibration
        )
        
        # Train model with early stopping based on log loss
        try:
            # Try with newer XGBoost API
            from xgboost import callback
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[callback.EarlyStopping(rounds=50)],
                verbose=False
            )
        except (ImportError, TypeError):
            # Fallback for older XGBoost versions or simpler training
            xgb_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = xgb_model.predict(X_val)
        y_val_prob = xgb_model.predict_proba(X_val)[:, 1]
        
        val_metrics = {
            'roc_auc': roc_auc_score(y_val, y_val_prob),
            'log_loss': log_loss(y_val, y_val_prob),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0),
            'accuracy': accuracy_score(y_val, y_val_pred),
            'brier_score': np.mean((y_val_prob - y_val) ** 2),  # Lower is better
            'mean_predicted_prob': y_val_prob.mean(),
            'actual_positive_rate': y_val.mean()
        }
        
        logger.info(f"XGBoost validation - ROC-AUC: {val_metrics['roc_auc']:.4f}, Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"  Mean predicted: {val_metrics['mean_predicted_prob']:.3%} vs Actual rate: {val_metrics['actual_positive_rate']:.3%}")
        
        # Return model parameters for consistency
        params = {
            'n_estimators': 400,
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 2.0,
            'reg_lambda': 3.0,
            'gamma': 1.0
        }
        
        return xgb_model, val_metrics, params
    
    def _train_lightgbm_model(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model optimized for probability accuracy (log loss)."""
        logger.info("Training LightGBM model with log loss optimization...")
        
        # Calculate scale_pos_weight for imbalanced data
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos_weight = max(1.0, neg_count / max(1, pos_count))
        base_rate = pos_count / (pos_count + neg_count)
        logger.info(f"Class balance - Positive rate: {base_rate:.3%}, Scale weight: {scale_pos_weight:.2f}")
        
        # Create model optimized for probability accuracy
        lgb_model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,  # Lower learning rate for better calibration
            max_depth=5,  # Less depth to prevent overfitting
            num_leaves=31,  # 2^max_depth - 1
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,  # More L1 regularization
            reg_lambda=3.0,  # More L2 regularization
            min_child_samples=30,  # More samples required for splits
            min_split_gain=0.1,  # Minimum gain for splits (helps calibration)
            objective='binary',
            metric='binary_logloss',  # Changed from auc to logloss
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
            force_col_wise=True,
            boost_from_average=True  # Start from base rate
        )
        
        # Train model with early stopping based on log loss
        try:
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        except (ImportError, TypeError, AttributeError):
            # Fallback for compatibility issues
            lgb_model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = lgb_model.predict(X_val)
        y_val_prob = lgb_model.predict_proba(X_val)[:, 1]
        
        val_metrics = {
            'roc_auc': roc_auc_score(y_val, y_val_prob),
            'log_loss': log_loss(y_val, y_val_prob),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0),
            'accuracy': accuracy_score(y_val, y_val_pred),
            'brier_score': np.mean((y_val_prob - y_val) ** 2),  # Lower is better
            'mean_predicted_prob': y_val_prob.mean(),
            'actual_positive_rate': y_val.mean()
        }
        
        logger.info(f"LightGBM validation - ROC-AUC: {val_metrics['roc_auc']:.4f}, Log Loss: {val_metrics['log_loss']:.4f}")
        logger.info(f"  Mean predicted: {val_metrics['mean_predicted_prob']:.3%} vs Actual rate: {val_metrics['actual_positive_rate']:.3%}")
        
        # Return model parameters for consistency
        params = {
            'n_estimators': 400,
            'learning_rate': 0.03,
            'max_depth': 5,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 2.0,
            'reg_lambda': 3.0,
            'min_child_samples': 30,
            'min_split_gain': 0.1
        }
        
        return lgb_model, val_metrics, params
    
    def _run_single_experiment(self, 
                             experiment: ExperimentConfig,
                             train_dataset: pd.DataFrame,
                             test_dataset: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run a single experiment with both XGBoost and LightGBM."""
        try:
            start_time = datetime.now()
            
            # Filter features
            available_features = [f for f in experiment.features_to_include 
                                if f in train_dataset.columns]
            missing_features = [f for f in experiment.features_to_include 
                              if f not in train_dataset.columns]
            
            logger.info(f"Available features: {len(available_features)}/{len(experiment.features_to_include)}")
            
            # Enhanced error reporting for missing features
            if missing_features:
                missing_pct = len(missing_features) / len(experiment.features_to_include) * 100
                logger.warning(f"Missing {len(missing_features)} features ({missing_pct:.1f}%)")
                if len(missing_features) <= 10:
                    logger.warning(f"Missing features: {missing_features}")
                else:
                    logger.warning(f"First 10 missing: {missing_features[:10]}...")
                
                # Fail loudly if too many features are missing
                if missing_pct > 50:
                    raise ValueError(f"Too many features missing ({missing_pct:.1f}%). Check feature engineering pipeline.")
            
            if len(available_features) < 5:
                raise ValueError(f"Only {len(available_features)} features available. Minimum 5 required.")
                return None
            
            # Prepare datasets with only available features + target
            feature_cols = available_features + ['hit_hr']
            train_data = train_dataset[feature_cols].copy()
            test_data = test_dataset[feature_cols].copy()
            
            # Check for sufficient data
            if len(train_data) < 100 or len(test_data) < 20:
                logger.warning("Insufficient data for reliable testing")
                return None
            
            # Split training data into train and validation
            X_train = train_data.drop('hit_hr', axis=1).values
            y_train = train_data['hit_hr'].values
            X_test = test_data.drop('hit_hr', axis=1).values
            y_test = test_data['hit_hr'].values
            
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.15, random_state=config.RANDOM_STATE, stratify=y_train
            )
            
            training_start = datetime.now()
            
            # Train both XGBoost and LightGBM
            xgb_model, xgb_val_metrics, xgb_params = self._train_xgboost_model(
                X_train_split, y_train_split, X_val_split, y_val_split
            )
            
            lgb_model, lgb_val_metrics, lgb_params = self._train_lightgbm_model(
                X_train_split, y_train_split, X_val_split, y_val_split
            )
            
            training_time = (datetime.now() - training_start).total_seconds()
            
            # Test both models on holdout set
            prediction_start = datetime.now()
            
            # XGBoost test metrics
            xgb_test_pred = xgb_model.predict(X_test)
            xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
            xgb_test_metrics = {
                'roc_auc': roc_auc_score(y_test, xgb_test_prob),
                'precision': precision_score(y_test, xgb_test_pred, zero_division=0),
                'recall': recall_score(y_test, xgb_test_pred, zero_division=0),
                'f1': f1_score(y_test, xgb_test_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, xgb_test_pred),
                'log_loss': log_loss(y_test, xgb_test_prob),
                'brier_score': np.mean((xgb_test_prob - y_test) ** 2),
                'mean_predicted_prob': xgb_test_prob.mean(),
                'actual_positive_rate': y_test.mean()
            }
            
            # LightGBM test metrics
            lgb_test_pred = lgb_model.predict(X_test)
            lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]
            lgb_test_metrics = {
                'roc_auc': roc_auc_score(y_test, lgb_test_prob),
                'precision': precision_score(y_test, lgb_test_pred, zero_division=0),
                'recall': recall_score(y_test, lgb_test_pred, zero_division=0),
                'f1': f1_score(y_test, lgb_test_pred, zero_division=0),
                'accuracy': accuracy_score(y_test, lgb_test_pred),
                'log_loss': log_loss(y_test, lgb_test_prob),
                'brier_score': np.mean((lgb_test_prob - y_test) ** 2),
                'mean_predicted_prob': lgb_test_prob.mean(),
                'actual_positive_rate': y_test.mean()
            }
            
            prediction_time = (datetime.now() - prediction_start).total_seconds()
            
            # Determine best model based on composite score (ROC-AUC and log loss)
            # Better log loss is more important for betting probability accuracy
            xgb_score = xgb_test_metrics['roc_auc'] - (xgb_test_metrics['log_loss'] * 0.5)  # Penalize higher log loss
            lgb_score = lgb_test_metrics['roc_auc'] - (lgb_test_metrics['log_loss'] * 0.5)
            
            logger.info(f"Model selection scores - XGBoost: {xgb_score:.4f}, LightGBM: {lgb_score:.4f}")
            
            if xgb_score >= lgb_score:
                best_model = xgb_model
                best_model_name = 'XGBoost'
                best_test_metrics = xgb_test_metrics
                best_val_metrics = xgb_val_metrics
                best_params = xgb_params
            else:
                best_model = lgb_model
                best_model_name = 'LightGBM'
                best_test_metrics = lgb_test_metrics
                best_val_metrics = lgb_val_metrics
                best_params = lgb_params
            
            # Get feature importance from best model
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(available_features, best_model.feature_importances_))
            else:
                feature_importance = {}
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Results:")
            logger.info(f"  Best Model: {best_model_name}")
            logger.info(f"  XGBoost - ROC-AUC: {xgb_test_metrics['roc_auc']:.4f}, Log Loss: {xgb_test_metrics['log_loss']:.4f}")
            logger.info(f"  LightGBM - ROC-AUC: {lgb_test_metrics['roc_auc']:.4f}, Log Loss: {lgb_test_metrics['log_loss']:.4f}")
            logger.info(f"  Best - ROC-AUC: {best_test_metrics['roc_auc']:.4f}, Log Loss: {best_test_metrics['log_loss']:.4f}")
            logger.info(f"  Predicted vs Actual: {best_test_metrics.get('mean_predicted_prob', 0):.3%} vs {best_test_metrics.get('actual_positive_rate', 0):.3%}")
            logger.info(f"  Precision: {best_test_metrics['precision']:.4f}")
            logger.info(f"  Recall: {best_test_metrics['recall']:.4f}")
            logger.info(f"  F1-Score: {best_test_metrics['f1']:.4f}")
            logger.info(f"  Features: {len(available_features)}")
            logger.info(f"  Training time: {training_time:.2f}s")
            logger.info(f"  Prediction time: {prediction_time:.4f}s")
            
            # Check if this is the best model overall (using composite score)
            current_composite_score = best_test_metrics['roc_auc'] - (best_test_metrics['log_loss'] * 0.5)
            best_composite_score = 0
            if self.best_model_info:
                best_composite_score = self.best_model_info.get('roc_auc', 0) - (self.best_model_info.get('log_loss', 1.0) * 0.5)
            
            if self.best_model is None or current_composite_score > best_composite_score:
                self.best_model = best_model
                self.best_model_info = {
                    'experiment_name': experiment.experiment_name,
                    'model_type': best_model_name,
                    'roc_auc': best_test_metrics['roc_auc'],
                    'log_loss': best_test_metrics['log_loss'],
                    'composite_score': current_composite_score,
                    'mean_predicted_prob': best_test_metrics['mean_predicted_prob'],
                    'actual_positive_rate': best_test_metrics['actual_positive_rate'],
                    'features': available_features,
                    'params': best_params
                }
                logger.info(f"  🏆 New best overall model! ({best_model_name})")
                logger.info(f"      ROC-AUC: {best_test_metrics['roc_auc']:.4f}, Log Loss: {best_test_metrics['log_loss']:.4f}")
                logger.info(f"      Composite Score: {current_composite_score:.4f}")
                logger.info(f"      Calibration: {best_test_metrics['mean_predicted_prob']:.3%} vs {best_test_metrics['actual_positive_rate']:.3%}")
            
            return {
                'experiment_name': experiment.experiment_name,
                'description': experiment.description,
                'metrics': best_test_metrics,
                'xgb_metrics': xgb_test_metrics,
                'lgb_metrics': lgb_test_metrics,
                'best_model_name': best_model_name,
                'feature_count': len(available_features),
                'training_time': training_time,
                'prediction_time': prediction_time,
                'total_time': total_time,
                'available_features': available_features,
                'missing_features': missing_features,
                'feature_importance': feature_importance,
                'validation_metrics': best_val_metrics,
                'xgb_validation_metrics': xgb_val_metrics,
                'lgb_validation_metrics': lgb_val_metrics,
                'calibration_info': {
                    'mean_predicted': best_test_metrics['mean_predicted_prob'],
                    'actual_rate': best_test_metrics['actual_positive_rate'],
                    'brier_score': best_test_metrics['brier_score']
                },
                'best_params': best_params,
                'model': best_model  # Store the actual model
            }
            
        except Exception as e:
            logger.error(f"Experiment {experiment.experiment_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_optimized_experiment(self,
                                train_dataset: pd.DataFrame,
                                test_dataset: pd.DataFrame,
                                feature_importance_data: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
        """Run experiment with optimized feature subset."""
        try:
            # Aggregate feature importance across all experiments
            feature_scores = {}
            
            for exp_name, importance_dict in feature_importance_data.items():
                for feature, score in importance_dict.items():
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(score)
            
            # Calculate average importance
            avg_importance = {
                feature: np.mean(scores) 
                for feature, scores in feature_scores.items()
            }
            
            # Select top features (e.g., top 50 or features with importance > threshold)
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 50 features or those with importance > 0.001
            selected_features = []
            for feature, importance in sorted_features:
                if len(selected_features) >= 50:
                    break
                if importance > 0.001 and feature in train_dataset.columns:
                    selected_features.append(feature)
            
            logger.info(f"Selected {len(selected_features)} top features for optimization")
            
            # Create optimized experiment config
            optimized_experiment = ExperimentConfig(
                experiment_name="optimized_subset",
                features_to_include=selected_features,
                description=f"Optimized subset of top {len(selected_features)} features"
            )
            
            # Run the experiment
            return self._run_single_experiment(
                optimized_experiment, train_dataset, test_dataset
            )
            
        except Exception as e:
            logger.error(f"Optimized experiment failed: {e}")
            return None
    
    def _get_feature_importance(self, 
                              model_system: EnhancedDualModelSystem,
                              features: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        try:
            # Get feature importance from the best model
            if hasattr(model_system, 'best_model') and model_system.best_model:
                model = model_system.best_model
                
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models (XGBoost, Random Forest)
                    importance_values = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importance_values = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
                else:
                    # Fallback: equal importance
                    importance_values = np.ones(len(features)) / len(features)
                
                # Ensure we have the right number of importance values
                if len(importance_values) != len(features):
                    logger.warning(f"Feature importance length mismatch: {len(importance_values)} vs {len(features)}")
                    importance_values = np.ones(len(features)) / len(features)
                
                return dict(zip(features, importance_values))
            else:
                # No trained model available
                return {feature: 1.0/len(features) for feature in features}
                
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {feature: 1.0/len(features) for feature in features}
    
    def _generate_analysis_report(self,
                                results: List[Dict[str, Any]],
                                feature_importance_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        
        if not results:
            return {"error": "No successful experiments to analyze"}
        
        # Validate that experiments have different feature counts
        feature_counts = [r['feature_count'] for r in results]
        unique_counts = set(feature_counts)
        
        if len(unique_counts) < len(results) * 0.7:  # At least 70% should be unique
            logger.warning(f"⚠️  Feature count validation issue: Only {len(unique_counts)} unique feature counts across {len(results)} experiments")
            logger.warning(f"Feature counts by experiment: {feature_counts}")
            
            # Check if all experiments have the same feature count (major issue)
            if len(unique_counts) == 1:
                logger.error("❌ CRITICAL: All experiments using the same number of features!")
                logger.error("This indicates the feature engineering pipeline is not working correctly.")
                logger.error("Please check that all feature modules are generating their features properly.")
        
        logger.info("\\n" + "="*60)
        logger.info("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        logger.info("="*60)
        
        # Performance comparison
        performance_comparison = self._analyze_performance_progression(results)
        
        # Feature impact analysis
        feature_impact = self._analyze_feature_impact(results)
        
        # Efficiency analysis
        efficiency_analysis = self._analyze_efficiency(results)
        
        # ROI analysis
        roi_analysis = self._analyze_roi(results)
        
        # Best practices and recommendations
        recommendations = self._generate_recommendations(results, feature_importance_data)
        
        analysis = {
            'summary': {
                'total_experiments': len(results),
                'best_performance': performance_comparison['best_experiment'],
                'largest_improvement': performance_comparison['largest_improvement'],
                'feature_engineering_roi': roi_analysis['overall_roi']
            },
            'performance_progression': performance_comparison,
            'feature_impact_analysis': feature_impact,
            'efficiency_analysis': efficiency_analysis,
            'roi_analysis': roi_analysis,
            'recommendations': recommendations,
            'detailed_results': results
        }
        
        # Print summary
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _analyze_performance_progression(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance progression across experiments."""
        
        # Extract key metrics
        experiments = []
        for result in results:
            experiments.append({
                'name': result['experiment_name'],
                'description': result['description'],
                'roc_auc': result['metrics'].get('roc_auc', 0),
                'precision': result['metrics'].get('precision', 0),
                'recall': result['metrics'].get('recall', 0),
                'f1_score': result['metrics'].get('f1', 0),
                'feature_count': result['feature_count']
            })
        
        # Find best experiment
        best_experiment = max(experiments, key=lambda x: x['roc_auc'])
        
        # Calculate improvements
        baseline = experiments[0] if experiments else {'roc_auc': 0}
        improvements = []
        
        for exp in experiments[1:]:
            improvement = {
                'experiment': exp['name'],
                'roc_auc_improvement': exp['roc_auc'] - baseline['roc_auc'],
                'relative_improvement': (exp['roc_auc'] - baseline['roc_auc']) / max(baseline['roc_auc'], 0.01) * 100
            }
            improvements.append(improvement)
        
        # Find largest single-step improvement
        largest_improvement = {'step': 'none', 'improvement': 0}
        for i in range(1, len(experiments)):
            step_improvement = experiments[i]['roc_auc'] - experiments[i-1]['roc_auc']
            if step_improvement > largest_improvement['improvement']:
                largest_improvement = {
                    'step': experiments[i]['name'],
                    'improvement': step_improvement,
                    'relative_improvement': step_improvement / max(experiments[i-1]['roc_auc'], 0.01) * 100
                }
        
        return {
            'experiments': experiments,
            'best_experiment': best_experiment,
            'improvements': improvements,
            'largest_improvement': largest_improvement,
            'overall_improvement': {
                'absolute': best_experiment['roc_auc'] - baseline['roc_auc'],
                'relative': (best_experiment['roc_auc'] - baseline['roc_auc']) / max(baseline['roc_auc'], 0.01) * 100
            }
        }
    
    def _analyze_feature_impact(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the impact of different feature categories."""
        
        feature_categories = {
            'matchup': config.MATCHUP_FEATURES,
            'situational': config.SITUATIONAL_FEATURES,
            'weather': config.WEATHER_FEATURES,
            'recent_form': config.RECENT_FORM_FEATURES,
            'streak_momentum': config.STREAK_MOMENTUM_FEATURES,
            'ballpark': config.BALLPARK_FEATURES,
            'temporal_fatigue': config.TEMPORAL_FATIGUE_FEATURES,
            'interactions': config.INTERACTION_FEATURES
        }
        
        category_impact = {}
        
        # Map experiments to feature categories
        experiment_mapping = {
            'baseline_core': [],
            'step1_matchup': ['matchup'],
            'step2_situational': ['matchup', 'situational'],
            'step3_weather': ['matchup', 'situational', 'weather'],
            'step4_recent_form': ['matchup', 'situational', 'weather', 'recent_form'],
            'step5_streaks': ['matchup', 'situational', 'weather', 'recent_form', 'streak_momentum'],
            'step6_ballpark': ['matchup', 'situational', 'weather', 'recent_form', 'streak_momentum', 'ballpark'],
            'step7_temporal': ['matchup', 'situational', 'weather', 'recent_form', 'streak_momentum', 'ballpark', 'temporal_fatigue'],
            'step8_interactions': ['matchup', 'situational', 'weather', 'recent_form', 'streak_momentum', 'ballpark', 'temporal_fatigue', 'interactions']
        }
        
        # Calculate impact of each category
        for i, result in enumerate(results[1:], 1):  # Skip baseline
            prev_result = results[i-1]
            
            experiment_name = result['experiment_name']
            if experiment_name in experiment_mapping:
                current_categories = experiment_mapping[experiment_name]
                prev_categories = experiment_mapping[prev_result['experiment_name']] if prev_result['experiment_name'] in experiment_mapping else []
                
                # Find newly added category
                new_categories = [cat for cat in current_categories if cat not in prev_categories]
                
                if new_categories:
                    category = new_categories[0]  # Should be only one new category per step
                    
                    impact = {
                        'roc_auc_improvement': result['metrics'].get('roc_auc', 0) - prev_result['metrics'].get('roc_auc', 0),
                        'precision_improvement': result['metrics'].get('precision', 0) - prev_result['metrics'].get('precision', 0),
                        'feature_count': len([f for f in feature_categories[category] if f in result['available_features']]),
                        'step_name': experiment_name
                    }
                    
                    category_impact[category] = impact
        
        # Rank categories by impact
        ranked_categories = sorted(
            category_impact.items(),
            key=lambda x: x[1]['roc_auc_improvement'],
            reverse=True
        )
        
        return {
            'category_impacts': category_impact,
            'ranked_categories': ranked_categories,
            'most_impactful': ranked_categories[0] if ranked_categories else None,
            'least_impactful': ranked_categories[-1] if ranked_categories else None
        }
    
    def _analyze_efficiency(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze computational efficiency across experiments."""
        
        efficiency_metrics = []
        
        for result in results:
            metrics = {
                'experiment': result['experiment_name'],
                'feature_count': result['feature_count'],
                'training_time': result['training_time'],
                'prediction_time': result['prediction_time'],
                'roc_auc': result['metrics'].get('roc_auc', 0),
                'efficiency_ratio': result['metrics'].get('roc_auc', 0) / max(result['training_time'], 1),  # Performance per second
                'features_per_second': result['feature_count'] / max(result['training_time'], 1)
            }
            efficiency_metrics.append(metrics)
        
        # Find most/least efficient
        most_efficient = max(efficiency_metrics, key=lambda x: x['efficiency_ratio'])
        least_efficient = min(efficiency_metrics, key=lambda x: x['efficiency_ratio'])
        
        return {
            'efficiency_metrics': efficiency_metrics,
            'most_efficient': most_efficient,
            'least_efficient': least_efficient,
            'scalability_analysis': {
                'training_time_growth': efficiency_metrics[-1]['training_time'] / max(efficiency_metrics[0]['training_time'], 1),
                'feature_count_growth': efficiency_metrics[-1]['feature_count'] / max(efficiency_metrics[0]['feature_count'], 1),
                'prediction_time_impact': efficiency_metrics[-1]['prediction_time'] / max(efficiency_metrics[0]['prediction_time'], 0.001)
            }
        }
    
    def _analyze_roi(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze return on investment for feature engineering efforts."""
        
        baseline_performance = results[0]['metrics'].get('roc_auc', 0) if results else 0
        
        roi_analysis = []
        
        for i, result in enumerate(results[1:], 1):
            # Estimate development effort (simplified)
            feature_count = result['feature_count']
            prev_feature_count = results[i-1]['feature_count']
            new_features = feature_count - prev_feature_count
            
            # Rough effort estimate (hours per feature)
            estimated_effort_hours = new_features * 2  # 2 hours per feature (simplified)
            
            # Performance improvement
            performance_improvement = result['metrics'].get('roc_auc', 0) - results[i-1]['metrics'].get('roc_auc', 0)
            
            # ROI calculation
            roi = performance_improvement / max(estimated_effort_hours, 1) * 100  # Performance improvement per hour
            
            roi_analysis.append({
                'step': result['experiment_name'],
                'new_features': new_features,
                'estimated_effort_hours': estimated_effort_hours,
                'performance_improvement': performance_improvement,
                'roi_score': roi,
                'cumulative_improvement': result['metrics'].get('roc_auc', 0) - baseline_performance
            })
        
        # Overall ROI
        total_improvement = results[-1]['metrics'].get('roc_auc', 0) - baseline_performance if len(results) > 1 else 0
        total_features = results[-1]['feature_count'] - results[0]['feature_count'] if len(results) > 1 else 0
        total_effort = total_features * 2  # Simplified
        
        overall_roi = total_improvement / max(total_effort, 1) * 100
        
        return {
            'step_roi_analysis': roi_analysis,
            'overall_roi': overall_roi,
            'best_roi_step': max(roi_analysis, key=lambda x: x['roi_score']) if roi_analysis else None,
            'worst_roi_step': min(roi_analysis, key=lambda x: x['roi_score']) if roi_analysis else None,
            'total_improvement': total_improvement,
            'total_effort_estimate': total_effort
        }
    
    def _generate_recommendations(self,
                                results: List[Dict[str, Any]],
                                feature_importance_data: Dict[str, Dict]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        
        if not results:
            return ["No successful experiments to analyze"]
        
        # Performance-based recommendations
        best_result = max(results, key=lambda x: x['metrics'].get('roc_auc', 0))
        baseline_result = results[0]
        
        improvement = best_result['metrics'].get('roc_auc', 0) - baseline_result['metrics'].get('roc_auc', 0)
        
        if improvement > 0.05:
            recommendations.append(f"✅ Excellent improvement achieved: {improvement:.4f} ROC-AUC gain. Feature engineering pipeline is highly effective.")
        elif improvement > 0.02:
            recommendations.append(f"✅ Good improvement achieved: {improvement:.4f} ROC-AUC gain. Continue optimizing high-impact features.")
        else:
            recommendations.append(f"⚠️ Limited improvement: {improvement:.4f} ROC-AUC gain. Consider feature selection and engineering refinement.")
        
        # Feature category recommendations
        if len(results) > 1:
            # Find most impactful steps
            step_improvements = []
            for i in range(1, len(results)):
                step_improvement = results[i]['metrics'].get('roc_auc', 0) - results[i-1]['metrics'].get('roc_auc', 0)
                step_improvements.append((results[i]['experiment_name'], step_improvement))
            
            best_step = max(step_improvements, key=lambda x: x[1])
            if best_step[1] > 0.01:
                recommendations.append(f"🎯 Most impactful step: {best_step[0]} (+{best_step[1]:.4f} ROC-AUC). Prioritize similar features.")
        
        # Efficiency recommendations
        if len(results) > 2:
            training_times = [r['training_time'] for r in results]
            if training_times[-1] > training_times[0] * 5:
                recommendations.append("⚠️ Training time has increased significantly. Consider feature selection for production.")
        
        # Feature count recommendations
        final_feature_count = results[-1]['feature_count']
        if final_feature_count > 100:
            recommendations.append(f"⚠️ High feature count ({final_feature_count}). Consider feature selection for model interpretability and efficiency.")
        elif final_feature_count < 20:
            recommendations.append(f"ℹ️ Relatively low feature count ({final_feature_count}). There may be room for additional feature engineering.")
        
        # Production readiness recommendations
        prediction_time = results[-1]['prediction_time']
        if prediction_time > 1.0:
            recommendations.append("⚠️ Slow prediction time for production betting. Optimize feature calculation pipeline.")
        else:
            recommendations.append("✅ Prediction time is suitable for real-time betting applications.")
        
        # Model performance recommendations
        final_roc_auc = results[-1]['metrics'].get('roc_auc', 0)
        if final_roc_auc > 0.65:
            recommendations.append("🏆 Excellent model performance. Ready for production deployment with proper risk management.")
        elif final_roc_auc > 0.55:
            recommendations.append("✅ Good model performance. Consider additional feature engineering or ensemble methods.")
        else:
            recommendations.append("⚠️ Model performance needs improvement. Focus on data quality and feature relevance.")
        
        return recommendations
    
    def _print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print concise analysis summary."""
        
        logger.info("\\n📊 ANALYSIS SUMMARY")
        logger.info("-" * 50)
        
        summary = analysis['summary']
        logger.info(f"Total experiments: {summary['total_experiments']}")
        
        if 'best_performance' in summary:
            best = summary['best_performance']
            logger.info(f"Best performance: {best['name']} (ROC-AUC: {best['roc_auc']:.4f})")
        
        if 'largest_improvement' in summary:
            largest = summary['largest_improvement']
            logger.info(f"Largest step improvement: {largest.get('step', 'unknown')} (+{largest.get('improvement', 0):.4f})")
        
        logger.info(f"Overall ROI: {summary.get('feature_engineering_roi', 0):.2f} performance/hour")
        
        logger.info("\\n🎯 KEY RECOMMENDATIONS:")
        for rec in analysis['recommendations'][:5]:  # Top 5 recommendations
            logger.info(f"  {rec}")
    
    def _save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], timestamp: str = None):
        """Save results to JSON file."""
        
        output_data = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'training_period': f"{self.start_date} to {self.end_date}",
                'testing_period': f"{self.test_start_date} to {self.test_end_date}",
                'total_experiments': len(results)
            },
            'results': results,
            'analysis': analysis
        }
        
        # Save to file with timestamp for large analysis runs
        if timestamp:
            output_file = f'/home/charlesbenfer/betting_models/comparative_analysis_results_{timestamp}.json'
        else:
            output_file = '/home/charlesbenfer/betting_models/comparative_analysis_results.json'
            
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            logger.info(f"Results saved to {output_file}")
            
            # Also save a summary file for quick reference
            summary_file = output_file.replace('.json', '_summary.json')
            summary_data = {
                'training_period': f"{self.start_date} to {self.end_date}",
                'test_period': f"{self.test_start_date} to {self.test_end_date}",
                'total_experiments': len(results),
                'best_performance': analysis.get('summary', {}).get('best_performance', {}),
                'feature_engineering_roi': analysis.get('summary', {}).get('feature_engineering_roi', 0),
                'top_recommendations': analysis.get('recommendations', [])[:5]
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            logger.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _save_best_model(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
        """Save the best performing model (XGBoost or LightGBM) to the model directory."""
        try:
            if self.best_model is None:
                logger.warning("No best model found to save")
                return
            
            # Create model directory if it doesn't exist
            model_dir = Path(config.MODEL_DIR)
            model_dir.mkdir(exist_ok=True, parents=True)
            
            # Save the best model
            model_path = model_dir / f"best_model_{self.best_model_info['model_type'].lower()}.pkl"
            joblib.dump(self.best_model, model_path)
            
            # Save feature list
            features_path = model_dir / "best_model_features.json"
            with open(features_path, 'w') as f:
                json.dump(self.best_model_info['features'], f, indent=2)
            
            # Save model metadata
            best_model_metadata = {
                'experiment_name': self.best_model_info['experiment_name'],
                'model_type': self.best_model_info['model_type'],
                'roc_auc': self.best_model_info['roc_auc'],
                'feature_count': len(self.best_model_info['features']),
                'parameters': self.best_model_info['params'],
                'training_date': datetime.now().isoformat(),
                'training_period': f"{self.start_date} to {self.end_date}",
                'test_period': f"{self.test_start_date} to {self.test_end_date}",
                'model_file': str(model_path.name),
                'features_file': str(features_path.name)
            }
            
            metadata_path = model_dir / "best_model_info.json"
            with open(metadata_path, 'w') as f:
                json.dump(best_model_metadata, f, indent=2)
            
            logger.info(f"\\n✅ Best model saved successfully to {config.MODEL_DIR}")
            logger.info(f"   Experiment: {self.best_model_info['experiment_name']}")
            logger.info(f"   Model Type: {self.best_model_info['model_type']}")
            logger.info(f"   Features: {len(self.best_model_info['features'])}")
            logger.info(f"   Performance: ROC-AUC {self.best_model_info['roc_auc']:.4f}")
            logger.info(f"   Model file: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run comparative analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comparative analysis of MLB home run prediction models')
    parser.add_argument('--use-cache', action='store_true', default=False,
                       help='Use cached datasets instead of rebuilding from scratch')
    parser.add_argument('--quick-test', action='store_true', default=False,
                       help='Run quick test with smaller date ranges')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting comprehensive comparative analysis...")
    
    # Set date ranges based on mode
    if args.quick_test:
        start_date = "2024-06-01"
        end_date = "2024-07-31" 
        test_start_date = "2024-08-01"
        test_end_date = "2024-08-31"
        logger.info("🚀 Quick test mode: Using smaller date ranges")
    else:
        start_date = "2024-04-01"  
        end_date = "2024-08-31"
        test_start_date = "2024-09-01"  
        test_end_date = "2024-10-31"
    
    # Initialize analyzer with date ranges
    analyzer = ComparativeAnalyzer(
        start_date=start_date,
        end_date=end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        use_cache=args.use_cache
    )
    
    # Run analysis
    results = analyzer.run_comparative_analysis()
    
    if results:
        logger.info("\\n" + "="*80)
        logger.info("🎉 COMPARATIVE ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("Feature engineering pipeline evaluation complete.")
        logger.info("Check comparative_analysis_results.json for detailed results.")
        logger.info("="*80)
        return 0
    else:
        logger.error("\\nComparative analysis failed!")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)