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
                 start_date: str = "2021-01-01",
                 end_date: str = "2024-08-31", 
                 test_start_date: str = "2024-09-01",
                 test_end_date: str = "2024-10-31"):
        self.start_date = start_date
        self.end_date = end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.experiments = []
        self.results = []
        
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
        logger.info(f"Training period: {self.start_date} to {self.end_date} (4 years)")
        logger.info(f"Testing period: {self.test_start_date} to {self.test_end_date} (2 months)")
        logger.info("⚠️  Large dataset build - this may take 30-60 minutes on first run")
        
        train_dataset = self._build_dataset(self.start_date, self.end_date, "training")
        test_dataset = self._build_dataset(self.test_start_date, self.test_end_date, "testing")
        
        if train_dataset.empty or test_dataset.empty:
            logger.error("Failed to build datasets")
            return {}
        
        # Run experiments with progress tracking
        results = []
        feature_importance_data = {}
        total_experiments = len(experiments) - 1  # Exclude optimized subset initially
        
        logger.info(f"\\n🚀 Starting {total_experiments} experiments with 4 years of training data")
        logger.info("Each experiment may take 10-30 minutes depending on feature complexity")
        
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
            
            dataset = builder.build_dataset(force_rebuild=True)
            
            if dataset.empty:
                logger.error(f"{dataset_type.title()} dataset is empty")
                return pd.DataFrame()
            
            logger.info(f"{dataset_type.title()} dataset: {len(dataset)} rows, {len(dataset.columns)} columns")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to build {dataset_type} dataset: {e}")
            return pd.DataFrame()
    
    def _run_single_experiment(self, 
                             experiment: ExperimentConfig,
                             train_dataset: pd.DataFrame,
                             test_dataset: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run a single experiment."""
        try:
            start_time = datetime.now()
            
            # Filter features
            available_features = [f for f in experiment.features_to_include 
                                if f in train_dataset.columns]
            missing_features = [f for f in experiment.features_to_include 
                              if f not in train_dataset.columns]
            
            logger.info(f"Available features: {len(available_features)}")
            if missing_features:
                logger.warning(f"Missing features: {len(missing_features)}")
            
            if len(available_features) < 5:
                logger.warning("Too few features available, skipping experiment")
                return None
            
            # Prepare datasets with only available features + target
            feature_cols = available_features + ['hit_hr']
            train_data = train_dataset[feature_cols].copy()
            test_data = test_dataset[feature_cols].copy()
            
            # Check for sufficient data
            if len(train_data) < 100 or len(test_data) < 20:
                logger.warning("Insufficient data for reliable testing")
                return None
            
            # Initialize and train model
            model_system = EnhancedDualModelSystem()
            
            training_start = datetime.now()
            
            # Train with time-based split to prevent leakage
            # For large datasets, use larger validation sets for more robust estimates
            results = model_system.fit(
                train_data,
                splitting_strategy='temporal',
                test_size=0.15,  # Smaller test set since we have holdout data
                val_size=0.15,
                cross_validate=True,
                cv_folds=5  # More folds for robust CV with large data
            )
            
            training_time = (datetime.now() - training_start).total_seconds()
            
            # Test on holdout set
            prediction_start = datetime.now()
            
            test_predictions = model_system.predict(test_data.drop('hit_hr', axis=1))
            
            prediction_time = (datetime.now() - prediction_start).total_seconds()
            
            # Calculate test metrics
            test_metrics = model_system._calculate_metrics(
                test_data['hit_hr'].values,
                test_predictions
            )
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model_system, available_features)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Results:")
            logger.info(f"  ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}")
            logger.info(f"  Precision: {test_metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall: {test_metrics.get('recall', 0):.4f}")
            logger.info(f"  F1-Score: {test_metrics.get('f1', 0):.4f}")
            logger.info(f"  Features: {len(available_features)}")
            logger.info(f"  Training time: {training_time:.2f}s")
            logger.info(f"  Prediction time: {prediction_time:.4f}s")
            
            return {
                'experiment_name': experiment.experiment_name,
                'description': experiment.description,
                'metrics': test_metrics,
                'feature_count': len(available_features),
                'training_time': training_time,
                'prediction_time': prediction_time,
                'total_time': total_time,
                'available_features': available_features,
                'missing_features': missing_features,
                'feature_importance': feature_importance,
                'cross_validation_results': results.get('cv_results', {}),
                'model_info': {
                    'best_model': results.get('best_model_name', 'unknown'),
                    'ensemble_used': results.get('ensemble_used', False)
                }
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


def main():
    """Run comparative analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting comprehensive comparative analysis...")
    
    # Initialize analyzer with comprehensive date ranges for robust analysis
    analyzer = ComparativeAnalyzer(
        start_date="2021-01-01",  # 4 years of training data
        end_date="2024-08-31",
        test_start_date="2024-09-01",  # 2 months of test data  
        test_end_date="2024-10-31"
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