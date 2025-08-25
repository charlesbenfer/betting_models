"""
Feature Impact Analysis
======================

Compare model performance with different feature sets to validate improvements.
Tests baseline vs enhanced features (matchup, situational, weather).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
import sys
from datetime import datetime

# Import our modules
from config import config
from dataset_builder import PregameDatasetBuilder
from modeling import EnhancedDualModelSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_feature_impact():
    """Test the impact of enhanced features on model performance."""
    logger.info("="*80)
    logger.info("FEATURE IMPACT ANALYSIS")
    logger.info("Testing Steps 1-3 Feature Enhancements")
    logger.info("="*80)
    
    try:
        # Build comprehensive dataset
        logger.info("Building comprehensive dataset for testing...")
        dataset = build_test_dataset()
        
        if dataset.empty:
            logger.error("Failed to build test dataset")
            return False
        
        logger.info(f"Test dataset: {len(dataset)} rows, {len(dataset.columns)} columns")
        
        # Test different feature combinations
        feature_results = test_feature_combinations(dataset)
        
        # Analyze feature importance
        analyze_feature_importance(dataset, feature_results)
        
        # Generate recommendations
        generate_recommendations(feature_results)
        
        logger.info("\nFeature impact analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Feature impact analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def build_test_dataset() -> pd.DataFrame:
    """Build a larger dataset for comprehensive testing."""
    logger.info("Building test dataset with all enhanced features...")
    
    # Use a longer period for more robust testing
    builder = PregameDatasetBuilder(
        start_date="2024-07-01",  # Extended period
        end_date="2024-08-31"
    )
    
    dataset = builder.build_dataset(force_rebuild=True)
    
    if not dataset.empty:
        logger.info(f"Dataset built: {len(dataset)} games")
        logger.info(f"HR rate: {dataset['hit_hr'].mean():.3f}")
        logger.info(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")
    
    return dataset

def test_feature_combinations(dataset: pd.DataFrame) -> Dict[str, Dict]:
    """Test different feature combinations and compare performance."""
    logger.info("\n" + "="*70)
    logger.info("TESTING FEATURE COMBINATIONS")
    logger.info("="*70)
    
    results = {}
    
    # Initialize model system
    model_system = EnhancedDualModelSystem()
    available_features = model_system.feature_selector.identify_available_features(dataset)
    
    # Feature combinations to test
    feature_sets = {
        "baseline_core": {
            "name": "Baseline (Core Features Only)",
            "features": available_features['core'],
            "description": "Traditional rolling stats, handedness, pitcher features"
        },
        "core_plus_matchup": {
            "name": "Core + Matchup Features",
            "features": available_features['core'] + available_features.get('matchup', []),
            "description": "Baseline + batter vs pitcher history"
        },
        "core_plus_situational": {
            "name": "Core + Situational Features", 
            "features": available_features['core'] + available_features.get('situational', []),
            "description": "Baseline + pressure/leverage/inning context"
        },
        "core_plus_weather": {
            "name": "Core + Weather Features",
            "features": available_features['core'] + available_features.get('weather', []),
            "description": "Baseline + atmospheric conditions"
        },
        "enhanced_all": {
            "name": "All Enhanced Features",
            "features": available_features['enhanced'],
            "description": "All features: core + matchup + situational + weather"
        }
    }
    
    logger.info(f"Testing {len(feature_sets)} feature combinations...")
    
    for set_name, feature_info in feature_sets.items():
        logger.info(f"\nTesting: {feature_info['name']}")
        logger.info(f"Features: {len(feature_info['features'])} total")
        logger.info(f"Description: {feature_info['description']}")
        
        if len(feature_info['features']) == 0:
            logger.warning(f"No features available for {set_name}, skipping")
            continue
        
        try:
            # Create feature-specific model
            test_model = create_feature_specific_model(feature_info['features'])
            
            # Train and evaluate
            metrics = train_and_evaluate_model(test_model, dataset, feature_info['features'])
            
            results[set_name] = {
                'name': feature_info['name'],
                'feature_count': len(feature_info['features']),
                'metrics': metrics,
                'features': feature_info['features']
            }
            
            # Log key metrics
            if metrics:
                logger.info(f"Results: ROC-AUC={metrics.get('roc_auc', 0):.4f}, "
                          f"Precision={metrics.get('precision', 0):.4f}, "
                          f"Recall={metrics.get('recall', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Failed to test {set_name}: {e}")
            results[set_name] = {
                'name': feature_info['name'],
                'feature_count': len(feature_info['features']),
                'metrics': {},
                'error': str(e)
            }
    
    return results

def create_feature_specific_model(features: List[str]):
    """Create a model system with specific features."""
    from modeling import BaseModelComponent
    
    # Create a simple model component for testing
    class FeatureTestModel:
        def __init__(self, feature_list):
            self.feature_list = feature_list
            self.model_component = BaseModelComponent(feature_list)
        
        def fit(self, X_train, y_train, X_val=None, y_val=None):
            # BaseModelComponent uses train() method, not fit()
            # Create DataFrame for training
            import pandas as pd
            train_df = pd.DataFrame(X_train, columns=self.feature_list[:X_train.shape[1]])
            train_df['hit_hr'] = y_train
            return self.model_component.train(train_df)
        
        def predict_proba(self, X):
            import pandas as pd
            test_df = pd.DataFrame(X, columns=self.feature_list[:X.shape[1]])
            predictions = self.model_component.predict_proba(test_df)
            # Return probabilities as 1D array for binary classification
            if hasattr(predictions, 'values'):
                return predictions.values
            return predictions
    
    return FeatureTestModel(features)

def train_and_evaluate_model(model, dataset: pd.DataFrame, features: List[str]) -> Dict:
    """Train and evaluate a model with specific features."""
    
    # Prepare data
    available_features = [f for f in features if f in dataset.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")
    
    if len(available_features) < 5:
        logger.warning(f"Too few features available: {len(available_features)}")
        return {}
    
    # Create feature matrix
    X = dataset[available_features].copy()
    y = dataset['hit_hr'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Time-based split (more realistic for time series)
    split_date = dataset['date'].quantile(0.7)  # 70% train, 30% test
    train_mask = dataset['date'] <= split_date
    
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    
    # Further split training into train/val
    val_split_date = dataset[train_mask]['date'].quantile(0.8)
    val_mask = dataset['date'] <= val_split_date
    val_mask = val_mask & train_mask
    
    X_val = X[val_mask & ~train_mask]
    y_val = y[val_mask & ~train_mask]
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    if len(X_train) < 50 or len(X_test) < 20:
        logger.warning("Insufficient data for reliable evaluation")
        return {}
    
    try:
        # Train model
        model.fit(X_train, y_train, X_val, y_val)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
        
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'baseline_accuracy': 1 - y_test.mean(),  # Accuracy if always predicting no HR
            'test_hr_rate': y_test.mean(),
            'predicted_hr_rate': y_pred.mean()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Model training/evaluation failed: {e}")
        return {}

def analyze_feature_importance(dataset: pd.DataFrame, results: Dict) -> None:
    """Analyze which feature additions provided the most benefit."""
    logger.info("\n" + "="*70)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*70)
    
    # Extract key metrics for comparison
    comparison_data = []
    
    for set_name, result in results.items():
        if 'metrics' in result and result['metrics']:
            metrics = result['metrics']
            comparison_data.append({
                'name': result['name'],
                'feature_count': result['feature_count'],
                'roc_auc': metrics.get('roc_auc', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0)
            })
    
    if not comparison_data:
        logger.warning("No valid results for comparison")
        return
    
    # Sort by ROC-AUC (primary metric)
    comparison_data.sort(key=lambda x: x['roc_auc'], reverse=True)
    
    logger.info("\nPerformance Ranking (by ROC-AUC):")
    logger.info("-" * 90)
    logger.info(f"{'Rank':<4} {'Feature Set':<25} {'Features':<8} {'ROC-AUC':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    logger.info("-" * 90)
    
    baseline_auc = None
    for i, data in enumerate(comparison_data, 1):
        logger.info(f"{i:<4} {data['name']:<25} {data['feature_count']:<8} "
                   f"{data['roc_auc']:<8.4f} {data['precision']:<10.4f} "
                   f"{data['recall']:<8.4f} {data['f1']:<8.4f}")
        
        if 'Baseline' in data['name']:
            baseline_auc = data['roc_auc']
    
    # Calculate improvements over baseline
    if baseline_auc is not None:
        logger.info("\nImprovement vs Baseline:")
        logger.info("-" * 60)
        for data in comparison_data:
            if 'Baseline' not in data['name']:
                improvement = data['roc_auc'] - baseline_auc
                improvement_pct = (improvement / baseline_auc) * 100 if baseline_auc > 0 else 0
                logger.info(f"{data['name']:<25}: +{improvement:+.4f} ({improvement_pct:+.1f}%)")

def generate_recommendations(results: Dict) -> None:
    """Generate recommendations based on the feature impact analysis."""
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS")
    logger.info("="*70)
    
    # Find best performing feature set
    valid_results = {k: v for k, v in results.items() 
                    if 'metrics' in v and v['metrics'] and 'roc_auc' in v['metrics']}
    
    if not valid_results:
        logger.warning("No valid results to analyze")
        return
    
    best_set = max(valid_results.items(), key=lambda x: x[1]['metrics']['roc_auc'])
    baseline = valid_results.get('baseline_core', {})
    
    logger.info(f"🏆 Best Performing: {best_set[1]['name']}")
    logger.info(f"   Features: {best_set[1]['feature_count']}")
    logger.info(f"   ROC-AUC: {best_set[1]['metrics']['roc_auc']:.4f}")
    
    if baseline and 'metrics' in baseline:
        baseline_auc = baseline['metrics']['roc_auc']
        improvement = best_set[1]['metrics']['roc_auc'] - baseline_auc
        improvement_pct = (improvement / baseline_auc) * 100 if baseline_auc > 0 else 0
        
        logger.info(f"   Improvement vs Baseline: +{improvement:+.4f} ({improvement_pct:+.1f}%)")
        
        if improvement > 0.01:  # 1% improvement threshold
            logger.info("\n✅ RECOMMENDATION: Enhanced features show significant improvement!")
            logger.info("   → Proceed with Step 4 and continue feature engineering")
        elif improvement > 0.005:  # 0.5% improvement threshold  
            logger.info("\n⚠️  RECOMMENDATION: Modest improvement detected")
            logger.info("   → Enhanced features are helping, but consider optimization")
        else:
            logger.info("\n❌ RECOMMENDATION: Limited improvement from enhanced features")
            logger.info("   → Focus on data quality and feature selection before adding more")
    
    # Feature-specific recommendations
    logger.info("\nFeature Set Analysis:")
    
    feature_impacts = []
    if baseline and 'metrics' in baseline:
        baseline_auc = baseline['metrics']['roc_auc']
        
        for set_name, result in valid_results.items():
            if set_name != 'baseline_core' and 'Core +' in result['name']:
                improvement = result['metrics']['roc_auc'] - baseline_auc
                feature_impacts.append((result['name'], improvement))
    
    feature_impacts.sort(key=lambda x: x[1], reverse=True)
    
    for name, impact in feature_impacts:
        impact_pct = (impact / baseline_auc) * 100 if baseline_auc > 0 else 0
        status = "🔥" if impact > 0.01 else "✅" if impact > 0.005 else "📊" if impact > 0 else "❌"
        logger.info(f"   {status} {name}: {impact:+.4f} ({impact_pct:+.1f}%)")
    
    # Next steps
    logger.info(f"\n📋 Next Steps:")
    best_auc = best_set[1]['metrics']['roc_auc']
    
    if best_auc > 0.6:
        logger.info("   1. Model performance is promising - continue with Step 4")
        logger.info("   2. Consider hyperparameter tuning for current features")
        logger.info("   3. Experiment with feature interactions")
    elif best_auc > 0.55:
        logger.info("   1. Moderate performance - enhanced features are helping")
        logger.info("   2. Focus on feature quality and engineering in Step 4")
        logger.info("   3. Consider ensemble methods")
    else:
        logger.info("   1. Low performance - investigate data quality issues")
        logger.info("   2. Review feature engineering approach")
        logger.info("   3. Consider alternative modeling strategies")

def main():
    """Main testing function."""
    logger.info("Starting comprehensive feature impact analysis...")
    logger.info("This will test Steps 1-3 enhancements vs baseline performance")
    
    success = test_feature_impact()
    
    if success:
        logger.info("\n" + "="*80)
        logger.info("✅ FEATURE IMPACT ANALYSIS COMPLETED!")
        logger.info("Review the recommendations above to decide on Step 4")
        logger.info("="*80)
        return 0
    else:
        logger.error("\nFeature impact analysis failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)