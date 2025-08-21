"""
Comprehensive Feature Testing Framework (Step 9)
===============================================

Systematic validation and testing infrastructure for all feature categories.
Includes automated testing, validation, benchmarking, and reporting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from config import config
from dataset_builder import PregameDatasetBuilder
from modeling import EnhancedDualModelSystem

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for test results."""
    test_name: str
    category: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    score: float
    details: Dict[str, Any]
    execution_time: float

@dataclass
class FeatureCategoryTest:
    """Definition of tests for a feature category."""
    category: str
    features: List[str]
    tests: List[str]
    validation_rules: Dict[str, Any]

class ComprehensiveTestingFramework:
    """Comprehensive testing framework for all feature categories."""
    
    def __init__(self, test_config: Dict[str, Any] = None):
        self.test_config = test_config or self._default_test_config()
        self.results = []
        self.feature_categories = self._define_feature_categories()
        
    def run_comprehensive_tests(self, dataset: pd.DataFrame, 
                               target_col: str = 'hit_hr',
                               extended_tests: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive testing suite on the dataset.
        
        Args:
            dataset: DataFrame with all features
            target_col: Target variable for correlation analysis
            extended_tests: Whether to run extended validation tests
            
        Returns:
            Complete test results and summary
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE FEATURE TESTING FRAMEWORK")
        logger.info("="*80)
        
        start_time = time.time()
        
        # Run test categories
        self._test_data_quality(dataset)
        self._test_feature_coverage(dataset)
        self._test_feature_distributions(dataset)
        self._test_feature_correlations(dataset, target_col)
        self._test_feature_interactions(dataset)
        self._test_model_integration(dataset, target_col)
        
        if extended_tests:
            self._test_feature_stability(dataset)
            self._test_feature_importance(dataset, target_col)
            self._benchmark_feature_categories(dataset, target_col)
            self._test_production_readiness(dataset)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        summary = self._generate_test_summary(total_time)
        
        logger.info(f"\nTesting completed in {total_time:.2f} seconds")
        return summary
    
    def _default_test_config(self) -> Dict[str, Any]:
        """Default testing configuration."""
        return {
            'min_coverage_threshold': 0.8,
            'max_missing_rate': 0.2,
            'correlation_significance': 0.01,
            'outlier_threshold': 3.0,
            'stability_threshold': 0.95,
            'performance_threshold': 0.55,
            'execution_timeout': 300  # 5 minutes
        }
    
    def _define_feature_categories(self) -> List[FeatureCategoryTest]:
        """Define test categories for each feature group."""
        return [
            FeatureCategoryTest(
                category="core",
                features=config.CORE_FEATURES,
                tests=["coverage", "distribution", "correlation", "stability"],
                validation_rules={"min_coverage": 0.95, "required_correlation": 0.02}
            ),
            FeatureCategoryTest(
                category="matchup",
                features=config.MATCHUP_FEATURES,
                tests=["coverage", "distribution", "correlation", "logic"],
                validation_rules={"min_coverage": 0.7, "max_extreme_values": 0.05}
            ),
            FeatureCategoryTest(
                category="situational",
                features=config.SITUATIONAL_FEATURES,
                tests=["coverage", "distribution", "correlation", "logic"],
                validation_rules={"min_coverage": 0.8, "logical_bounds": True}
            ),
            FeatureCategoryTest(
                category="weather",
                features=config.WEATHER_FEATURES,
                tests=["coverage", "distribution", "correlation", "physics"],
                validation_rules={"min_coverage": 0.6, "physics_validation": True}
            ),
            FeatureCategoryTest(
                category="recent_form",
                features=config.RECENT_FORM_FEATURES,
                tests=["coverage", "distribution", "correlation", "decay"],
                validation_rules={"min_coverage": 0.8, "decay_validation": True}
            ),
            FeatureCategoryTest(
                category="streak_momentum",
                features=config.STREAK_MOMENTUM_FEATURES,
                tests=["coverage", "distribution", "correlation", "psychology"],
                validation_rules={"min_coverage": 0.7, "psychological_validation": True}
            ),
            FeatureCategoryTest(
                category="ballpark",
                features=config.BALLPARK_FEATURES,
                tests=["coverage", "distribution", "correlation", "physics"],
                validation_rules={"min_coverage": 0.9, "dimensional_validation": True}
            ),
            FeatureCategoryTest(
                category="temporal_fatigue",
                features=config.TEMPORAL_FATIGUE_FEATURES,
                tests=["coverage", "distribution", "correlation", "temporal"],
                validation_rules={"min_coverage": 0.8, "temporal_validation": True}
            ),
            FeatureCategoryTest(
                category="interactions",
                features=config.INTERACTION_FEATURES,
                tests=["coverage", "distribution", "correlation", "interaction"],
                validation_rules={"min_coverage": 0.7, "interaction_validation": True}
            )
        ]
    
    def _test_data_quality(self, dataset: pd.DataFrame):
        """Test overall data quality."""
        logger.info("\n🔍 Testing Data Quality...")
        
        start_time = time.time()
        
        # Basic data quality metrics
        total_rows = len(dataset)
        total_cols = len(dataset.columns)
        missing_cells = dataset.isnull().sum().sum()
        missing_rate = missing_cells / (total_rows * total_cols)
        
        # Test results
        quality_score = 1.0 - missing_rate
        status = "PASS" if missing_rate < self.test_config['max_missing_rate'] else "FAIL"
        
        result = TestResult(
            test_name="data_quality_overall",
            category="data_quality",
            status=status,
            score=quality_score,
            details={
                "total_rows": total_rows,
                "total_columns": total_cols,
                "missing_cells": int(missing_cells),
                "missing_rate": missing_rate,
                "memory_usage_mb": dataset.memory_usage(deep=True).sum() / 1024**2
            },
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        logger.info(f"   Data Quality: {status} (Score: {quality_score:.3f})")
    
    def _test_feature_coverage(self, dataset: pd.DataFrame):
        """Test feature coverage across categories."""
        logger.info("\n📊 Testing Feature Coverage...")
        
        for category_test in self.feature_categories:
            start_time = time.time()
            
            available_features = [f for f in category_test.features if f in dataset.columns]
            coverage_rate = len(available_features) / len(category_test.features)
            min_coverage = category_test.validation_rules.get('min_coverage', 0.7)
            
            status = "PASS" if coverage_rate >= min_coverage else "FAIL"
            
            # Test individual feature coverage (non-null rate)
            feature_coverage = {}
            for feature in available_features:
                non_null_rate = dataset[feature].notna().mean()
                feature_coverage[feature] = non_null_rate
            
            avg_feature_coverage = np.mean(list(feature_coverage.values())) if feature_coverage else 0
            
            result = TestResult(
                test_name=f"coverage_{category_test.category}",
                category="coverage",
                status=status,
                score=coverage_rate,
                details={
                    "available_features": len(available_features),
                    "total_features": len(category_test.features),
                    "coverage_rate": coverage_rate,
                    "avg_feature_coverage": avg_feature_coverage,
                    "feature_coverage": feature_coverage
                },
                execution_time=time.time() - start_time
            )
            
            self.results.append(result)
            logger.info(f"   {category_test.category.title()} Coverage: {status} "
                       f"({len(available_features)}/{len(category_test.features)})")
    
    def _test_feature_distributions(self, dataset: pd.DataFrame):
        """Test feature distributions for anomalies."""
        logger.info("\n📈 Testing Feature Distributions...")
        
        for category_test in self.feature_categories:
            start_time = time.time()
            
            available_features = [f for f in category_test.features if f in dataset.columns]
            if not available_features:
                continue
            
            distribution_issues = {}
            outlier_counts = {}
            
            for feature in available_features:
                data = dataset[feature].dropna()
                if len(data) == 0:
                    continue
                
                # Test for extreme outliers
                if data.dtype in ['int64', 'float64']:
                    q1, q3 = data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    outlier_bounds = (q1 - 3 * iqr, q3 + 3 * iqr)
                    outliers = ((data < outlier_bounds[0]) | (data > outlier_bounds[1])).sum()
                    outlier_rate = outliers / len(data)
                    outlier_counts[feature] = outlier_rate
                    
                    if outlier_rate > 0.05:  # More than 5% outliers
                        distribution_issues[feature] = f"High outlier rate: {outlier_rate:.1%}"
                
                # Test for constant values
                if data.nunique() == 1:
                    distribution_issues[feature] = "Constant values"
                elif data.nunique() < 3 and len(data) > 100:
                    distribution_issues[feature] = "Very low variance"
            
            avg_outlier_rate = np.mean(list(outlier_counts.values())) if outlier_counts else 0
            status = "PASS" if len(distribution_issues) == 0 else "WARNING"
            
            result = TestResult(
                test_name=f"distributions_{category_test.category}",
                category="distributions",
                status=status,
                score=1.0 - avg_outlier_rate,
                details={
                    "features_tested": len(available_features),
                    "distribution_issues": distribution_issues,
                    "avg_outlier_rate": avg_outlier_rate,
                    "outlier_counts": outlier_counts
                },
                execution_time=time.time() - start_time
            )
            
            self.results.append(result)
            logger.info(f"   {category_test.category.title()} Distributions: {status}")
    
    def _test_feature_correlations(self, dataset: pd.DataFrame, target_col: str):
        """Test feature correlations with target variable."""
        logger.info("\n🎯 Testing Feature Correlations...")
        
        if target_col not in dataset.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return
        
        for category_test in self.feature_categories:
            start_time = time.time()
            
            available_features = [f for f in category_test.features if f in dataset.columns]
            if not available_features:
                continue
            
            correlations = {}
            significant_correlations = 0
            
            for feature in available_features:
                if dataset[feature].dtype in ['int64', 'float64']:
                    corr = dataset[feature].corr(dataset[target_col])
                    if not pd.isna(corr):
                        correlations[feature] = corr
                        if abs(corr) >= self.test_config['correlation_significance']:
                            significant_correlations += 1
            
            correlation_rate = significant_correlations / len(correlations) if correlations else 0
            avg_abs_correlation = np.mean([abs(c) for c in correlations.values()]) if correlations else 0
            
            status = "PASS" if correlation_rate > 0.1 else "WARNING"  # At least 10% should be significant
            
            result = TestResult(
                test_name=f"correlations_{category_test.category}",
                category="correlations",
                status=status,
                score=avg_abs_correlation,
                details={
                    "features_tested": len(correlations),
                    "significant_correlations": significant_correlations,
                    "correlation_rate": correlation_rate,
                    "avg_abs_correlation": avg_abs_correlation,
                    "top_correlations": dict(sorted(correlations.items(), 
                                                   key=lambda x: abs(x[1]), reverse=True)[:5])
                },
                execution_time=time.time() - start_time
            )
            
            self.results.append(result)
            logger.info(f"   {category_test.category.title()} Correlations: {status} "
                       f"({significant_correlations}/{len(correlations)} significant)")
    
    def _test_feature_interactions(self, dataset: pd.DataFrame):
        """Test feature interaction validity."""
        logger.info("\n🔗 Testing Feature Interactions...")
        
        start_time = time.time()
        
        interaction_features = [f for f in config.INTERACTION_FEATURES if f in dataset.columns]
        if not interaction_features:
            logger.warning("No interaction features found")
            return
        
        interaction_issues = {}
        
        for feature in interaction_features:
            data = dataset[feature].dropna()
            if len(data) == 0:
                continue
            
            # Test multiplier features
            if 'multiplier' in feature:
                if data.min() < 0 or data.max() > 10:
                    interaction_issues[feature] = "Multiplier values outside reasonable range"
                if abs(data.mean() - 1.0) > 2.0:
                    interaction_issues[feature] = "Multiplier not centered around 1.0"
            
            # Test ratio features
            elif 'ratio' in feature:
                if data.min() < 0 or data.max() > 1000:
                    interaction_issues[feature] = "Ratio values extremely large"
            
            # Test index features
            elif 'index' in feature:
                if data.std() < 0.01:
                    interaction_issues[feature] = "Index has very low variance"
            
            # Test indicator features
            elif 'indicator' in feature:
                unique_vals = data.unique()
                if len(unique_vals) > 10:
                    interaction_issues[feature] = "Indicator has too many unique values"
        
        status = "PASS" if len(interaction_issues) == 0 else "WARNING"
        
        result = TestResult(
            test_name="interactions_validation",
            category="interactions",
            status=status,
            score=1.0 - (len(interaction_issues) / len(interaction_features)),
            details={
                "features_tested": len(interaction_features),
                "interaction_issues": interaction_issues,
                "issue_rate": len(interaction_issues) / len(interaction_features)
            },
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        logger.info(f"   Interaction Validation: {status}")
    
    def _test_model_integration(self, dataset: pd.DataFrame, target_col: str):
        """Test integration with model system."""
        logger.info("\n🤖 Testing Model Integration...")
        
        start_time = time.time()
        
        try:
            # Initialize model system
            model_system = EnhancedDualModelSystem()
            
            # Test feature identification
            available_features = model_system.feature_selector.identify_available_features(dataset)
            
            # Quick model training test
            if target_col in dataset.columns and len(dataset) > 100:
                test_data = dataset.head(500).copy()
                
                if test_data[target_col].sum() < 5:  # Need some positive examples
                    test_data.loc[:10, target_col] = 1
                
                try:
                    results = model_system.fit(
                        test_data,
                        splitting_strategy='random',
                        test_size=0.3,
                        val_size=0.2,
                        cross_validate=False
                    )
                    
                    model_performance = results.get('test_metrics', {}).get('roc_auc', 0)
                    status = "PASS" if model_performance > 0.5 else "WARNING"
                    
                except Exception as e:
                    model_performance = 0
                    status = "FAIL"
                    logger.warning(f"Model training failed: {e}")
            else:
                model_performance = 0
                status = "WARNING"
            
            result = TestResult(
                test_name="model_integration",
                category="integration",
                status=status,
                score=model_performance,
                details={
                    "total_enhanced_features": len(available_features['enhanced']),
                    "feature_breakdown": {k: len(v) for k, v in available_features.items()},
                    "model_performance": model_performance,
                    "training_successful": status != "FAIL"
                },
                execution_time=time.time() - start_time
            )
            
            self.results.append(result)
            logger.info(f"   Model Integration: {status} (Performance: {model_performance:.3f})")
            
        except Exception as e:
            logger.error(f"Model integration test failed: {e}")
    
    def _test_feature_stability(self, dataset: pd.DataFrame):
        """Test feature stability across time periods."""
        logger.info("\n⏰ Testing Feature Stability...")
        
        if 'date' not in dataset.columns:
            logger.warning("Date column not available for stability testing")
            return
        
        start_time = time.time()
        
        # Split data into time periods
        dataset['date'] = pd.to_datetime(dataset['date'])
        sorted_data = dataset.sort_values('date')
        
        split_point = len(sorted_data) // 2
        early_data = sorted_data.iloc[:split_point]
        late_data = sorted_data.iloc[split_point:]
        
        stability_scores = {}
        
        # Test numeric features for stability
        numeric_features = dataset.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col not in ['home_runs', 'hit_hr', 'batter', 'game_pk']]
        
        for feature in numeric_features[:50]:  # Test first 50 to avoid timeout
            early_mean = early_data[feature].mean()
            late_mean = late_data[feature].mean()
            
            if early_mean != 0:
                stability = 1 - abs(late_mean - early_mean) / abs(early_mean)
                stability_scores[feature] = max(0, stability)
        
        avg_stability = np.mean(list(stability_scores.values())) if stability_scores else 0
        status = "PASS" if avg_stability >= self.test_config['stability_threshold'] else "WARNING"
        
        result = TestResult(
            test_name="feature_stability",
            category="stability",
            status=status,
            score=avg_stability,
            details={
                "features_tested": len(stability_scores),
                "avg_stability": avg_stability,
                "low_stability_features": [f for f, s in stability_scores.items() if s < 0.8]
            },
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        logger.info(f"   Feature Stability: {status} (Avg: {avg_stability:.3f})")
    
    def _test_feature_importance(self, dataset: pd.DataFrame, target_col: str):
        """Test feature importance using simple methods."""
        logger.info("\n🎖️ Testing Feature Importance...")
        
        if target_col not in dataset.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return
        
        start_time = time.time()
        
        # Calculate correlation-based importance
        numeric_features = dataset.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col not in ['home_runs', 'hit_hr', 'batter', 'game_pk']]
        
        importance_scores = {}
        for feature in numeric_features:
            corr = abs(dataset[feature].corr(dataset[target_col]))
            if not pd.isna(corr):
                importance_scores[feature] = corr
        
        # Find top features
        top_features = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:20])
        avg_importance = np.mean(list(importance_scores.values())) if importance_scores else 0
        
        status = "PASS" if avg_importance > 0.02 else "WARNING"
        
        result = TestResult(
            test_name="feature_importance",
            category="importance",
            status=status,
            score=avg_importance,
            details={
                "features_tested": len(importance_scores),
                "avg_importance": avg_importance,
                "top_features": top_features,
                "high_importance_count": sum(1 for s in importance_scores.values() if s > 0.05)
            },
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        logger.info(f"   Feature Importance: {status} (Avg: {avg_importance:.4f})")
    
    def _benchmark_feature_categories(self, dataset: pd.DataFrame, target_col: str):
        """Benchmark different feature categories."""
        logger.info("\n🏆 Benchmarking Feature Categories...")
        
        if target_col not in dataset.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return
        
        start_time = time.time()
        
        category_performance = {}
        
        for category_test in self.feature_categories:
            available_features = [f for f in category_test.features if f in dataset.columns]
            if len(available_features) < 3:
                continue
            
            # Calculate average correlation for category
            correlations = []
            for feature in available_features:
                if dataset[feature].dtype in ['int64', 'float64']:
                    corr = abs(dataset[feature].corr(dataset[target_col]))
                    if not pd.isna(corr):
                        correlations.append(corr)
            
            if correlations:
                avg_correlation = np.mean(correlations)
                category_performance[category_test.category] = {
                    'avg_correlation': avg_correlation,
                    'feature_count': len(available_features),
                    'tested_features': len(correlations)
                }
        
        # Rank categories
        ranked_categories = sorted(category_performance.items(), 
                                 key=lambda x: x[1]['avg_correlation'], reverse=True)
        
        status = "PASS"
        
        result = TestResult(
            test_name="category_benchmark",
            category="benchmark",
            status=status,
            score=max([perf['avg_correlation'] for perf in category_performance.values()]) if category_performance else 0,
            details={
                "category_performance": category_performance,
                "ranked_categories": [(cat, perf['avg_correlation']) for cat, perf in ranked_categories]
            },
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        logger.info(f"   Category Benchmark: {status}")
        
        if ranked_categories:
            logger.info("   Top performing categories:")
            for i, (category, perf) in enumerate(ranked_categories[:5], 1):
                logger.info(f"     {i}. {category}: {perf['avg_correlation']:.4f}")
    
    def _test_production_readiness(self, dataset: pd.DataFrame):
        """Test production readiness metrics."""
        logger.info("\n🚀 Testing Production Readiness...")
        
        start_time = time.time()
        
        readiness_checks = {}
        
        # Check data size
        readiness_checks['sufficient_data'] = len(dataset) >= 1000
        
        # Check feature completeness
        total_expected = sum(len(cat.features) for cat in self.feature_categories)
        total_available = sum(1 for cat in self.feature_categories 
                            for f in cat.features if f in dataset.columns)
        readiness_checks['feature_completeness'] = total_available / total_expected >= 0.7
        
        # Check missing data
        missing_rate = dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))
        readiness_checks['acceptable_missing_rate'] = missing_rate < 0.3
        
        # Check memory usage
        memory_mb = dataset.memory_usage(deep=True).sum() / 1024**2
        readiness_checks['reasonable_memory_usage'] = memory_mb < 1000  # Less than 1GB
        
        # Overall readiness score
        readiness_score = sum(readiness_checks.values()) / len(readiness_checks)
        status = "PASS" if readiness_score >= 0.8 else "WARNING"
        
        result = TestResult(
            test_name="production_readiness",
            category="production",
            status=status,
            score=readiness_score,
            details={
                "readiness_checks": readiness_checks,
                "readiness_score": readiness_score,
                "dataset_size": len(dataset),
                "memory_usage_mb": memory_mb,
                "missing_rate": missing_rate
            },
            execution_time=time.time() - start_time
        )
        
        self.results.append(result)
        logger.info(f"   Production Readiness: {status} (Score: {readiness_score:.3f})")
    
    def _generate_test_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        
        # Aggregate results by category
        category_summary = {}
        for result in self.results:
            if result.category not in category_summary:
                category_summary[result.category] = {
                    'tests': [],
                    'pass_count': 0,
                    'fail_count': 0,
                    'warning_count': 0,
                    'avg_score': 0
                }
            
            category_summary[result.category]['tests'].append(result)
            
            if result.status == 'PASS':
                category_summary[result.category]['pass_count'] += 1
            elif result.status == 'FAIL':
                category_summary[result.category]['fail_count'] += 1
            else:
                category_summary[result.category]['warning_count'] += 1
        
        # Calculate average scores
        for category in category_summary:
            scores = [r.score for r in category_summary[category]['tests']]
            category_summary[category]['avg_score'] = np.mean(scores) if scores else 0
        
        # Overall summary
        total_tests = len(self.results)
        total_pass = sum(1 for r in self.results if r.status == 'PASS')
        total_fail = sum(1 for r in self.results if r.status == 'FAIL')
        total_warning = sum(1 for r in self.results if r.status == 'WARNING')
        
        overall_score = np.mean([r.score for r in self.results]) if self.results else 0
        
        summary = {
            'overall': {
                'total_tests': total_tests,
                'pass_count': total_pass,
                'fail_count': total_fail,
                'warning_count': total_warning,
                'pass_rate': total_pass / total_tests if total_tests > 0 else 0,
                'overall_score': overall_score,
                'execution_time': total_time
            },
            'category_summary': category_summary,
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'category': r.category,
                    'status': r.status,
                    'score': r.score,
                    'execution_time': r.execution_time
                }
                for r in self.results
            ]
        }
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE TESTING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {total_pass} ({total_pass/total_tests:.1%})")
        logger.info(f"⚠️  Warnings: {total_warning} ({total_warning/total_tests:.1%})")
        logger.info(f"❌ Failed: {total_fail} ({total_fail/total_tests:.1%})")
        logger.info(f"📊 Overall Score: {overall_score:.3f}")
        logger.info(f"⏱️  Execution Time: {total_time:.2f} seconds")
        
        logger.info("\nCategory Performance:")
        for category, data in category_summary.items():
            score = data['avg_score']
            tests = len(data['tests'])
            passes = data['pass_count']
            logger.info(f"  {category.title()}: {score:.3f} ({passes}/{tests} passed)")
        
        return summary
    
    def save_results(self, filepath: str):
        """Save test results to file."""
        summary = self._generate_test_summary(0)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filepath}")

def run_comprehensive_testing(start_date: str = "2024-08-01", 
                             end_date: str = "2024-08-15",
                             save_results: bool = True) -> Dict[str, Any]:
    """
    Run the complete comprehensive testing framework.
    
    Args:
        start_date: Start date for dataset building
        end_date: End date for dataset building
        save_results: Whether to save results to file
        
    Returns:
        Complete test results
    """
    logger.info("Starting comprehensive feature testing framework...")
    
    # Build dataset
    logger.info("Building comprehensive dataset...")
    builder = PregameDatasetBuilder(start_date=start_date, end_date=end_date)
    dataset = builder.build_dataset(force_rebuild=True)
    
    if dataset.empty:
        logger.error("Failed to build dataset")
        return {}
    
    logger.info(f"Dataset built: {len(dataset)} rows, {len(dataset.columns)} columns")
    
    # Run testing framework
    framework = ComprehensiveTestingFramework()
    results = framework.run_comprehensive_tests(dataset, extended_tests=True)
    
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        framework.save_results(results_file)
    
    return results

if __name__ == "__main__":
    # Run comprehensive testing
    results = run_comprehensive_testing()
    
    if results:
        overall = results['overall']
        logger.info(f"\n🎉 Testing completed! Overall score: {overall['overall_score']:.3f}")
        logger.info(f"Pass rate: {overall['pass_rate']:.1%}")
    else:
        logger.error("Testing failed!")