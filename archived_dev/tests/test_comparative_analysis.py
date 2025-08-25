"""
Test Comparative Analysis (Step 10) - Simplified
==============================================

Quick test of the comparative analysis system.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import sys

# Import our modules
from comparative_analysis import ComparativeAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_comparative_analysis():
    """Test the comparative analysis system with a smaller scope."""
    logger.info("="*80)
    logger.info("COMPARATIVE ANALYSIS TESTING")
    logger.info("Testing feature engineering improvements (Steps 1-8)")
    logger.info("="*80)
    
    try:
        # Initialize analyzer with smaller date range for faster testing
        analyzer = ComparativeAnalyzer(
            start_date="2024-08-01",  # 2 weeks of training data
            end_date="2024-08-15",
            test_start_date="2024-08-16",  # 1 week of test data
            test_end_date="2024-08-23"
        )
        
        # Setup experiments (first few steps for quick test)
        experiments = analyzer.setup_experiments()
        
        logger.info(f"Set up {len(experiments)} experiments:")
        for i, exp in enumerate(experiments, 1):
            logger.info(f"  {i}. {exp.experiment_name}: {exp.description}")
        
        # Run a subset of experiments for testing
        logger.info("\\nRunning comparative analysis (subset for testing)...")
        
        # For testing, we'll just run the first 4 experiments
        test_experiments = experiments[:4]
        analyzer.experiments = test_experiments
        
        results = analyzer.run_comparative_analysis()
        
        if results:
            logger.info("\\n✅ Comparative analysis test completed successfully!")
            
            # Print key findings
            summary = results.get('summary', {})
            if summary:
                logger.info(f"\\n📊 Test Results Summary:")
                logger.info(f"  Experiments completed: {summary.get('total_experiments', 0)}")
                
                best_perf = summary.get('best_performance', {})
                if best_perf:
                    logger.info(f"  Best performance: {best_perf.get('name', 'unknown')} (ROC-AUC: {best_perf.get('roc_auc', 0):.4f})")
                
                roi = summary.get('feature_engineering_roi', 0)
                logger.info(f"  Feature engineering ROI: {roi:.2f}")
            
            # Print recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                logger.info(f"\\n🎯 Key Recommendations:")
                for rec in recommendations[:3]:  # Top 3
                    logger.info(f"    {rec}")
            
            return True
        else:
            logger.error("Comparative analysis test failed")
            return False
            
    except Exception as e:
        logger.error(f"Comparative analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main testing function."""
    logger.info("Starting comparative analysis testing...")
    logger.info("This will test the feature engineering evaluation system")
    
    success = test_comparative_analysis()
    
    if success:
        logger.info("\\n" + "="*80)
        logger.info("✅ COMPARATIVE ANALYSIS TEST COMPLETED!")
        logger.info("The feature engineering evaluation system is working.")
        logger.info("Run comparative_analysis.py for full analysis.")
        logger.info("="*80)
        return 0
    else:
        logger.error("\\nComparative analysis test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)