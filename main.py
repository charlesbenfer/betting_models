"""
Enhanced Main Entry Point with Proper Data Splitting and Testing
==============================================================

Updated main script that uses proper data splitting instead of fixed 2023/2024 splits.
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import our modules
from config import config
from dataset_builder import PregameDatasetBuilder
from modeling import EnhancedDualModelSystem  # Use the enhanced version
from live_prediction_system import LivePredictionSystem, create_live_system
from betting_utils import BettingAnalyzer, PortfolioManager

# Import the new testing framework
try:
    from model_testing import ModelTester, run_model_testing_suite, quick_model_diagnosis
    TESTING_AVAILABLE = True
    print("Testing framework successfully imported!")
except ImportError as e:
    print(f"Testing framework import failed: {e}")
    print("Please save the model_testing.py file to your project directory")
    TESTING_AVAILABLE = False
except Exception as e:
    print(f"Unexpected error importing testing framework: {e}")
    TESTING_AVAILABLE = False

logger = logging.getLogger(__name__)

def train_models(start_date: str = None, end_date: str = None, 
                force_rebuild: bool = False, splitting_strategy: str = 'time_based',
                test_size: float = 0.2, val_size: float = 0.1,
                gap_days: int = 7, test_seasons: str = None,
                cross_validate: bool = True) -> EnhancedDualModelSystem:
    """
    Train the enhanced dual model system with proper data splitting.
    
    Args:
        start_date: Training data start date
        end_date: Training data end date
        force_rebuild: Force rebuild of dataset
        splitting_strategy: 'time_based', 'random', or 'seasonal'
        test_size: Proportion for test set
        val_size: Proportion for validation set
        gap_days: Days gap between splits (time_based only)
        test_seasons: Comma-separated seasons for test (seasonal only)
        cross_validate: Whether to perform cross-validation
    
    Returns:
        Trained EnhancedDualModelSystem
    """
    logger.info("="*70)
    logger.info("TRAINING ENHANCED DUAL MODEL SYSTEM")
    logger.info("="*70)
    
    # Set default dates
    start_date = start_date or config.DEFAULT_START_DATE
    end_date = end_date or config.DEFAULT_END_DATE
    
    try:
        # Build dataset
        logger.info(f"Building training dataset: {start_date} to {end_date}")
        builder = PregameDatasetBuilder(start_date=start_date, end_date=end_date)
        
        dataset = builder.build_dataset(
            force_rebuild=force_rebuild,
            cache_format=config.CACHE_FORMAT
        )
        
        logger.info(f"Dataset built: {len(dataset)} rows, {len(dataset.columns)} columns")
        
        # Validate dataset
        if dataset.empty:
            raise ValueError("Empty dataset - cannot train models")
        
        # Add season column if not present
        if 'season' not in dataset.columns:
            dataset['season'] = pd.to_datetime(dataset['date']).dt.year
        
        # Parse test seasons if provided
        test_seasons_list = None
        if test_seasons and splitting_strategy == 'seasonal':
            test_seasons_list = [int(s.strip()) for s in test_seasons.split(',')]
            logger.info(f"Using test seasons: {test_seasons_list}")
        
        # Initialize enhanced model system
        model_system = EnhancedDualModelSystem(
            model_dir=config.MODEL_DIR,
            splitting_strategy=splitting_strategy
        )
        
        # Train with enhanced splitting
        training_results = model_system.fit(
            dataset,
            splitting_strategy=splitting_strategy,
            test_size=test_size,
            val_size=val_size,
            gap_days=gap_days,
            test_seasons=test_seasons_list,
            cross_validate=cross_validate
        )
        
        # Print training summary
        logger.info("\n" + "="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        logger.info(f"Split strategy: {training_results['split_strategy']}")
        logger.info(f"Train size: {training_results['train_size']}")
        logger.info(f"Validation size: {training_results['val_size']}")
        logger.info(f"Test size: {training_results['test_size']}")
        
        if 'test_metrics' in training_results:
            test_metrics = training_results['test_metrics'].get('system', {})
            if test_metrics:
                logger.info(f"\nTest Performance:")
                logger.info(f"  ROC AUC: {test_metrics.get('roc_auc', 0):.4f}")
                logger.info(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
                logger.info(f"  Precision: {test_metrics.get('precision', 0):.4f}")
                logger.info(f"  Recall: {test_metrics.get('recall', 0):.4f}")
        
        if 'cv_results' in training_results:
            cv_results = training_results['cv_results']
            if 'core' in cv_results and 'xgb' in cv_results['core']:
                cv_auc = cv_results['core']['xgb'].get('roc_auc_mean', 0)
                cv_std = cv_results['core']['xgb'].get('roc_auc_std', 0)
                logger.info(f"\nCross-validation AUC: {cv_auc:.4f} ± {cv_std:.4f}")
        
        # Save models
        years_in_data = sorted(dataset['season'].unique()) if 'season' in dataset.columns else []
        model_system.save(train_years=years_in_data)
        
        logger.info("Enhanced model training completed successfully!")
        return model_system
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

def test_models(start_date: str = None, end_date: str = None,
               model_dir: str = None, output_dir: str = "test_results",
               splitting_strategy: str = 'time_based') -> dict:
    """
    Run comprehensive model testing.
    
    Args:
        start_date: Start date for test data
        end_date: End date for test data
        model_dir: Model directory to test
        output_dir: Output directory for test results
        splitting_strategy: Splitting strategy to test
    
    Returns:
        Test results dictionary
    """
    logger.info("="*70)
    logger.info("COMPREHENSIVE MODEL TESTING")
    logger.info("="*70)
    
    if not TESTING_AVAILABLE:
        logger.error("Testing framework not available. Please install model_testing module.")
        return {'error': 'Testing framework not available'}
    
    try:
        # Run comprehensive test suite
        results = run_model_testing_suite(
            start_date=start_date,
            end_date=end_date,
            model_dir=model_dir or config.MODEL_DIR,
            output_dir=output_dir,
            splitting_strategy=splitting_strategy,
            save_plots=True
        )
        
        logger.info("Model testing completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Model testing failed: {e}")
        return {'error': str(e)}

def quick_diagnosis(model_dir: str = None) -> dict:
    """
    Run quick model diagnosis on latest available data.
    
    Args:
        model_dir: Model directory
    
    Returns:
        Quick diagnosis results
    """
    logger.info("="*70)
    logger.info("QUICK MODEL DIAGNOSIS")
    logger.info("="*70)
    
    if not TESTING_AVAILABLE:
        logger.error("Testing framework not available.")
        return {'error': 'Testing framework not available'}
    
    try:
        # Load model system
        model_system = EnhancedDualModelSystem(model_dir=model_dir or config.MODEL_DIR)
        model_system.load()
        
        # Get recent data for testing (last 30 days of available data)
        builder = PregameDatasetBuilder()
        recent_data = builder.build_dataset(force_rebuild=False)
        
        if recent_data.empty:
            return {'error': 'No data available for diagnosis'}
        
        # Use the most recent 1000 samples for quick diagnosis
        test_data = recent_data.tail(1000).copy()
        
        # Run quick diagnosis
        diagnosis = quick_model_diagnosis(model_system, test_data)
        
        # Print results
        if 'error' not in diagnosis:
            logger.info("\nQUICK DIAGNOSIS RESULTS:")
            logger.info(f"Overall Status: {diagnosis.get('overall_status', 'UNKNOWN')}")
            logger.info(f"ROC AUC: {diagnosis['metrics']['roc_auc']:.4f}")
            logger.info(f"Accuracy: {diagnosis['metrics']['accuracy']:.4f}")
            logger.info(f"Calibration Error: {diagnosis['calibration_error']:.4f}")
            
            if diagnosis['assessment']:
                logger.info("\nIssues Found:")
                for issue in diagnosis['assessment']:
                    logger.info(f"  - {issue}")
            else:
                logger.info("\nNo issues detected in quick diagnosis.")
        
        return diagnosis
        
    except Exception as e:
        logger.error(f"Quick diagnosis failed: {e}")
        return {'error': str(e)}

def run_live_predictions(date: str = None, api_key: str = None,
                        min_ev: float = None, min_prob: float = None) -> dict:
    """Run live predictions and betting analysis (unchanged from original)."""
    logger.info("="*70)
    logger.info("LIVE PREDICTION ANALYSIS")
    logger.info("="*70)
    
    try:
        # Create and initialize live system
        live_system = create_live_system(api_key=api_key)
        
        # Set custom thresholds if provided
        if min_ev is not None:
            live_system.betting_analyzer.min_ev = min_ev
        if min_prob is not None:
            live_system.betting_analyzer.min_probability = min_prob
        
        # Run full analysis
        results = live_system.run_full_analysis(target_date=date, print_results=True)
        
        if results['success']:
            logger.info("Live prediction analysis completed successfully!")
        else:
            logger.warning("Live prediction analysis completed with errors")
        
        return results
        
    except Exception as e:
        logger.error(f"Live prediction analysis failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def run_portfolio_analysis(date: str = None, api_key: str = None,
                          bankroll: float = 1000.0) -> dict:
    """Run portfolio analysis (unchanged from original)."""
    logger.info("="*70)
    logger.info("PORTFOLIO ANALYSIS")
    logger.info("="*70)
    
    try:
        # Get betting opportunities
        live_system = create_live_system(api_key=api_key)
        opportunities = live_system.get_todays_betting_opportunities(date)
        
        if not opportunities:
            logger.info("No betting opportunities found for portfolio analysis")
            return {'success': True, 'opportunities': 0, 'message': 'No opportunities'}
        
        # Initialize portfolio manager
        portfolio_manager = PortfolioManager(bankroll=bankroll)
        
        # Get bet sizing recommendations
        bet_portfolio = portfolio_manager.evaluate_bet_sizing(opportunities)
        
        if bet_portfolio.empty:
            logger.info("No bets recommended after portfolio constraints")
            return {'success': True, 'recommended_bets': 0}
        
        # Run Monte Carlo simulation
        logger.info("Running portfolio simulation...")
        simulation_results = portfolio_manager.simulate_outcomes(bet_portfolio)
        
        # Print results
        logger.info(f"\n{'='*60}")
        logger.info("PORTFOLIO RECOMMENDATIONS")
        logger.info(f"{'='*60}")
        
        logger.info(f"\nBankroll: ${bankroll:,.2f}")
        logger.info(f"Recommended bets: {len(bet_portfolio)}")
        logger.info(f"Total allocation: ${bet_portfolio['recommended_bet'].sum():,.2f}")
        logger.info(f"Portfolio allocation: {bet_portfolio['recommended_bet'].sum()/bankroll:.1%}")
        
        # Print individual recommendations
        if len(bet_portfolio) > 0:
            logger.info(f"\nBET RECOMMENDATIONS:")
            for _, bet in bet_portfolio.iterrows():
                logger.info(f"  {bet['player']}: ${bet['recommended_bet']:.0f} "
                          f"(EV: {bet['ev']:+.3f}, Kelly: {bet['kelly_fraction']:.1%})")
        
        # Print simulation results
        if simulation_results:
            logger.info(f"\nSIMULATION RESULTS (1000 trials):")
            logger.info(f"  Expected profit: ${simulation_results['mean_profit']:+,.2f}")
            logger.info(f"  Probability of profit: {simulation_results['prob_profit']:.1%}")
            logger.info(f"  95% confidence interval: ${simulation_results['percentile_5']:+,.2f} "
                       f"to ${simulation_results['percentile_95']:+,.2f}")
            logger.info(f"  Max potential loss: ${simulation_results['max_loss']:+,.2f}")
        
        return {
            'success': True,
            'bankroll': bankroll,
            'bet_portfolio': bet_portfolio.to_dict('records'),
            'simulation_results': simulation_results,
            'summary': {
                'total_bets': len(bet_portfolio),
                'total_allocation': bet_portfolio['recommended_bet'].sum(),
                'expected_profit': simulation_results.get('mean_profit', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        return {'success': False, 'error': str(e)}

def run_backtest(start_date: str = "2024-01-01", end_date: str = "2024-10-01") -> dict:
    """Run historical backtest (unchanged from original)."""
    logger.info("="*70)
    logger.info("HISTORICAL BACKTEST")
    logger.info("="*70)
    
    try:
        # Load trained models
        model_system = EnhancedDualModelSystem(config.MODEL_DIR)
        model_system.load()
        
        # Build backtest dataset
        logger.info(f"Building backtest dataset: {start_date} to {end_date}")
        builder = PregameDatasetBuilder(start_date=start_date, end_date=end_date)
        backtest_data = builder.build_dataset(force_rebuild=False)
        
        if backtest_data.empty:
            raise ValueError("No backtest data available")
        
        # Generate predictions
        logger.info("Generating backtest predictions...")
        probabilities = model_system.predict_proba(backtest_data)
        
        # Evaluate performance
        metrics = model_system.evaluate_comprehensive(backtest_data, "Backtest")
        
        # Calculate additional statistics
        actual_hr_rate = backtest_data['hit_hr'].mean()
        predicted_hr_rate = probabilities.mean()
        
        logger.info(f"\nBACKTEST SUMMARY:")
        logger.info(f"  Period: {start_date} to {end_date}")
        logger.info(f"  Total games: {len(backtest_data)}")
        logger.info(f"  Actual HR rate: {actual_hr_rate:.3f}")
        logger.info(f"  Predicted HR rate: {predicted_hr_rate:.3f}")
        
        if 'system' in metrics:
            logger.info(f"\nPERFORMANCE METRICS:")
            for metric, value in metrics['system'].items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return {
            'success': True,
            'period': {'start': start_date, 'end': end_date},
            'total_games': len(backtest_data),
            'actual_hr_rate': actual_hr_rate,
            'predicted_hr_rate': predicted_hr_rate,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {'success': False, 'error': str(e)}

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced Baseball Home Run Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models with time-based splitting
  python main.py train --start-date 2023-03-01 --end-date 2024-10-01 --splitting-strategy time_based --gap-days 7
  
  # Train with seasonal splitting  
  python main.py train --splitting-strategy seasonal --test-seasons "2024"
  
  # Run comprehensive testing
  python main.py test --output-dir my_test_results
  
  # Quick model diagnosis
  python main.py diagnose
  
  # Run live predictions
  python main.py live --api-key YOUR_API_KEY --min-ev 0.05
  
  # Portfolio analysis
  python main.py portfolio --bankroll 5000 --api-key YOUR_API_KEY
  
  # Backtest
  python main.py backtest --start-date 2024-01-01 --end-date 2024-10-01
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command - enhanced with splitting options
    train_parser = subparsers.add_parser('train', help='Train models with enhanced splitting')
    train_parser.add_argument('--start-date', help='Training start date (YYYY-MM-DD)')
    train_parser.add_argument('--end-date', help='Training end date (YYYY-MM-DD)')
    train_parser.add_argument('--force-rebuild', action='store_true', 
                             help='Force rebuild of dataset')
    train_parser.add_argument('--splitting-strategy', default='time_based',
                             choices=['time_based', 'random', 'seasonal'],
                             help='Data splitting strategy')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                             help='Test set proportion (0.0-1.0)')
    train_parser.add_argument('--val-size', type=float, default=0.1,
                             help='Validation set proportion (0.0-1.0)')
    train_parser.add_argument('--gap-days', type=int, default=7,
                             help='Days gap between splits (time_based only)')
    train_parser.add_argument('--test-seasons', 
                             help='Comma-separated test seasons (seasonal only)')
    train_parser.add_argument('--no-cv', action='store_true',
                             help='Skip cross-validation')
    
    # Test command - new comprehensive testing
    test_parser = subparsers.add_parser('test', help='Run comprehensive model testing')
    test_parser.add_argument('--start-date', help='Test data start date (YYYY-MM-DD)')
    test_parser.add_argument('--end-date', help='Test data end date (YYYY-MM-DD)')
    test_parser.add_argument('--model-dir', help='Model directory to test')
    test_parser.add_argument('--output-dir', default='test_results', 
                            help='Output directory for test results')
    test_parser.add_argument('--splitting-strategy', default='time_based',
                            choices=['time_based', 'random', 'seasonal'],
                            help='Splitting strategy to test')
    
    # Diagnose command - new quick diagnosis
    diagnose_parser = subparsers.add_parser('diagnose', help='Quick model diagnosis')
    diagnose_parser.add_argument('--model-dir', help='Model directory')
    
    # Live predictions command (unchanged)
    live_parser = subparsers.add_parser('live', help='Run live predictions')
    live_parser.add_argument('--date', help='Target date (YYYY-MM-DD), defaults to today')
    live_parser.add_argument('--api-key', help='The Odds API key')
    live_parser.add_argument('--min-ev', type=float, help='Minimum expected value threshold')
    live_parser.add_argument('--min-prob', type=float, help='Minimum probability threshold')
    
    # Portfolio analysis command (unchanged)
    portfolio_parser = subparsers.add_parser('portfolio', help='Portfolio analysis')
    portfolio_parser.add_argument('--date', help='Target date (YYYY-MM-DD)')
    portfolio_parser.add_argument('--api-key', help='The Odds API key')
    portfolio_parser.add_argument('--bankroll', type=float, default=1000.0, 
                                 help='Total bankroll for bet sizing')
    
    # Backtest command (unchanged)
    backtest_parser = subparsers.add_parser('backtest', help='Run historical backtest')
    backtest_parser.add_argument('--start-date', default='2024-01-01', 
                                help='Backtest start date')
    backtest_parser.add_argument('--end-date', default='2024-10-01', 
                                help='Backtest end date')
    
    # Global options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', help='Log file path (in addition to console)')
    
    return parser

def setup_logging(log_level: str, log_file: str = None):
    """Setup logging configuration (unchanged from original)."""
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def main():
    """Enhanced main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logger.info("Enhanced Baseball HR Prediction System starting...")
    logger.info(f"Configuration: {config.MODEL_DIR}")
    
    try:
        if args.command == 'train':
            results = train_models(
                start_date=args.start_date,
                end_date=args.end_date,
                force_rebuild=args.force_rebuild,
                splitting_strategy=args.splitting_strategy,
                test_size=args.test_size,
                val_size=args.val_size,
                gap_days=args.gap_days,
                test_seasons=args.test_seasons,
                cross_validate=not args.no_cv
            )
            
        elif args.command == 'test':
            results = test_models(
                start_date=args.start_date,
                end_date=args.end_date,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                splitting_strategy=args.splitting_strategy
            )
            
        elif args.command == 'diagnose':
            results = quick_diagnosis(
                model_dir=args.model_dir
            )
            
        elif args.command == 'live':
            results = run_live_predictions(
                date=args.date,
                api_key=args.api_key or config.THEODDS_API_KEY,
                min_ev=args.min_ev,
                min_prob=args.min_prob
            )
            
        elif args.command == 'portfolio':
            results = run_portfolio_analysis(
                date=args.date,
                api_key=args.api_key or config.THEODDS_API_KEY,
                bankroll=args.bankroll
            )
            
        elif args.command == 'backtest':
            results = run_backtest(
                start_date=args.start_date,
                end_date=args.end_date
            )
            
        else:
            parser.print_help()
            sys.exit(1)
        
        # Print final status
        if hasattr(results, 'get') and results.get('success', True):
            logger.info("Operation completed successfully!")
            sys.exit(0)
        else:
            logger.error("Operation completed with errors")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

# Interactive mode functions for Jupyter/development
def quick_train(splitting_strategy: str = 'time_based', cross_validate: bool = True):
    """Quick training for development/testing with enhanced options."""
    return train_models(
        force_rebuild=False,
        splitting_strategy=splitting_strategy,
        cross_validate=cross_validate
    )

def quick_live_analysis(api_key: str = None):
    """Quick live analysis for development/testing (unchanged)."""
    return run_live_predictions(api_key=api_key or config.THEODDS_API_KEY)

def quick_test():
    """Quick model testing for development."""
    if TESTING_AVAILABLE:
        return test_models(output_dir="quick_test_results")
    else:
        logger.error("Testing framework not available")
        return {'error': 'Testing framework not available'}

def demo_enhanced_system():
    """Demonstrate enhanced system capabilities."""
    logger.info("="*70)
    logger.info("ENHANCED BASEBALL HR PREDICTION SYSTEM DEMO")
    logger.info("="*70)
    
    try:
        # Check if models exist
        model_files = config.get_file_paths()
        if not model_files['core_model'].exists():
            logger.info("No trained models found. Training with enhanced splitting...")
            train_models(splitting_strategy='time_based', cross_validate=True)
        
        # Run quick diagnosis
        logger.info("Running quick model diagnosis...")
        diagnosis = quick_diagnosis()
        
        if 'error' not in diagnosis:
            logger.info(f"Model Status: {diagnosis.get('overall_status', 'UNKNOWN')}")
            
            # Run comprehensive testing if available
            if TESTING_AVAILABLE:
                logger.info("Running comprehensive model testing...")
                test_results = test_models(output_dir="demo_test_results")
                
                if 'recommendations' in test_results:
                    recommendations = test_results['recommendations']
                    logger.info(f"Generated {len(recommendations)} recommendations")
                    if recommendations:
                        logger.info("Top recommendations:")
                        for i, rec in enumerate(recommendations[:3], 1):
                            logger.info(f"  {i}. {rec}")
        
        # Run live analysis if API key available
        if config.validate_api_key():
            logger.info("Running live analysis demo...")
            results = run_live_predictions()
            
            if results.get('success'):
                logger.info("Demo completed successfully!")
            else:
                logger.warning("Demo completed with some issues")
        else:
            logger.info("No API key available. Skipping live analysis.")
            logger.info("Set THEODDS_API_KEY environment variable for full demo.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")

if __name__ == "__main__":
    main()