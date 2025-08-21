"""
Monitor Comparative Analysis Progress
===================================

Simple monitoring script to track the progress of the long-running comparative analysis.
"""

import time
import os
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_analysis_progress(check_interval: int = 300):  # Check every 5 minutes
    """Monitor the comparative analysis progress."""
    
    logger.info("="*60)
    logger.info("COMPARATIVE ANALYSIS PROGRESS MONITOR")
    logger.info("="*60)
    logger.info("Monitoring analysis progress...")
    logger.info(f"Checking every {check_interval/60:.1f} minutes")
    logger.info("Press Ctrl+C to stop monitoring")
    
    start_time = datetime.now()
    check_count = 0
    
    try:
        while True:
            check_count += 1
            elapsed = datetime.now() - start_time
            
            logger.info(f"\\n--- Check #{check_count} ({elapsed}) ---")
            
            # Check for result files
            result_files = []
            for file in os.listdir('/home/charlesbenfer/betting_models/'):
                if file.startswith('comparative_analysis_results') and file.endswith('.json'):
                    result_files.append(file)
            
            if result_files:
                logger.info(f"✅ Found {len(result_files)} result file(s):")
                for file in result_files:
                    file_path = f'/home/charlesbenfer/betting_models/{file}'
                    file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    logger.info(f"  📄 {file} ({file_size:.1f} MB, modified: {mod_time.strftime('%H:%M:%S')})")
                    
                    # Try to read progress from summary file
                    if 'summary' in file:
                        try:
                            with open(file_path, 'r') as f:
                                summary = json.load(f)
                            experiments = summary.get('total_experiments', 0)
                            best_perf = summary.get('best_performance', {})
                            roi = summary.get('feature_engineering_roi', 0)
                            
                            logger.info(f"  📊 Progress: {experiments} experiments completed")
                            if best_perf:
                                logger.info(f"  🏆 Best: {best_perf.get('name', 'unknown')} (ROC-AUC: {best_perf.get('roc_auc', 0):.4f})")
                            logger.info(f"  💰 ROI: {roi:.2f}")
                        except:
                            pass
            else:
                logger.info("⏳ No result files found yet - analysis still running...")
            
            # Check log files for recent activity
            log_files = ['dataset_builder.log', 'modeling.log', 'weather_scraper.log']
            for log_file in log_files:
                log_path = f'/home/charlesbenfer/betting_models/{log_file}'
                if os.path.exists(log_path):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(log_path))
                    age_minutes = (datetime.now() - mod_time).total_seconds() / 60
                    if age_minutes < 10:  # Recently modified
                        logger.info(f"📝 {log_file} recently active ({age_minutes:.1f} min ago)")
            
            # System resource info (if available)
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                logger.info(f"💻 System: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}% used")
            except ImportError:
                pass
            
            logger.info(f"Next check in {check_interval/60:.1f} minutes...")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("\\n🛑 Monitoring stopped by user")
        total_time = datetime.now() - start_time
        logger.info(f"Total monitoring time: {total_time}")
        
    except Exception as e:
        logger.error(f"Monitoring error: {e}")

def show_latest_results():
    """Show the latest results if available."""
    
    logger.info("="*60)
    logger.info("LATEST ANALYSIS RESULTS")
    logger.info("="*60)
    
    # Find most recent results
    result_files = []
    for file in os.listdir('/home/charlesbenfer/betting_models/'):
        if file.startswith('comparative_analysis_results') and file.endswith('_summary.json'):
            file_path = f'/home/charlesbenfer/betting_models/{file}'
            mod_time = os.path.getmtime(file_path)
            result_files.append((file, mod_time))
    
    if not result_files:
        logger.info("No results found yet.")
        return
    
    # Get most recent
    latest_file = max(result_files, key=lambda x: x[1])[0]
    file_path = f'/home/charlesbenfer/betting_models/{latest_file}'
    
    try:
        with open(file_path, 'r') as f:
            summary = json.load(f)
        
        logger.info(f"📄 Latest results: {latest_file}")
        logger.info(f"📅 Training period: {summary.get('training_period', 'unknown')}")
        logger.info(f"📅 Test period: {summary.get('test_period', 'unknown')}")
        logger.info(f"📊 Experiments: {summary.get('total_experiments', 0)}")
        
        best_perf = summary.get('best_performance', {})
        if best_perf:
            logger.info(f"🏆 Best performance: {best_perf.get('name', 'unknown')} (ROC-AUC: {best_perf.get('roc_auc', 0):.4f})")
        
        roi = summary.get('feature_engineering_roi', 0)
        logger.info(f"💰 Feature engineering ROI: {roi:.2f}")
        
        recommendations = summary.get('top_recommendations', [])
        if recommendations:
            logger.info("\\n🎯 Top recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                logger.info(f"  {i}. {rec}")
                
    except Exception as e:
        logger.error(f"Error reading results: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "results":
        show_latest_results()
    else:
        monitor_analysis_progress()