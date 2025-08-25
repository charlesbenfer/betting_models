#!/usr/bin/env python3
"""
Performance Testing Suite
=========================

Comprehensive performance testing for the baseball HR prediction system.
"""

import os
import sys
import time
import json
import threading
import statistics
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from modeling import EnhancedDualModelSystem
from live_prediction_system import LivePredictionSystem
from data_utils import DataValidator

class PerformanceTestSuite:
    """Comprehensive performance testing suite."""
    
    def __init__(self):
        self.results = {}
        self.test_data = {}
        self.baseline_metrics = {}
        
        # Performance thresholds
        self.thresholds = {
            'prediction_latency_ms': 2000,  # 2 seconds max
            'batch_prediction_throughput': 100,  # predictions per minute
            'api_response_time_ms': 5000,  # 5 seconds max
            'memory_usage_mb': 2000,  # 2GB max
            'cpu_usage_percent': 80,  # 80% max
            'concurrent_requests': 10  # max concurrent requests
        }
        
        print("🏃 Performance testing suite initialized")
    
    def run_all_performance_tests(self) -> Dict[str, Any]:
        """Run complete performance test suite."""
        print("🚀 Starting comprehensive performance testing...")
        
        # Initialize system
        if not self._initialize_test_environment():
            return {'status': 'failed', 'error': 'Failed to initialize test environment'}
        
        test_results = {}
        
        # Run individual test categories
        test_categories = [
            ('prediction_latency', self._test_prediction_latency),
            ('batch_throughput', self._test_batch_throughput),
            ('concurrent_predictions', self._test_concurrent_predictions),
            ('memory_usage', self._test_memory_usage),
            ('api_performance', self._test_api_performance),
            ('load_testing', self._test_load_scenarios),
            ('stress_testing', self._test_stress_scenarios)
        ]
        
        for category, test_func in test_categories:
            print(f"\n🔄 Running {category} tests...")
            try:
                test_results[category] = test_func()
                status = "✅ PASS" if test_results[category]['passed'] else "❌ FAIL"
                print(f"{status} {category}")
            except Exception as e:
                test_results[category] = {
                    'passed': False, 
                    'error': str(e),
                    'metrics': {}
                }
                print(f"❌ FAIL {category}: {e}")
        
        # Generate overall report
        overall_report = self._generate_performance_report(test_results)
        
        # Export results
        self._export_performance_results(overall_report)
        
        return overall_report
    
    def _initialize_test_environment(self) -> bool:
        """Initialize testing environment."""
        try:
            print("🔧 Initializing test environment...")
            
            # Initialize models
            self.model_system = EnhancedDualModelSystem()
            self.live_system = LivePredictionSystem()
            self.data_validator = DataValidator()
            
            # Load models if available
            model_files = list(Path("models").glob("*.joblib"))
            if model_files:
                latest_model = max(model_files, key=os.path.getctime)
                self.model_system.load_model(str(latest_model))
                print(f"📁 Loaded model: {latest_model.name}")
            
            # Prepare test data
            self._prepare_test_data()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize test environment: {e}")
            return False
    
    def _prepare_test_data(self):
        """Prepare test datasets for performance testing."""
        try:
            # Try to load existing data
            import os
            if os.path.exists('data/processed_games.csv'):
                sample_data = pd.read_csv('data/processed_games.csv').head(100)
            elif os.path.exists('data/games.csv'):
                sample_data = pd.read_csv('data/games.csv').head(100)
            else:
                sample_data = pd.DataFrame()
            
            if sample_data.empty:
                # Generate synthetic test data
                import pandas as pd
                import numpy as np
                
                sample_data = pd.DataFrame({
                    'game_date': [datetime.now().strftime('%Y-%m-%d')] * 50,
                    'home_team': ['LAD'] * 50,
                    'away_team': ['NYY'] * 50,
                    'temperature': np.random.normal(75, 10, 50),
                    'humidity': np.random.normal(60, 15, 50),
                    'wind_speed': np.random.normal(8, 3, 50),
                    'home_runs': np.random.poisson(2.2, 50)
                })
            
            self.test_data['sample_games'] = sample_data
            print(f"📊 Prepared {len(sample_data)} test records")
            
        except Exception as e:
            print(f"⚠️  Using minimal test data due to error: {e}")
            # Minimal fallback data
            import pandas as pd
            self.test_data['sample_games'] = pd.DataFrame({
                'game_date': ['2024-08-22'],
                'home_team': ['LAD'],
                'away_team': ['NYY']
            })
    
    def _test_prediction_latency(self) -> Dict[str, Any]:
        """Test individual prediction latency."""
        latencies = []
        
        # Test multiple individual predictions
        for i in range(20):
            start_time = time.time()
            
            try:
                # Make a single prediction
                test_game = self.test_data['sample_games'].iloc[i % len(self.test_data['sample_games'])]
                prediction = self.live_system.get_todays_predictions()
                
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
                
            except Exception as e:
                print(f"⚠️  Prediction failed: {e}")
                latencies.append(self.thresholds['prediction_latency_ms'] + 1000)  # Penalty
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max_latency
        
        passed = avg_latency <= self.thresholds['prediction_latency_ms']
        
        return {
            'passed': passed,
            'metrics': {
                'average_latency_ms': round(avg_latency, 2),
                'max_latency_ms': round(max_latency, 2),
                'min_latency_ms': round(min_latency, 2),
                'p95_latency_ms': round(p95_latency, 2),
                'threshold_ms': self.thresholds['prediction_latency_ms'],
                'total_predictions': len(latencies)
            }
        }
    
    def _test_batch_throughput(self) -> Dict[str, Any]:
        """Test batch prediction throughput."""
        start_time = time.time()
        predictions_made = 0
        
        try:
            # Simulate batch predictions for 60 seconds
            end_time = start_time + 60  # 1 minute test
            
            while time.time() < end_time:
                # Make batch predictions
                predictions = self.live_system.get_todays_predictions()
                predictions_made += len(predictions) if isinstance(predictions, (list, dict)) else 1
                
                # Small delay to prevent overwhelming system
                time.sleep(0.1)
        
        except Exception as e:
            print(f"⚠️  Batch throughput test error: {e}")
        
        actual_duration = time.time() - start_time
        throughput = (predictions_made / actual_duration) * 60  # per minute
        
        passed = throughput >= self.thresholds['batch_prediction_throughput']
        
        return {
            'passed': passed,
            'metrics': {
                'throughput_per_minute': round(throughput, 2),
                'total_predictions': predictions_made,
                'test_duration_seconds': round(actual_duration, 2),
                'threshold_per_minute': self.thresholds['batch_prediction_throughput']
            }
        }
    
    def _test_concurrent_predictions(self) -> Dict[str, Any]:
        """Test concurrent prediction handling."""
        concurrent_requests = self.thresholds['concurrent_requests']
        response_times = []
        errors = 0
        
        def make_prediction():
            start_time = time.time()
            try:
                predictions = self.live_system.get_todays_predictions()
                response_time = (time.time() - start_time) * 1000
                return response_time
            except Exception as e:
                print(f"⚠️  Concurrent prediction error: {e}")
                return None
        
        # Run concurrent predictions
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_prediction) for _ in range(concurrent_requests * 2)]
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    response_times.append(result)
                else:
                    errors += 1
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = float('inf')
            max_response_time = float('inf')
        
        success_rate = (len(response_times) / (len(response_times) + errors)) * 100
        passed = success_rate >= 90 and avg_response_time <= self.thresholds['prediction_latency_ms'] * 2
        
        return {
            'passed': passed,
            'metrics': {
                'concurrent_requests': concurrent_requests,
                'success_rate_percent': round(success_rate, 2),
                'average_response_time_ms': round(avg_response_time, 2),
                'max_response_time_ms': round(max_response_time, 2),
                'total_errors': errors,
                'successful_requests': len(response_times)
            }
        }
    
    def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage during operations."""
        process = psutil.Process()
        memory_readings = []
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_readings.append(baseline_memory)
        
        try:
            # Memory test during predictions
            for i in range(10):
                predictions = self.live_system.get_todays_predictions()
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_readings.append(current_memory)
                time.sleep(1)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Final memory reading
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_readings.append(final_memory)
            
        except Exception as e:
            print(f"⚠️  Memory test error: {e}")
        
        max_memory = max(memory_readings)
        avg_memory = statistics.mean(memory_readings)
        memory_growth = max_memory - baseline_memory
        
        passed = max_memory <= self.thresholds['memory_usage_mb']
        
        return {
            'passed': passed,
            'metrics': {
                'baseline_memory_mb': round(baseline_memory, 2),
                'max_memory_mb': round(max_memory, 2),
                'average_memory_mb': round(avg_memory, 2),
                'memory_growth_mb': round(memory_growth, 2),
                'final_memory_mb': round(final_memory, 2),
                'threshold_mb': self.thresholds['memory_usage_mb']
            }
        }
    
    def _test_api_performance(self) -> Dict[str, Any]:
        """Test API performance and connectivity."""
        api_response_times = []
        api_errors = 0
        
        # Test API calls
        for i in range(5):
            start_time = time.time()
            try:
                # Test live prediction system (which includes API calls)
                predictions = self.live_system.get_todays_predictions()
                response_time = (time.time() - start_time) * 1000
                api_response_times.append(response_time)
            except Exception as e:
                print(f"⚠️  API test error: {e}")
                api_errors += 1
                api_response_times.append(self.thresholds['api_response_time_ms'])
            
            time.sleep(2)  # Rate limiting
        
        if api_response_times:
            avg_api_time = statistics.mean(api_response_times)
            max_api_time = max(api_response_times)
        else:
            avg_api_time = float('inf')
            max_api_time = float('inf')
        
        success_rate = ((len(api_response_times) - api_errors) / len(api_response_times)) * 100 if api_response_times else 0
        passed = success_rate >= 80 and avg_api_time <= self.thresholds['api_response_time_ms']
        
        return {
            'passed': passed,
            'metrics': {
                'average_response_time_ms': round(avg_api_time, 2),
                'max_response_time_ms': round(max_api_time, 2),
                'api_success_rate_percent': round(success_rate, 2),
                'total_api_calls': len(api_response_times),
                'api_errors': api_errors,
                'threshold_ms': self.thresholds['api_response_time_ms']
            }
        }
    
    def _test_load_scenarios(self) -> Dict[str, Any]:
        """Test system under normal load scenarios."""
        load_metrics = {
            'scenario_1_light_load': self._run_load_scenario(requests_per_minute=60, duration_minutes=2),
            'scenario_2_moderate_load': self._run_load_scenario(requests_per_minute=120, duration_minutes=1),
            'scenario_3_peak_load': self._run_load_scenario(requests_per_minute=200, duration_minutes=1)
        }
        
        # Evaluate overall load performance
        all_passed = all(scenario['passed'] for scenario in load_metrics.values())
        
        return {
            'passed': all_passed,
            'metrics': load_metrics
        }
    
    def _run_load_scenario(self, requests_per_minute: int, duration_minutes: float) -> Dict[str, Any]:
        """Run specific load scenario."""
        interval = 60.0 / requests_per_minute  # seconds between requests
        total_requests = int(requests_per_minute * duration_minutes)
        
        response_times = []
        errors = 0
        start_time = time.time()
        
        for i in range(total_requests):
            request_start = time.time()
            
            try:
                predictions = self.live_system.get_todays_predictions()
                response_time = (time.time() - request_start) * 1000
                response_times.append(response_time)
            except Exception as e:
                errors += 1
            
            # Wait for next request
            elapsed = time.time() - request_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        actual_duration = time.time() - start_time
        
        if response_times:
            avg_response = statistics.mean(response_times)
            max_response = max(response_times)
        else:
            avg_response = float('inf')
            max_response = float('inf')
        
        success_rate = (len(response_times) / total_requests) * 100
        passed = success_rate >= 95 and avg_response <= self.thresholds['prediction_latency_ms'] * 1.5
        
        return {
            'passed': passed,
            'requests_per_minute': requests_per_minute,
            'duration_minutes': duration_minutes,
            'total_requests': total_requests,
            'successful_requests': len(response_times),
            'errors': errors,
            'success_rate_percent': round(success_rate, 2),
            'average_response_time_ms': round(avg_response, 2),
            'max_response_time_ms': round(max_response, 2),
            'actual_duration_seconds': round(actual_duration, 2)
        }
    
    def _test_stress_scenarios(self) -> Dict[str, Any]:
        """Test system under stress conditions."""
        print("⚡ Running stress test scenarios...")
        
        stress_results = {}
        
        # Stress Test 1: Resource exhaustion
        try:
            stress_results['resource_exhaustion'] = self._stress_test_resources()
        except Exception as e:
            stress_results['resource_exhaustion'] = {'passed': False, 'error': str(e)}
        
        # Stress Test 2: Burst requests
        try:
            stress_results['burst_requests'] = self._stress_test_burst()
        except Exception as e:
            stress_results['burst_requests'] = {'passed': False, 'error': str(e)}
        
        overall_passed = all(
            result.get('passed', False) for result in stress_results.values()
        )
        
        return {
            'passed': overall_passed,
            'metrics': stress_results
        }
    
    def _stress_test_resources(self) -> Dict[str, Any]:
        """Stress test system resources."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        max_memory = initial_memory
        
        # Run intensive operations
        for i in range(50):
            try:
                predictions = self.live_system.get_todays_predictions()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                
                if max_memory > self.thresholds['memory_usage_mb'] * 1.5:
                    break
                    
            except Exception as e:
                print(f"⚠️  Resource stress test error: {e}")
                break
        
        memory_growth = max_memory - initial_memory
        passed = max_memory <= self.thresholds['memory_usage_mb'] * 1.2
        
        return {
            'passed': passed,
            'initial_memory_mb': round(initial_memory, 2),
            'max_memory_mb': round(max_memory, 2),
            'memory_growth_mb': round(memory_growth, 2),
            'stress_operations': i + 1
        }
    
    def _stress_test_burst(self) -> Dict[str, Any]:
        """Stress test with burst requests."""
        burst_size = 20
        response_times = []
        errors = 0
        
        # Send burst requests
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=burst_size) as executor:
            futures = []
            for _ in range(burst_size):
                future = executor.submit(self._single_prediction_timed)
                futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    response_times.append(result['response_time_ms'])
                else:
                    errors += 1
        
        total_time = time.time() - start_time
        
        if response_times:
            avg_response = statistics.mean(response_times)
            max_response = max(response_times)
        else:
            avg_response = float('inf')
            max_response = float('inf')
        
        success_rate = (len(response_times) / burst_size) * 100
        passed = success_rate >= 70 and avg_response <= self.thresholds['prediction_latency_ms'] * 3
        
        return {
            'passed': passed,
            'burst_size': burst_size,
            'successful_requests': len(response_times),
            'errors': errors,
            'success_rate_percent': round(success_rate, 2),
            'average_response_time_ms': round(avg_response, 2),
            'max_response_time_ms': round(max_response, 2),
            'total_burst_time_seconds': round(total_time, 2)
        }
    
    def _single_prediction_timed(self) -> Dict[str, Any]:
        """Make a single timed prediction."""
        start_time = time.time()
        try:
            predictions = self.live_system.get_todays_predictions()
            response_time_ms = (time.time() - start_time) * 1000
            return {
                'success': True,
                'response_time_ms': response_time_ms
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': (time.time() - start_time) * 1000
            }
    
    def _generate_performance_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result.get('passed', False))
        
        overall_status = "PASS" if passed_tests == total_tests else "FAIL"
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Extract key performance metrics
        key_metrics = {}
        
        if 'prediction_latency' in test_results:
            key_metrics['avg_prediction_latency_ms'] = test_results['prediction_latency']['metrics'].get('average_latency_ms')
        
        if 'batch_throughput' in test_results:
            key_metrics['batch_throughput_per_minute'] = test_results['batch_throughput']['metrics'].get('throughput_per_minute')
        
        if 'memory_usage' in test_results:
            key_metrics['max_memory_usage_mb'] = test_results['memory_usage']['metrics'].get('max_memory_mb')
        
        # Performance grade
        if pass_rate >= 95:
            grade = "A"
        elif pass_rate >= 85:
            grade = "B"
        elif pass_rate >= 75:
            grade = "C"
        else:
            grade = "F"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'performance_grade': grade,
            'pass_rate_percent': round(pass_rate, 2),
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'key_metrics': key_metrics,
            'thresholds': self.thresholds,
            'detailed_results': test_results,
            'recommendations': self._generate_performance_recommendations(test_results)
        }
    
    def _generate_performance_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Check prediction latency
        if 'prediction_latency' in test_results:
            latency = test_results['prediction_latency']['metrics'].get('average_latency_ms', 0)
            if latency > self.thresholds['prediction_latency_ms'] * 0.8:
                recommendations.append("Consider optimizing model inference for faster predictions")
        
        # Check memory usage
        if 'memory_usage' in test_results:
            memory = test_results['memory_usage']['metrics'].get('max_memory_mb', 0)
            if memory > self.thresholds['memory_usage_mb'] * 0.8:
                recommendations.append("Monitor memory usage - approaching threshold")
        
        # Check concurrent performance
        if 'concurrent_predictions' in test_results:
            success_rate = test_results['concurrent_predictions']['metrics'].get('success_rate_percent', 100)
            if success_rate < 95:
                recommendations.append("Improve concurrent request handling and error recovery")
        
        # Check API performance
        if 'api_performance' in test_results:
            api_success = test_results['api_performance']['metrics'].get('api_success_rate_percent', 100)
            if api_success < 90:
                recommendations.append("Implement better API error handling and retry logic")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable thresholds")
        
        return recommendations
    
    def _export_performance_results(self, report: Dict[str, Any]):
        """Export performance test results."""
        output_dir = Path("logs/performance")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"performance_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also create a summary file
        summary_file = output_dir / f"performance_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Performance Test Report - {report['timestamp']}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Status: {report['overall_status']}\n")
            f.write(f"Performance Grade: {report['performance_grade']}\n")
            f.write(f"Pass Rate: {report['pass_rate_percent']}%\n")
            f.write(f"Tests Passed: {report['tests_passed']}/{report['total_tests']}\n\n")
            
            f.write("Key Metrics:\n")
            for metric, value in report['key_metrics'].items():
                f.write(f"  - {metric}: {value}\n")
            
            f.write("\nRecommendations:\n")
            for rec in report['recommendations']:
                f.write(f"  - {rec}\n")
        
        print(f"📊 Performance results exported:")
        print(f"  📄 Full Report: {report_file}")
        print(f"  📋 Summary: {summary_file}")

def main():
    """Run performance testing suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Testing Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick performance tests only')
    parser.add_argument('--test', choices=['latency', 'throughput', 'concurrent', 'memory', 'api', 'load', 'stress'], 
                       help='Run specific test category')
    parser.add_argument('--export-only', action='store_true', help='Export existing results only')
    
    args = parser.parse_args()
    
    perf_suite = PerformanceTestSuite()
    
    if args.export_only:
        print("📊 Exporting previous results...")
        # This would export cached results if available
        return
    
    if args.test:
        print(f"🎯 Running specific test: {args.test}")
        # Run specific test category
        test_methods = {
            'latency': perf_suite._test_prediction_latency,
            'throughput': perf_suite._test_batch_throughput,
            'concurrent': perf_suite._test_concurrent_predictions,
            'memory': perf_suite._test_memory_usage,
            'api': perf_suite._test_api_performance,
            'load': perf_suite._test_load_scenarios,
            'stress': perf_suite._test_stress_scenarios
        }
        
        if args.test in test_methods:
            result = test_methods[args.test]()
            print(f"Result: {result}")
        return
    
    # Run full test suite
    try:
        report = perf_suite.run_all_performance_tests()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏁 PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {report['overall_status']}")
        print(f"Performance Grade: {report['performance_grade']}")
        print(f"Pass Rate: {report['pass_rate_percent']}%")
        print(f"Tests Passed: {report['tests_passed']}/{report['total_tests']}")
        
        if report['key_metrics']:
            print(f"\nKey Metrics:")
            for metric, value in report['key_metrics'].items():
                print(f"  📊 {metric}: {value}")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  💡 {rec}")
        
        print("\n" + "=" * 60)
        
        if report['overall_status'] == 'FAIL':
            print("❌ Some performance tests failed - review detailed results")
            exit(1)
        else:
            print("✅ All performance tests passed!")
            
    except KeyboardInterrupt:
        print("\n🛑 Performance testing interrupted")
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()