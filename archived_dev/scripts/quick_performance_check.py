#!/usr/bin/env python3
"""
Quick Performance Check
=======================

Fast performance validation for production readiness.
"""

import os
import sys
import time
import json
import psutil
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from modeling import EnhancedDualModelSystem
from live_prediction_system import LivePredictionSystem

def quick_performance_check() -> dict:
    """Run quick performance validation."""
    print("⚡ Running quick performance check...")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {},
        'overall_status': 'unknown'
    }
    
    # Test 1: System initialization time
    print("🔧 Testing system initialization...")
    start_time = time.time()
    
    try:
        model_system = EnhancedDualModelSystem()
        live_system = LivePredictionSystem()
        
        init_time = time.time() - start_time
        results['tests']['initialization'] = {
            'passed': init_time < 10.0,  # 10 seconds max
            'time_seconds': round(init_time, 2),
            'threshold_seconds': 10.0
        }
        print(f"✅ Initialization: {init_time:.2f}s")
        
    except Exception as e:
        results['tests']['initialization'] = {
            'passed': False,
            'error': str(e)
        }
        print(f"❌ Initialization failed: {e}")
        return results
    
    # Test 2: Memory usage
    print("💾 Testing memory usage...")
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    results['tests']['memory_usage'] = {
        'passed': memory_mb < 1000,  # 1GB max
        'memory_mb': round(memory_mb, 2),
        'threshold_mb': 1000
    }
    print(f"✅ Memory: {memory_mb:.2f} MB")
    
    # Test 3: Basic prediction speed
    print("🔮 Testing prediction performance...")
    prediction_times = []
    
    for i in range(3):  # Quick test with 3 predictions
        start_time = time.time()
        try:
            predictions = live_system.get_todays_predictions()
            prediction_time = (time.time() - start_time) * 1000  # ms
            prediction_times.append(prediction_time)
            print(f"  Prediction {i+1}: {prediction_time:.1f}ms")
        except Exception as e:
            print(f"  ⚠️  Prediction {i+1} failed: {e}")
            prediction_times.append(5000)  # Penalty time
    
    avg_prediction_time = sum(prediction_times) / len(prediction_times)
    results['tests']['prediction_speed'] = {
        'passed': avg_prediction_time < 3000,  # 3 seconds max
        'average_time_ms': round(avg_prediction_time, 2),
        'threshold_ms': 3000,
        'individual_times': [round(t, 2) for t in prediction_times]
    }
    
    # Test 4: CPU usage during operation
    print("🔥 Testing CPU usage...")
    cpu_percent = psutil.cpu_percent(interval=1)
    results['tests']['cpu_usage'] = {
        'passed': cpu_percent < 80,  # 80% max
        'cpu_percent': cpu_percent,
        'threshold_percent': 80
    }
    print(f"✅ CPU: {cpu_percent}%")
    
    # Overall assessment
    passed_tests = sum(1 for test in results['tests'].values() if test.get('passed', False))
    total_tests = len(results['tests'])
    
    if passed_tests == total_tests:
        results['overall_status'] = 'excellent'
    elif passed_tests >= total_tests * 0.75:
        results['overall_status'] = 'good'
    elif passed_tests >= total_tests * 0.5:
        results['overall_status'] = 'acceptable'
    else:
        results['overall_status'] = 'needs_improvement'
    
    results['pass_rate'] = (passed_tests / total_tests) * 100
    results['tests_passed'] = passed_tests
    results['total_tests'] = total_tests
    
    return results

def print_performance_summary(results: dict):
    """Print performance summary."""
    print("\n" + "="*50)
    print("⚡ QUICK PERFORMANCE CHECK SUMMARY")
    print("="*50)
    
    status_emoji = {
        'excellent': '🟢',
        'good': '🟡',
        'acceptable': '🟠',
        'needs_improvement': '🔴'
    }
    
    emoji = status_emoji.get(results['overall_status'], '❓')
    print(f"Overall Status: {emoji} {results['overall_status'].upper()}")
    print(f"Tests Passed: {results['tests_passed']}/{results['total_tests']} ({results['pass_rate']:.1f}%)")
    
    print("\nTest Results:")
    for test_name, test_data in results['tests'].items():
        status = "✅ PASS" if test_data.get('passed', False) else "❌ FAIL"
        print(f"  {status} {test_name}")
        
        # Show key metrics
        if 'time_seconds' in test_data:
            print(f"      Time: {test_data['time_seconds']}s (threshold: {test_data['threshold_seconds']}s)")
        elif 'memory_mb' in test_data:
            print(f"      Memory: {test_data['memory_mb']} MB (threshold: {test_data['threshold_mb']} MB)")
        elif 'average_time_ms' in test_data:
            print(f"      Avg Speed: {test_data['average_time_ms']}ms (threshold: {test_data['threshold_ms']}ms)")
        elif 'cpu_percent' in test_data:
            print(f"      CPU: {test_data['cpu_percent']}% (threshold: {test_data['threshold_percent']}%)")
        
        if 'error' in test_data:
            print(f"      Error: {test_data['error']}")
    
    print("\nRecommendations:")
    if results['overall_status'] == 'excellent':
        print("  🎯 System performance is excellent - ready for production!")
    elif results['overall_status'] == 'good':
        print("  👍 System performance is good - minor optimizations recommended")
    elif results['overall_status'] == 'acceptable':
        print("  ⚠️  System performance is acceptable - optimization needed")
    else:
        print("  🚨 System performance needs significant improvement")
    
    print("="*50)

def main():
    """Run quick performance check."""
    try:
        results = quick_performance_check()
        print_performance_summary(results)
        
        # Export results
        output_file = f"logs/quick_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("logs", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📊 Results exported to: {output_file}")
        
        # Exit code based on results
        if results['overall_status'] in ['excellent', 'good']:
            exit(0)
        elif results['overall_status'] == 'acceptable':
            exit(1)
        else:
            exit(2)
            
    except Exception as e:
        print(f"❌ Performance check failed: {e}")
        exit(3)

if __name__ == "__main__":
    main()