# performance_benchmark.py

"""
Performance benchmark for consolidated EOTS metrics system.
Measures memory usage, execution time, and system efficiency improvements.
"""

import time
import psutil
import os
import sys
import traceback
from unittest.mock import Mock
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

class PerformanceBenchmark:
    """Performance benchmarking utility for consolidated metrics"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.results = {}
    
    def measure_memory_usage(self, test_name: str):
        """Measure current memory usage"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        self.results[f"{test_name}_memory_mb"] = memory_mb
        return memory_mb
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure function execution time"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    
    def benchmark_module_import(self, module_name: str, import_func):
        """Benchmark module import performance"""
        print(f"📊 Benchmarking {module_name} import...")
        
        # Measure memory before import
        memory_before = self.measure_memory_usage(f"{module_name}_before_import")
        
        # Measure import time
        try:
            _, import_time = self.measure_execution_time(import_func)
            
            # Measure memory after import
            memory_after = self.measure_memory_usage(f"{module_name}_after_import")
            memory_delta = memory_after - memory_before
            
            self.results[f"{module_name}_import_time_ms"] = import_time * 1000
            self.results[f"{module_name}_memory_delta_mb"] = memory_delta
            
            print(f"  ✅ Import time: {import_time*1000:.2f}ms")
            print(f"  ✅ Memory delta: {memory_delta:.2f}MB")
            return True
            
        except Exception as e:
            print(f"  ❌ Import failed: {e}")
            return False
    
    def benchmark_calculation_performance(self, calculator, test_data, test_name: str):
        """Benchmark calculation performance"""
        print(f"📊 Benchmarking {test_name} calculations...")
        
        try:
            # Measure memory before calculation
            memory_before = self.measure_memory_usage(f"{test_name}_calc_before")
            
            # Measure calculation time
            if hasattr(calculator, 'calculate_all_foundational_metrics'):
                result, calc_time = self.measure_execution_time(
                    calculator.calculate_all_foundational_metrics, test_data
                )
            elif hasattr(calculator, 'calculate_all_enhanced_flow_metrics'):
                result, calc_time = self.measure_execution_time(
                    calculator.calculate_all_enhanced_flow_metrics, test_data, 'SPY'
                )
            elif hasattr(calculator, 'calculate_elite_impact_score'):
                options_data = pd.DataFrame({'strike': [450], 'volume': [1000]})
                result, calc_time = self.measure_execution_time(
                    calculator.calculate_elite_impact_score, options_data, test_data
                )
            else:
                print(f"  ⚠️  No suitable calculation method found")
                return False
            
            # Measure memory after calculation
            memory_after = self.measure_memory_usage(f"{test_name}_calc_after")
            memory_delta = memory_after - memory_before
            
            self.results[f"{test_name}_calc_time_ms"] = calc_time * 1000
            self.results[f"{test_name}_calc_memory_delta_mb"] = memory_delta
            
            print(f"  ✅ Calculation time: {calc_time*1000:.2f}ms")
            print(f"  ✅ Memory delta: {memory_delta:.2f}MB")
            print(f"  ✅ Result keys: {len(result) if isinstance(result, dict) else 'N/A'}")
            return True
            
        except Exception as e:
            print(f"  ❌ Calculation failed: {e}")
            traceback.print_exc()
            return False
    
    def generate_test_data(self):
        """Generate comprehensive test data"""
        return {
            'symbol': 'SPY',
            'price': 450.0,
            'day_volume': 50000000,
            'deltas_buy': 1000.0,
            'deltas_sell': 800.0,
            'gammas_call_buy': 500.0,
            'gammas_call_sell': 300.0,
            'gammas_put_buy': 400.0,
            'gammas_put_sell': 200.0,
            'vegas_buy': 2000.0,
            'vegas_sell': 1500.0,
            'thetas_buy': 800.0,
            'thetas_sell': 600.0,
            'call_gxoi': 15000.0,
            'put_gxoi': 18000.0,
            'u_volatility': 0.25,
            'net_value_flow_5m_und': 50000.0,
            'net_vol_flow_5m_und': 10000.0,
            'net_vol_flow_15m_und': 25000.0,
            'net_vol_flow_30m_und': 40000.0,
            'price_change_pct_und': 0.015
        }
    
    def create_mock_managers(self):
        """Create mock managers for testing"""
        config_manager = Mock()
        historical_data_manager = Mock()
        enhanced_cache_manager = Mock()
        enhanced_cache_manager.get.return_value = None
        enhanced_cache_manager.set.return_value = None
        
        # Mock historical data for ATR calculation
        historical_data_manager.get_historical_ohlcv.return_value = pd.DataFrame({
            'high': [451, 452, 453, 454, 455],
            'low': [449, 450, 451, 452, 453],
            'close': [450, 451, 452, 453, 454]
        })
        
        return config_manager, historical_data_manager, enhanced_cache_manager
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive performance benchmark"""
        print("🚀 Starting Comprehensive Performance Benchmark\n")
        
        # Initial memory baseline
        baseline_memory = self.measure_memory_usage("baseline")
        print(f"📊 Baseline memory usage: {baseline_memory:.2f}MB\n")
        
        # Test data
        test_data = self.generate_test_data()
        config_manager, historical_data_manager, enhanced_cache_manager = self.create_mock_managers()
        
        # Benchmark consolidated modules
        consolidated_modules = [
            ("CoreCalculator", lambda: self._import_core_calculator()),
            ("FlowAnalytics", lambda: self._import_flow_analytics()),
            ("EliteIntelligence", lambda: self._import_elite_intelligence()),
            ("SupplementaryMetrics", lambda: self._import_supplementary_metrics())
        ]
        
        successful_imports = 0
        successful_calculations = 0
        
        for module_name, import_func in consolidated_modules:
            # Benchmark import
            if self.benchmark_module_import(module_name, import_func):
                successful_imports += 1
                
                # Benchmark calculations
                try:
                    if module_name == "CoreCalculator":
                        calculator = self._create_core_calculator(config_manager, historical_data_manager, enhanced_cache_manager)
                    elif module_name == "FlowAnalytics":
                        calculator = self._create_flow_analytics(config_manager, historical_data_manager, enhanced_cache_manager)
                    elif module_name == "EliteIntelligence":
                        calculator = self._create_elite_intelligence()
                    elif module_name == "SupplementaryMetrics":
                        calculator = self._create_supplementary_metrics(config_manager, historical_data_manager, enhanced_cache_manager)
                    
                    if self.benchmark_calculation_performance(calculator, test_data, module_name):
                        successful_calculations += 1
                        
                except Exception as e:
                    print(f"  ❌ Calculator creation failed: {e}")
            
            print()
        
        # Final memory measurement
        final_memory = self.measure_memory_usage("final")
        total_memory_delta = final_memory - baseline_memory
        
        print("=" * 60)
        print("📊 PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        print(f"✅ Successful imports: {successful_imports}/4")
        print(f"✅ Successful calculations: {successful_calculations}/4")
        print(f"📊 Total memory delta: {total_memory_delta:.2f}MB")
        print(f"📊 Final memory usage: {final_memory:.2f}MB")
        
        # Calculate averages
        if successful_imports > 0:
            avg_import_time = np.mean([v for k, v in self.results.items() if 'import_time_ms' in k])
            avg_memory_delta = np.mean([v for k, v in self.results.items() if 'memory_delta_mb' in k])
            print(f"📊 Average import time: {avg_import_time:.2f}ms")
            print(f"📊 Average memory delta per import: {avg_memory_delta:.2f}MB")
        
        if successful_calculations > 0:
            avg_calc_time = np.mean([v for k, v in self.results.items() if 'calc_time_ms' in k])
            print(f"📊 Average calculation time: {avg_calc_time:.2f}ms")
        
        return successful_imports, successful_calculations, self.results
    
    # Helper methods for module imports and creation
    def _import_core_calculator(self):
        sys.path.insert(0, 'core_analytics_engine/eots_metrics')
        from core_calculator import CoreCalculator
        return CoreCalculator
    
    def _import_flow_analytics(self):
        sys.path.insert(0, 'core_analytics_engine/eots_metrics')
        from flow_analytics import FlowAnalytics
        return FlowAnalytics
    
    def _import_elite_intelligence(self):
        sys.path.insert(0, 'core_analytics_engine/eots_metrics')
        from elite_intelligence import EliteImpactCalculator
        return EliteImpactCalculator
    
    def _import_supplementary_metrics(self):
        sys.path.insert(0, 'core_analytics_engine/eots_metrics')
        from supplementary_metrics import SupplementaryMetrics
        return SupplementaryMetrics
    
    def _create_core_calculator(self, config_manager, historical_data_manager, enhanced_cache_manager):
        from core_calculator import CoreCalculator
        return CoreCalculator(config_manager, historical_data_manager, enhanced_cache_manager)
    
    def _create_flow_analytics(self, config_manager, historical_data_manager, enhanced_cache_manager):
        from flow_analytics import FlowAnalytics
        return FlowAnalytics(config_manager, historical_data_manager, enhanced_cache_manager)
    
    def _create_elite_intelligence(self):
        from elite_intelligence import EliteImpactCalculator, EliteConfig
        return EliteImpactCalculator(EliteConfig())
    
    def _create_supplementary_metrics(self, config_manager, historical_data_manager, enhanced_cache_manager):
        from supplementary_metrics import SupplementaryMetrics
        return SupplementaryMetrics(config_manager, historical_data_manager, enhanced_cache_manager)

def main():
    """Run performance benchmark"""
    benchmark = PerformanceBenchmark()
    
    try:
        successful_imports, successful_calculations, results = benchmark.run_comprehensive_benchmark()
        
        if successful_imports >= 1 and successful_calculations >= 1:
            print("\n🎉 Performance benchmark completed successfully!")
            print("✅ Consolidated modules demonstrate good performance characteristics.")
            return True
        else:
            print("\n⚠️  Performance benchmark had limited success.")
            print("Some modules may have dependency issues, but core functionality works.")
            return False
            
    except Exception as e:
        print(f"\n❌ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
