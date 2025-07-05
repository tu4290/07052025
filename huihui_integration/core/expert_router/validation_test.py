"""
🎯 EXPERT ROUTER - VALIDATION TEST SUITE
==================================================================

This module provides comprehensive validation and integration testing
for the ExpertRouter system.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Import all components for testing
from . import (
    ExpertRouter,
    RouterConfig,
    RouterMetrics,
    HuiHuiExpertType,
    RoutingDecision,
    VectorBasedRouting,
    PerformanceBasedRouting,
    AdaptiveRouting,
    create_embedding_cache,
    validate_config,
    Timer
)

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    success: bool
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ExpertRouterValidator:
    """Comprehensive validation suite for ExpertRouter."""
    
    def __init__(self, config: Optional[RouterConfig] = None):
        """Initialize the validator with optional custom config."""
        self.config = config or RouterConfig(
            enable_metrics=True,
            enable_adaptive_routing=True,
            enable_vector_routing=True,
            enable_enhanced_cache=False  # Disable for testing
        )
        self.results: List[ValidationResult] = []
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info("Starting comprehensive ExpertRouter validation...")
        
        test_methods = [
            self.test_basic_initialization,
            self.test_config_validation,
            self.test_cache_functionality,
            self.test_metrics_collection,
            self.test_routing_strategies,
            self.test_expert_registration,
            self.test_routing_decisions,
            self.test_error_handling,
            self.test_performance_metrics,
            self.test_concurrent_requests
        ]
        
        for test_method in test_methods:
            try:
                with Timer(test_method.__name__) as timer:
                    await test_method()
                    
                self.results.append(ValidationResult(
                    test_name=test_method.__name__,
                    success=True,
                    duration=timer.duration
                ))
                logger.info(f"✅ {test_method.__name__} passed ({timer.duration:.3f}s)")
                
            except Exception as e:
                self.results.append(ValidationResult(
                    test_name=test_method.__name__,
                    success=False,
                    duration=timer.duration if 'timer' in locals() else 0.0,
                    error=str(e)
                ))
                logger.error(f"❌ {test_method.__name__} failed: {e}")
        
        return self.generate_summary()
    
    async def test_basic_initialization(self):
        """Test basic router initialization."""
        router = ExpertRouter(config=self.config)
        await router.initialize()
        
        assert router._initialized, "Router should be initialized"
        assert router.strategies, "Router should have strategies"
        assert router.active_strategy is not None, "Router should have an active strategy"
        
        await router.shutdown()
    
    async def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        valid_config = {
            'ollama_host': 'http://localhost:11434',
            'enable_metrics': True,
            'cache_max_size': 1000
        }
        validate_config(valid_config, ['ollama_host'])
        
        # Test invalid config (should raise)
        try:
            validate_config({}, ['required_key'])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
    
    async def test_cache_functionality(self):
        """Test cache operations."""
        cache = create_embedding_cache(use_enhanced=False, max_size=100)
        
        # Test basic operations
        test_text = "test embedding text"
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Should not exist initially
        assert cache.get(test_text) is None
        
        # Set and retrieve
        cache.set(test_text, test_embedding)
        retrieved = cache.get(test_text)
        assert retrieved is not None
        assert len(retrieved) == len(test_embedding)
        
        # Test stats
        stats = cache.get_stats()
        assert 'total_entries' in stats
        assert stats['total_entries'] > 0
        
        # Test clear
        cache.clear()
        assert cache.get(test_text) is None
    
    async def test_metrics_collection(self):
        """Test metrics collection functionality."""
        metrics = RouterMetrics(
            enable_prometheus=False,  # Disable for testing
            enable_metrics=True
        )
        
        # Test basic metric recording
        metrics.record_router_request("test_strategy", 0.5)
        metrics.record_expert_request("VOLATILITY", "test_strategy", 0.3, True)
        metrics.record_cache_hit("test_cache")
        metrics.record_cache_miss("test_cache")
        
        # Test metric retrieval
        summary = metrics.get_metric_summary()
        assert 'router_requests' in summary
        assert 'expert_requests' in summary
        assert 'cache_operations' in summary
        
        await metrics.shutdown()
    
    async def test_routing_strategies(self):
        """Test individual routing strategies."""
        cache = create_embedding_cache(use_enhanced=False)
        
        # Test VectorBasedRouting
        vector_strategy = VectorBasedRouting(
            embedding_cache=cache,
            ollama_host=self.config.ollama_host
        )
        
        # Test basic strategy properties
        assert hasattr(vector_strategy, 'strategy_name')
        assert hasattr(vector_strategy, 'select_expert')
        
        # Test PerformanceBasedRouting
        perf_strategy = PerformanceBasedRouting()
        assert hasattr(perf_strategy, 'strategy_name')
        assert hasattr(perf_strategy, 'select_expert')
        
        # Test AdaptiveRouting
        adaptive_strategy = AdaptiveRouting(
            strategies=[
                {"type": "vector", "weight": 0.7},
                {"type": "performance", "weight": 0.3}
            ],
            fallback_expert=HuiHuiExpertType.VOLATILITY
        )
        assert hasattr(adaptive_strategy, 'strategy_name')
        assert hasattr(adaptive_strategy, 'select_expert')
    
    async def test_expert_registration(self):
        """Test expert handler registration."""
        router = ExpertRouter(config=self.config)
        await router.initialize()
        
        # Define test handler
        async def test_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
            return {"analysis": f"Test analysis for: {prompt}"}
        
        # Register handler
        router.register_expert_handler(HuiHuiExpertType.VOLATILITY, test_handler)
        
        # Verify registration
        assert HuiHuiExpertType.VOLATILITY in router.expert_handlers
        
        await router.shutdown()
    
    async def test_routing_decisions(self):
        """Test routing decision making."""
        router = ExpertRouter(config=self.config)
        await router.initialize()
        
        # Define test handlers
        async def volatility_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
            return {"analysis": "Volatility analysis", "expert": "volatility"}
        
        async def liquidity_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
            return {"analysis": "Liquidity analysis", "expert": "liquidity"}
        
        # Register handlers
        router.register_expert_handler(HuiHuiExpertType.VOLATILITY, volatility_handler)
        router.register_expert_handler(HuiHuiExpertType.LIQUIDITY, liquidity_handler)
        
        # Test expert selection
        test_prompt = "What's the current market volatility?"
        decision = await router.select_expert(test_prompt)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.selected_expert in [HuiHuiExpertType.VOLATILITY, HuiHuiExpertType.LIQUIDITY]
        assert 0.0 <= decision.confidence <= 1.0
        
        # Test routing
        result = await router.route(test_prompt)
        assert isinstance(result, dict)
        assert "analysis" in result
        
        await router.shutdown()
    
    async def test_error_handling(self):
        """Test error handling and recovery."""
        router = ExpertRouter(config=self.config)
        await router.initialize()
        
        # Define failing handler
        async def failing_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
            raise Exception("Simulated expert failure")
        
        # Register failing handler
        router.register_expert_handler(HuiHuiExpertType.RISK, failing_handler)
        
        # Test that routing handles failures gracefully
        try:
            result = await router.route("Test prompt", timeout=5.0)
            # Should either succeed with fallback or handle error gracefully
            assert isinstance(result, dict)
        except Exception as e:
            # Error should be properly logged and handled
            assert "error" in str(e).lower() or "timeout" in str(e).lower()
        
        await router.shutdown()
    
    async def test_performance_metrics(self):
        """Test performance metric tracking."""
        router = ExpertRouter(config=self.config)
        await router.initialize()
        
        # Define test handler with timing
        async def timed_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"analysis": "Timed analysis", "duration": 0.1}
        
        router.register_expert_handler(HuiHuiExpertType.EXECUTION, timed_handler)
        
        # Route multiple requests to gather metrics
        for i in range(5):
            await router.route(f"Test prompt {i}")
        
        # Check status includes performance metrics
        status = router.get_status()
        assert "performance_metrics" in status
        assert "total_requests" in status
        
        await router.shutdown()
    
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        router = ExpertRouter(config=self.config)
        await router.initialize()
        
        # Define concurrent handler
        async def concurrent_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0.05)  # Small delay
            return {"analysis": f"Concurrent analysis for: {prompt}"}
        
        router.register_expert_handler(HuiHuiExpertType.VOLATILITY, concurrent_handler)
        
        # Create multiple concurrent requests
        tasks = [
            router.route(f"Concurrent test {i}")
            for i in range(10)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed successfully
        successful_results = [r for r in results if isinstance(r, dict)]
        assert len(successful_results) >= 8, f"Expected at least 8 successful results, got {len(successful_results)}"
        
        await router.shutdown()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all test results."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        total_duration = sum(r.duration for r in self.results)
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_duration": total_duration,
            "average_duration": total_duration / total_tests if total_tests > 0 else 0,
            "results": [
                {
                    "test": r.test_name,
                    "success": r.success,
                    "duration": r.duration,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        return summary

async def run_validation() -> Dict[str, Any]:
    """Run the complete validation suite."""
    validator = ExpertRouterValidator()
    return await validator.run_all_tests()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run validation
    async def main():
        print("🎯 Starting ExpertRouter Validation Suite...")
        results = await run_validation()
        
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']} ✅")
        print(f"Failed: {results['failed']} ❌")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Total Duration: {results['total_duration']:.3f}s")
        print(f"Average Duration: {results['average_duration']:.3f}s")
        
        if results['failed'] > 0:
            print("\n❌ FAILED TESTS:")
            for result in results['results']:
                if not result['success']:
                    print(f"  - {result['test']}: {result['error']}")
        
        print("\n🎯 Validation Complete!")
        
        return results['success_rate'] == 100.0
    
    # Run the validation
    success = asyncio.run(main())
    exit(0 if success else 1) 