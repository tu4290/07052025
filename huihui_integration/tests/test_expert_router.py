"""
Test suite for the Expert Router system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from ..core.expert_router import (
    ExpertRouter,
    RouterConfig,
    HuiHuiExpertType,
    RoutingDecision,
    RouterMetrics,
    AdaptiveLearningManager
)

# Test data
TEST_PROMPT = "Analyze the current market regime"
TEST_CONTEXT = {"symbol": "SPY", "timeframe": "1h"}

# Fixtures
@pytest.fixture
def router_config():
    """Create a test router configuration."""
    return RouterConfig(
        ollama_host="http://localhost:11434",
        enable_metrics=False,
        enable_adaptive_learning=False
    )

@pytest.fixture
def mock_metrics():
    """Create a mock metrics collector."""
    metrics = MagicMock(spec=RouterMetrics)
    metrics.record_router_request = AsyncMock()
    metrics.record_router_error = AsyncMock()
    metrics.record_expert_request = AsyncMock()
    metrics.record_expert_error = AsyncMock()
    return metrics

@pytest.fixture
def mock_learning_manager():
    """Create a mock adaptive learning manager."""
    manager = MagicMock(spec=AdaptiveLearningManager)
    manager.initialize = AsyncMock()
    manager.update = AsyncMock()
    manager.get_learning_state = AsyncMock(return_value={"confidence": 0.9})
    return manager

@pytest.fixture
async def test_router(router_config, mock_metrics, mock_learning_manager):
    """Create a test router instance with mocks."""
    router = ExpertRouter(
        config=router_config,
        logger=MagicMock(),
        adaptive_learning_manager=mock_learning_manager
    )
    router.metrics = mock_metrics
    
    # Mock the routing strategies
    router.strategies = {
        "vector": MagicMock(spec=BaseRoutingStrategy),
        "performance": MagicMock(spec=BaseRoutingStrategy)
    }
    router.active_strategy = router.strategies["vector"]
    
    # Mock the expert handlers
    async def mock_handler(prompt, context):
        return {"response": f"Response to: {prompt}"}
        
    for expert in HuiHuiExpertType:
        router.register_expert_handler(expert, mock_handler)
    
    await router.initialize()
    return router

# Tests
@pytest.mark.asyncio
async def test_router_initialization(test_router):
    """Test that the router initializes correctly."""
    assert test_router is not None
    assert test_router._initialized is True
    assert test_router.active_strategy is not None

@pytest.mark.asyncio
async def test_route_prompt(test_router):
    """Test routing a prompt to an expert."""
    # Setup
    decision = RoutingDecision(
        expert_type=HuiHuiExpertType.MARKET_REGIME,
        confidence=0.9,
        strategy_used="test",
        metadata={"test": "data"}
    )
    test_router.active_strategy.select_expert = AsyncMock(return_value=decision)
    
    # Execute
    response = await test_router.route(TEST_PROMPT, TEST_CONTEXT)
    
    # Verify
    assert response is not None
    assert "response" in response
    test_router.active_strategy.select_expert.assert_called_once()
    test_router.metrics.record_router_request.assert_called_once()
    test_router.metrics.record_expert_request.assert_called_once()

@pytest.mark.asyncio
async def test_route_with_error(test_router):
    """Test error handling during routing."""
    # Setup
    test_router.active_strategy.select_expert = AsyncMock(side_effect=Exception("Test error"))
    
    # Execute & Verify
    with pytest.raises(Exception):
        await test_router.route(TEST_PROMPT, TEST_CONTEXT)
    
    test_router.metrics.record_router_error.assert_called_once()

@pytest.mark.asyncio
async def test_performance_tracking(test_router):
    """Test performance metrics tracking."""
    # Setup
    decision = RoutingDecision(
        expert_type=HuiHuiExpertType.OPTIONS_FLOW,
        confidence=0.85,
        strategy_used="test"
    )
    
    # Execute
    start_time = datetime.now()
    await test_router._record_metrics(
        expert_type=HuiHuiExpertType.OPTIONS_FLOW,
        success=True,
        duration=0.5,
        routing_decision=decision
    )
    
    # Verify
    test_router.metrics.record_expert_request.assert_called_once()
    assert test_router.performance_metrics[HuiHuiExpertType.OPTIONS_FLOW].successful_queries > 0

@pytest.mark.asyncio
async def test_adaptive_learning_integration(test_router, mock_learning_manager):
    """Test integration with adaptive learning."""
    # Setup
    test_router.config.enable_adaptive_learning = True
    
    # Execute
    await test_router.initialize()
    
    # Verify
    mock_learning_manager.initialize.assert_called_once()

@pytest.mark.asyncio
async def test_cache_operations(test_router):
    """Test cache operations."""
    # Setup
    test_key = "test_key"
    test_value = [0.1, 0.2, 0.3]
    
    # Test set and get
    await test_router.cache.set_async(test_key, test_value)
    cached = await test_router.cache.get_async(test_key)
    
    # Verify
    assert cached is not None
    assert len(cached) == len(test_value)
    
    # Test cache miss
    assert await test_router.cache.get_async("nonexistent") is None

if __name__ == "__main__":
    pytest.main(["-v", "test_expert_router.py"])
