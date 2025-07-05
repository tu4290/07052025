# 🎯 Expert Router

A modular, extensible, and high-performance routing system for intelligently directing queries to specialized AI experts.

## Features

- **Modular Architecture**: Easily extensible with custom routing strategies
- **Multiple Routing Strategies**:
  - Vector-based semantic routing using embeddings
  - Performance-based routing using historical metrics
  - Adaptive routing that combines multiple strategies
- **Built-in Caching**: Ultra-fast embedding and response caching
- **Metrics and Monitoring**: Comprehensive metrics collection and Prometheus integration
- **Async-First**: Built with asyncio for high concurrency
- **Type Annotations**: Full type hints for better developer experience
- **Backward Compatibility**: Legacy API support for smooth migration

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from huihui_integration.core.expert_router import (
    ExpertRouter,
    HuiHuiExpertType,
    RouterConfig
)

async def main():
    # Create a router with default configuration
    router = ExpertRouter(
        config=RouterConfig(
            ollama_host="http://localhost:11434",
            enable_metrics=True,
            metrics_port=8000
        )
    )
    
    try:
        # Initialize the router
        await router.initialize()
        
        # Register expert handlers
        @router.register_expert_handler(HuiHuiExpertType.MARKET_REGIME)
        async def handle_market_regime(prompt: str, context: dict) -> dict:
            return {"analysis": f"Market regime analysis for: {prompt}"}
        
        # Route a prompt
        response = await router.route("What's the current market regime?")
        print(f"Response: {response}")
        
    finally:
        # Clean up
        await router.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

### Core Components

- **ExpertRouter**: Main class that coordinates between different routing strategies
- **Routing Strategies**:
  - `VectorBasedRouting`: Routes based on semantic similarity using embeddings
  - `PerformanceBasedRouting`: Routes based on historical performance metrics
  - `AdaptiveRouting`: Combines multiple strategies with weighted voting
- **Cache Layer**: Ultra-fast in-memory caching for embeddings and responses
- **Metrics**: Comprehensive metrics collection and monitoring

### Data Models

- `HuiHuiExpertType`: Enum of available expert types
- `PerformanceMetrics`: Tracks performance metrics for experts
- `ExpertPerformance`: Performance data for a specific expert
- `RoutingDecision`: Result of expert selection

## Configuration

```python
from huihui_integration.core.expert_router import RouterConfig

config = RouterConfig(
    ollama_host="http://localhost:11434",  # Ollama server URL
    default_expert=HuiHuiExpertType.ORCHESTRATOR,
    enable_metrics=True,
    metrics_port=8000,
    metrics_namespace="expert_router",
    cache_max_size=10000,
    cache_ttl_seconds=86400,  # 24 hours
    default_timeout=30.0,  # seconds
    enable_adaptive_routing=True,
    enable_performance_tracking=True,
    enable_vector_routing=True
)
```

## Metrics

The router exposes Prometheus metrics on the configured port (default: 8000):

- `expert_router_router_requests_total`: Total routing requests
- `expert_router_router_request_duration_seconds`: Request duration histogram
- `expert_router_router_errors_total`: Routing errors
- `expert_router_expert_requests_total`: Expert requests
- `expert_router_expert_request_duration_seconds`: Expert request duration
- `expert_router_expert_errors_total`: Expert errors
- `expert_router_cache_hits_total`: Cache hits
- `expert_router_cache_misses_total`: Cache misses
- `expert_router_cache_size_bytes`: Cache size in bytes
- `expert_router_active_strategies`: Number of active routing strategies

## Extending the Router

### Creating a Custom Routing Strategy

```python
from typing import Dict, Any, Optional
from huihui_integration.core.expert_router.strategies import BaseRoutingStrategy
from huihui_integration.core.expert_router.models import (
    HuiHuiExpertType,
    RoutingDecision
)

class CustomRoutingStrategy(BaseRoutingStrategy):
    """Custom routing strategy example."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strategy_name = "custom"
    
    async def select_expert(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """Select an expert based on custom logic."""
        # Your custom routing logic here
        return RoutingDecision(
            expert_type=HuiHuiExpertType.ORCHESTRATOR,
            confidence=1.0,
            strategy_used=self.strategy_name
        )
    
    async def update_performance(
        self,
        expert_type: HuiHuiExpertType,
        success: bool,
        response_time: float
    ) -> None:
        """Update performance metrics for an expert."""
        # Update your custom metrics here
        pass
```

### Registering a Custom Strategy

```python
from huihui_integration.core.expert_router import ExpertRouter

async def main():
    router = ExpertRouter()
    
    # Create and register custom strategy
    custom_strategy = CustomRoutingStrategy()
    router.strategies["custom"] = custom_strategy
    router.active_strategy = custom_strategy
    
    # Use the router as usual
    # ...
```

## Backward Compatibility

The module provides backward compatibility aliases:

- `HuiHuiRouter` is an alias for `ExpertRouter`
- `AIRouter` is also an alias for `ExpertRouter`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
