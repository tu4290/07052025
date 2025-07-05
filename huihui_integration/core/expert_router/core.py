"""
🎯 EXPERT ROUTER - CORE MODULE
==================================================================

This module implements the main ExpertRouter class that coordinates between
different routing strategies and manages the expert selection process.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from datetime import datetime
import uuid

# Import models and strategies
from data_models.ai_ml_models import MOEGatingNetworkV2_5, MOEExpertRegistryV2_5, HuiHuiExpertType
from data_models.huihui_learning_schemas import ExpertPerformanceHistory as PerformanceMetrics
from .cache import (
    create_embedding_cache
)
from .adaptive_learning import (
    AdaptiveLearningManager,
    AdaptiveLearningConfig
)
from .strategies import (
    BaseRoutingStrategy,
    VectorBasedRouting,
    PerformanceBasedRouting,
    AdaptiveRouting
)
from .metrics import RouterMetrics
from data_models.ai_ml_models import RoutingDecision

# Type aliases
ExpertResponse = Dict[str, Any]
ExpertHandler = Callable[[str, Dict[str, Any]], Awaitable[ExpertResponse]]

logger = logging.getLogger(__name__)

class RouterConfig:
    """Configuration for the ExpertRouter."""
    
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        default_expert: HuiHuiExpertType = HuiHuiExpertType.ORCHESTRATOR,
        enable_metrics: bool = True,
        metrics_port: int = 8000,
        metrics_namespace: str = "expert_router",
        cache_max_size: int = 10000,
        cache_ttl_seconds: int = 86400,  # 24 hours
        default_timeout: float = 30.0,  # seconds
        enable_adaptive_routing: bool = True,
        enable_performance_tracking: bool = True,
        enable_vector_routing: bool = True,
        enable_enhanced_cache: bool = True,
        enable_adaptive_learning: bool = True,
        adaptive_learning_config: Optional[Dict[str, Any]] = None,
        adaptive_routing_strategy_config: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        self.ollama_host = ollama_host
        self.default_expert = default_expert
        self.enable_metrics = enable_metrics
        self.metrics_port = metrics_port
        self.metrics_namespace = metrics_namespace
        self.cache_max_size = cache_max_size
        self.cache_ttl_seconds = cache_ttl_seconds
        self.default_timeout = default_timeout
        self.enable_adaptive_routing = enable_adaptive_routing
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_vector_routing = enable_vector_routing
        self.enable_enhanced_cache = enable_enhanced_cache
        self.enable_adaptive_learning = enable_adaptive_learning
        self.adaptive_learning_config = adaptive_learning_config or {}
        self.adaptive_routing_strategy_config = adaptive_routing_strategy_config or [
            {"type": "vector", "weight": 0.7},
            {"type": "performance", "weight": 0.3}
        ]
        self.extra_config = kwargs

class ExpertRouter:
    """
    Main ExpertRouter class that coordinates between different routing strategies
    and manages the expert selection process.
    """
    
    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        logger: Optional[logging.Logger] = None,
        adaptive_learning_manager: Optional[AdaptiveLearningManager] = None,
        **kwargs
    ):
        self.config = config or RouterConfig(**kwargs)
        self.logger = logger or logging.getLogger(__name__)
        self._initialized = False
        self._shutdown_flag = False
        self._start_time = time.time()
        self.metrics = RouterMetrics(
            enable_metrics=self.config.enable_metrics,
            port=self.config.metrics_port,
            namespace=self.config.metrics_namespace
        )
        self.cache = create_embedding_cache(
            use_enhanced=self.config.enable_enhanced_cache,
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds
        )
        self.adaptive_learning_manager = adaptive_learning_manager
        if self.config.enable_adaptive_learning and self.adaptive_learning_manager is None:
            self.adaptive_learning_manager = AdaptiveLearningManager(
                config=AdaptiveLearningConfig(**self.config.adaptive_learning_config),
                metrics_callback=self._record_adaptive_learning_metrics
            )
            asyncio.create_task(self.adaptive_learning_manager.initialize())
        self.expert_handlers: Dict[HuiHuiExpertType, ExpertHandler] = {}
        self.strategies: Dict[str, BaseRoutingStrategy] = {}
        self.active_strategy: Optional[BaseRoutingStrategy] = None
        self.performance_metrics: Dict[HuiHuiExpertType, PerformanceMetrics] = {
            expert: PerformanceMetrics() for expert in HuiHuiExpertType
        }
    
    async def initialize(self) -> None:
        if self._initialized:
            return
        self.logger.info("Initializing ExpertRouter...")
        await self._initialize_strategies()
        if self.strategies:
            if self.config.enable_adaptive_routing and "adaptive" in self.strategies:
                self.active_strategy = self.strategies["adaptive"]
            else:
                self.active_strategy = next(iter(self.strategies.values()))
        else:
            self.logger.warning("No routing strategies available")
        self._initialized = True
        self.logger.info(
            f"ExpertRouter initialized with {len(self.strategies)} strategies. "
            f"Active strategy: {getattr(self.active_strategy, 'strategy_name', 'none')}"
        )
    
    async def _initialize_strategies(self) -> None:
        strategies = []
        if self.config.enable_vector_routing:
            strategies.append(("vector", VectorBasedRouting(
                embedding_cache=self.cache,
                ollama_host=self.config.ollama_host,
                logger=self.logger.getChild("vector")
            )))
        if self.config.enable_performance_tracking:
            strategies.append(("performance", PerformanceBasedRouting(
                logger=self.logger.getChild("performance")
            )))
        if self.config.enable_adaptive_routing and len(strategies) > 1:
            strategies.append(("adaptive", AdaptiveRouting(
                strategies=self.config.adaptive_routing_strategy_config,
                fallback_expert=self.config.default_expert,
                logger=self.logger.getChild("adaptive")
            )))
        for name, strategy in strategies:
            try:
                if hasattr(strategy, 'initialize'):
                    await strategy.initialize()
                self.strategies[name] = strategy
                self.logger.debug(f"Initialized {name} strategy")
            except Exception as e:
                self.logger.error(f"Failed to initialize {name} strategy: {str(e)}", exc_info=True)
        self.metrics.set_active_strategies(len(self.strategies))
    
    async def _record_adaptive_learning_metrics(self, metrics: Dict[str, Any]) -> None:
        if not self.config.enable_metrics:
            return
        try:
            metric_type = metrics.get('type')
            if metric_type == 'adaptive_learning_feedback':
                self.metrics.record_adaptive_learning_feedback(
                    success=metrics.get('success', False),
                    expert=metrics.get('expert', 'unknown')
                )
        except Exception as e:
            self.logger.error(f"Error recording adaptive learning metrics: {e}")
    
    async def shutdown(self) -> None:
        if hasattr(self, 'metrics') and self.metrics:
            await self.metrics.shutdown()
        if hasattr(self, 'adaptive_learning_manager') and self.adaptive_learning_manager:
            await self.adaptive_learning_manager.shutdown()
        for name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'shutdown'):
                    await strategy.shutdown()
                self.logger.debug(f"Shut down {name} strategy")
            except Exception as e:
                self.logger.error(f"Error shutting down {name} strategy: {str(e)}", exc_info=True)
        self._initialized = False
        self.logger.info("ExpertRouter shutdown complete")
    
    def register_expert_handler(self, expert_type: Union[HuiHuiExpertType, str], handler: ExpertHandler) -> None:
        if isinstance(expert_type, str):
            try:
                expert_type = HuiHuiExpertType(expert_type)
            except ValueError:
                raise ValueError(f"Unknown expert type: {expert_type}")
        self.expert_handlers[expert_type] = handler
        self.logger.debug(f"Registered handler for expert type: {expert_type.value}")

    async def select_expert(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> MOEGatingNetworkV2_5:
        if not self.active_strategy:
            self.logger.warning("No active routing strategy, using default expert")
            return MOEGatingNetworkV2_5(
                expert_type=self.config.default_expert,
                confidence=0.0,
                strategy_used="fallback",
                metadata={"reason": "no_active_strategy"}
            )
        try:
            decision = await self.active_strategy.select_expert(prompt, context)
            self.logger.debug(
                f"Selected expert: {decision.expert_type.value} "
                f"(confidence: {decision.confidence:.2f}, "
                f"strategy: {decision.strategy_used})"
            )
            return decision
        except Exception as e:
            self.logger.error(f"Error in expert selection: {str(e)}", exc_info=True)
            return MOEGatingNetworkV2_5(
                expert_type=self.config.default_expert,
                confidence=0.0,
                strategy_used="fallback",
                metadata={
                    "error": str(e),
                    "fallback_reason": "selection_error"
                }
            )

    async def route(self, prompt: str, context: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        if not self._initialized and not self._shutdown_flag:
            await self.initialize()
        request_id = str(uuid.uuid4())
        start_time = time.time()
        timeout = timeout or self.config.default_timeout
        if context is None:
            context = {}
        context.update({
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        })
        self.logger.info(f"Routing request {request_id}: {prompt[:100]}...")
        try:
            routing_decision = await self.select_expert(prompt, context)
            expert_type = routing_decision.expert_type
            handler = self.expert_handlers.get(expert_type)
            if handler is None:
                raise ValueError(f"No handler registered for expert type: {expert_type}")
            self.logger.debug(f"Dispatching to {expert_type.value} expert with confidence {routing_decision.confidence:.2f}")
            try:
                response = await asyncio.wait_for(handler(prompt, context), timeout=timeout)
                success = True
            except asyncio.TimeoutError:
                raise TimeoutError(f"Expert {expert_type.value} timed out after {timeout} seconds")
            duration = time.time() - start_time
            self._record_metrics(expert_type=expert_type, success=success, duration=duration, routing_decision=routing_decision)
            return {
                "request_id": request_id,
                "expert_type": expert_type.value,
                "response": response,
                "metadata": {
                    "duration_seconds": duration,
                    "confidence": routing_decision.confidence,
                    "strategy": routing_decision.strategy_used,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            duration = time.time() - start_time
            self._record_error_metrics(e, duration)
            raise
    
    def _record_metrics(self, expert_type: HuiHuiExpertType, success: bool, duration: float, routing_decision: RoutingDecision) -> None:
        """Record performance metrics for a request."""
        if not self.config.enable_metrics:
            return
        
        self.metrics.record_request(
            expert_type=expert_type.value,
            success=success,
            duration=duration,
            confidence=routing_decision.confidence,
            strategy=routing_decision.strategy_used
        )
        
        # Update internal performance metrics
        if self.config.enable_performance_tracking:
            self.performance_metrics[expert_type].update(
                success=success,
                response_time=duration
            )

    def _record_error_metrics(self, error: Exception, duration: float) -> None:
        """
        Record metrics for a failed request.
        
        Args:
            error: The exception that was raised
            duration: How long the request took before failing (seconds)
        """
        if not self.config.enable_metrics:
            return
        self.metrics.record_error(error_type=type(error).__name__, duration=duration)

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the router.
        
        Returns:
            Dictionary with status information
        """
        return {
            "initialized": self._initialized,
            "uptime_seconds": time.time() - self._start_time,
            "active_strategy": getattr(self.active_strategy, 'strategy_name', 'none'),
            "available_strategies": list(self.strategies.keys()),
            "registered_handlers": [e.value for e in self.expert_handlers.keys()],
            "cache_stats": self.cache.get_stats(),
            "performance_metrics": {
                expert.value: metrics.model_dump(mode='json')
                for expert, metrics in self.performance_metrics.items()
            }
        }

# --- Backward Compatibility Aliases ---
HuiHuiRouter = ExpertRouter
AIRouter = ExpertRouter

# --- Example Usage ---
async def create_expert_router(ollama_host: str = "http://localhost:11434", **kwargs) -> Optional[ExpertRouter]:
    """Creates and initializes the ExpertRouter, including a connection check to the Ollama server."""
    router = ExpertRouter(config=RouterConfig(ollama_host=ollama_host, **kwargs))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ollama_host}/api/tags", timeout=5) as response:
                if response.status != 200:
                    logger.error(f"Failed to connect to Ollama server at {ollama_host}. Status: {response.status}")
                    return None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"Error connecting to Ollama server at {ollama_host}: {e}")
        return None
    await router.initialize()
    return router

async def market_regime_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    return {"analysis": f"Market regime analysis for: {prompt}"}

async def options_flow_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    return {"analysis": f"Options flow analysis for: {prompt}"}

async def sentiment_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    return {"analysis": f"Sentiment analysis for: {prompt}"}

async def orchestrator_handler(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    return {"analysis": f"Orchestrator analysis for: {prompt}"}



async def example_usage():
    """Sets up and runs an example of the ExpertRouter."""
    logger.info("Starting example_usage...")
    router = await create_expert_router()
    if not router:
        logger.error("❌ Failed to create ExpertRouter. Please check the logs for details.")
        return

    try:
        logger.info("Router created successfully. Registering expert handlers...")
        router.register_expert_handler(HuiHuiExpertType.MARKET_REGIME, market_regime_handler)
        router.register_expert_handler(HuiHuiExpertType.OPTIONS_FLOW, options_flow_handler)
        router.register_expert_handler(HuiHuiExpertType.SENTIMENT, sentiment_handler)
        router.register_expert_handler(HuiHuiExpertType.ORCHESTRATOR, orchestrator_handler)
        logger.info("Core expert handlers registered.")

        logger.info("--- Example router setup complete ---")
        await asyncio.sleep(0.1)

    except Exception:
        logger.error("An error occurred during example_usage execution.", exc_info=True)
    finally:
        if router and router._initialized:
            logger.info("Shutting down router in example_usage...")
            await router.shutdown()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    main_logger = logging.getLogger(__name__)
    main_logger.info("--- EXECUTING EXPERT ROUTER CORE ---")
    
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        main_logger.info("\nInterrupted by user. Shutting down.")
    except Exception as e:
        main_logger.error(f"An unexpected error occurred in main block: {e}", exc_info=True)
    
    main_logger.info("--- EXPERT ROUTER CORE EXECUTION FINISHED ---")



