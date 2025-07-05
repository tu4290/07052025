# EXPERT ROUTER - MODULAR IMPLEMENTATION
"""
This module provides a modular implementation of the ExpertRouter for
intelligent routing of queries to specialized AI experts.
"""

# Core components
from .core import ExpertRouter, RouterConfig
from data_models.ai_ml_models import (
    HuiHuiExpertType,
    MOEGatingNetworkV2_5 as RoutingDecision
)
from data_models import ExpertPerformanceHistory as PerformanceMetrics
from .cache import (
    UltraFastEmbeddingCache,
    create_embedding_cache,
    BaseCache
)
from .metrics import RouterMetrics
from .adaptive_learning import (
    AdaptiveLearningManager,
    AdaptiveLearningConfig,
    LearningMode
)


# Import strategies
from .strategies import (
    BaseRoutingStrategy,
    VectorBasedRouting,
    PerformanceBasedRouting,
    AdaptiveRouting
)

# Import utilities
from .utils import (
    generate_request_id,
    time_execution,
    retry_on_exception,
    validate_config,
    to_json_serializable,
    hash_string,
    Timer,
    AsyncCache,
    gather_with_concurrency,
    get_or_create_event_loop,
    run_async
)

# Backward compatibility
HuiHuiRouter = ExpertRouter
AIRouter = ExpertRouter

__all__ = [
    # Core classes
    'ExpertRouter',
    'RouterConfig',
    'RouterMetrics',
    'AdaptiveLearningManager',
    'AdaptiveLearningConfig',
    'LearningMode',
    'BaseCache',
    'create_embedding_cache',
    
    # Backward compatibility
    'HuiHuiRouter',
    'AIRouter',
    
    # Models
    'HuiHuiExpertType',
    'PerformanceMetrics',
    'ExpertPerformance',
    'RoutingDecision',
    
    # Caching
    'UltraFastEmbeddingCache',
    'AsyncCache',
    
    # Strategies
    'BaseRoutingStrategy',
    'VectorBasedRouting',
    'PerformanceBasedRouting',
    'AdaptiveRouting',
    
    # Utilities
    'generate_request_id',
    'time_execution',
    'retry_on_exception',
    'validate_config',
    'to_json_serializable',
    'hash_string',
    'Timer',
    'gather_with_concurrency',
    'get_or_create_event_loop',
    'run_async',
]
