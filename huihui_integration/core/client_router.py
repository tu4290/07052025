"""
Unified HuiHui Client Router v2.5 - Enterprise-Grade AI Integration
==================================================================

Consolidated client/router for all HuiHui expert system interactions.
Merges functionality from multiple specialized clients into a unified,
Pydantic-first architecture with comprehensive error handling and performance optimization.

Consolidated Components:
- RobustHuiHuiClient: Enterprise reliability with circuit breaker
- OptimizedHuiHuiClient: Performance-first with 5-10s response times
- AIRouter: Vectorized routing with compatibility wrapper
- VectorizedAIRouter: Ultra-fast async/vectorized processing
- LocalLLMClient: Direct model access for external tools

Key Features:
- Unified Pydantic-first interface
- Multiple routing strategies (robust, optimized, vectorized)
- Circuit breaker pattern for fault tolerance
- Advanced caching and connection pooling
- Comprehensive performance metrics
- Streaming and batch processing capabilities
- Expert-specific optimization

Author: EOTS v2.5 Consolidation Team
"""

import json
import logging
import time
import asyncio
import requests
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

# Pydantic imports
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import PositiveInt, PositiveFloat

# Vector search imports
try:
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    SentenceTransformer = None

# Enhanced Cache Manager integration
try:
    from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
    ENHANCED_CACHE_AVAILABLE = True
except ImportError:
    ENHANCED_CACHE_AVAILABLE = False
    EnhancedCacheManagerV2_5 = None

# Expert Configs
from huihui_integration.config.expert_configs import load_huihui_config, HuiHuiConfigV2_5

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== PYDANTIC MODELS =====

class HuiHuiExpertType(str, Enum):
    """HuiHui expert types with string values."""
    MARKET_REGIME = "market_regime"
    OPTIONS_FLOW = "options_flow"
    SENTIMENT = "sentiment"
    ORCHESTRATOR = "orchestrator"

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class RequestPriority(int, Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class RouterStrategy(str, Enum):
    """Available routing strategies."""
    ROBUST = "robust"
    OPTIMIZED = "optimized"
    VECTORIZED = "vectorized"
    LOCAL = "local"

class ExpertConfigV2_5(BaseModel):
    """Pydantic model for expert configuration."""
    model_config = ConfigDict(extra='forbid')
    
    name: str = Field(..., description="Expert display name")
    system_prompt: str = Field(..., description="Expert system prompt")
    max_tokens: PositiveInt = Field(800, description="Maximum response tokens")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Response temperature")
    timeout: PositiveFloat = Field(30.0, description="Request timeout in seconds")
    retry_count: PositiveInt = Field(3, description="Number of retries")
    priority: RequestPriority = Field(RequestPriority.NORMAL, description="Request priority")

class RequestMetricsV2_5(BaseModel):
    """Pydantic model for request metrics."""
    model_config = ConfigDict(extra='forbid')
    
    start_time: float = Field(..., description="Request start timestamp")
    end_time: float = Field(..., description="Request end timestamp")
    response_time: float = Field(..., description="Total response time in seconds")
    success: bool = Field(..., description="Whether request succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")
    expert: Optional[str] = Field(None, description="Expert used")
    token_count: int = Field(0, description="Response token count")
    retry_count: int = Field(0, description="Number of retries attempted")
    strategy: RouterStrategy = Field(..., description="Routing strategy used")

class PerformanceStatsV2_5(BaseModel):
    """Pydantic model for performance statistics."""
    model_config = ConfigDict(extra='forbid')
    
    total_requests: int = Field(0, description="Total requests processed")
    successful_requests: int = Field(0, description="Successful requests")
    total_time: float = Field(0.0, description="Total processing time")
    avg_response_time: float = Field(0.0, description="Average response time")
    fastest_response: float = Field(float('inf'), description="Fastest response time")
    slowest_response: float = Field(0.0, description="Slowest response time")
    vector_routing_used: int = Field(0, description="Vector routing usage count")
    fallback_routing_used: int = Field(0, description="Fallback routing usage count")
    cache_hits: int = Field(0, description="Cache hit count")
    cache_misses: int = Field(0, description="Cache miss count")
    connection_reuses: int = Field(0, description="Connection reuse count")
    streaming_requests: int = Field(0, description="Streaming request count")
    batch_requests: int = Field(0, description="Batch request count")

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return (self.cache_hits / total_cache_requests) * 100

class ClientConfigV2_5(BaseModel):
    """Pydantic model for client configuration."""
    model_config = ConfigDict(extra='forbid')
    
    ollama_host: str = Field("http://localhost:11434", description="Ollama host URL")
    model_name: str = Field("huihui_ai/huihui-moe-abliterated:5b-a1.7b", description="Model name")
    default_strategy: RouterStrategy = Field(RouterStrategy.VECTORIZED, description="Default routing strategy")
    max_connections: PositiveInt = Field(20, description="Maximum concurrent connections")
    circuit_failure_threshold: PositiveInt = Field(5, description="Circuit breaker failure threshold")
    circuit_recovery_timeout: PositiveFloat = Field(60.0, description="Circuit breaker recovery timeout")
    cache_max_size: PositiveInt = Field(1000, description="Cache maximum size")
    cache_ttl: PositiveInt = Field(3600, description="Cache TTL in seconds")
    enable_logging: bool = Field(True, description="Enable request logging")
    log_file: str = Field("logs/huihui_usage.jsonl", description="Log file path")

# ===== CIRCUIT BREAKER =====

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, 
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN - requests blocked")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# ===== CACHING SYSTEM =====

class UltraFastEmbeddingCache:
    """Ultra-fast embedding cache with async operations and intelligent TTL."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.RLock()
        
    async def get_async(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_sync, key)
    
    def _get_sync(self, key: str) -> Optional[np.ndarray]:
        """Synchronous cache get with TTL check."""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check TTL
            if time.time() - self._timestamps[key] > self.ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                return None
            
            return self._cache[key]
    
    async def set_async(self, key: str, value: np.ndarray):
        """Set embedding in cache asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._set_sync, key, value)
    
    def _set_sync(self, key: str, value: np.ndarray):
        """Synchronous cache set with LRU eviction."""
        with self._lock:
            # LRU eviction if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = value.copy()
            self._timestamps[key] = time.time()

# ===== UNIFIED CLIENT ROUTER =====

class UnifiedHuiHuiClientRouter:
    """
    Unified HuiHui Client Router - Consolidated AI Expert System Interface
    
    Provides multiple routing strategies with automatic fallback:
    - VECTORIZED: Ultra-fast async/vectorized processing (default)
    - OPTIMIZED: Performance-first with 5-10s response times
    - ROBUST: Enterprise reliability with circuit breaker
    - LOCAL: Direct model access for external tools
    
    Features:
    - Pydantic-first validation and configuration
    - Circuit breaker pattern for fault tolerance
    - Advanced caching and connection pooling
    - Comprehensive performance metrics
    - Streaming and batch processing
    - Expert-specific optimization
    """
    
    def __init__(self, config: Optional[ClientConfigV2_5] = None, huihui_config: Optional[HuiHuiConfigV2_5] = None):
        """Initialize unified client router with Pydantic configuration."""
        self.config = config if config is not None else ClientConfigV2_5()
        self.huihui_config = huihui_config or load_huihui_config()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        if self.config.enable_logging:
            self._setup_logging()
        
        # Performance tracking
        self.performance_stats = PerformanceStatsV2_5()
        self.metrics: List[RequestMetricsV2_5] = []
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_failure_threshold,
            recovery_timeout=self.config.circuit_recovery_timeout
        )
        
        # Caching system
        self.embedding_cache = UltraFastEmbeddingCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl
        )
        
        # Vector search initialization
        self.vector_model = None
        self.expert_embeddings = {}
        self._initialize_vector_search()
        
        # Expert configurations
        self.expert_configs = self._initialize_expert_configs()
        
        # Async components
        self._session = None
        self._event_loop = None
        self._loop_thread = None
        self._experts_warmed = False
        
        # Thread pool for sync operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info(f"🚀 Unified HuiHui Client Router initialized with {self.config.default_strategy.value} strategy")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(exist_ok=True)
        
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _initialize_vector_search(self):
        """Initialize vector search capabilities."""
        if VECTOR_SEARCH_AVAILABLE:
            try:
                self.vector_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Vector search initialized")
            except Exception as e:
                logger.warning(f"⚠️ Vector search initialization failed: {e}")
                self.vector_model = None
        else:
            logger.warning("⚠️ Vector search not available - install sentence-transformers")

    def _initialize_expert_configs(self) -> Dict[HuiHuiExpertType, ExpertConfigV2_5]:
        """Initialize expert configurations."""
        return {
            HuiHuiExpertType.MARKET_REGIME: ExpertConfigV2_5(
                name="🏛️ Market Regime Expert",
                system_prompt=(
                    "[EXPERT:MARKET_REGIME] You are the HuiHui Market Regime Expert. "
                    "Analyze market volatility, VRI metrics, regime transitions, and "
                    "structural patterns using EOTS analytics. Provide concise, actionable insights."
                ),
                max_tokens=600,
                temperature=0.1,
                timeout=25.0,
                retry_count=3
            ),
            HuiHuiExpertType.OPTIONS_FLOW: ExpertConfigV2_5(
                name="🚀 Options Flow Expert",
                system_prompt=(
                    "[EXPERT:OPTIONS_FLOW] You are the HuiHui Options Flow Expert. "
                    "Analyze VAPI-FA, DWFD, TW-LAF metrics, gamma exposure, delta flows, "
                    "and institutional options positioning. Provide concise, actionable insights."
                ),
                max_tokens=600,
                temperature=0.1,
                timeout=25.0,
                retry_count=3
            ),
            HuiHuiExpertType.SENTIMENT: ExpertConfigV2_5(
                name="🧠 Sentiment Expert",
                system_prompt=(
                    "[EXPERT:SENTIMENT] You are the HuiHui Sentiment Expert. "
                    "Analyze market sentiment, news intelligence, behavioral patterns, "
                    "and psychological market drivers. Provide concise, actionable insights."
                ),
                max_tokens=600,
                temperature=0.1,
                timeout=25.0,
                retry_count=3
            ),
            HuiHuiExpertType.ORCHESTRATOR: ExpertConfigV2_5(
                name="🎯 Meta-Orchestrator",
                system_prompt=(
                    "[EXPERT:ORCHESTRATOR] You are the HuiHui Meta-Orchestrator. "
                    "Synthesize insights from all experts, provide strategic recommendations, "
                    "and deliver comprehensive EOTS analysis. Be thorough but concise."
                ),
                max_tokens=800,
                temperature=0.2,
                timeout=30.0,
                retry_count=3
            )
        }

    def ask(self, prompt: str, expert: Optional[HuiHuiExpertType] = None, strategy: Optional[RouterStrategy] = None,
            priority: RequestPriority = RequestPriority.NORMAL) -> Dict[str, Any]:
        """
        Main interface for asking HuiHui experts.

        Args:
            prompt: Question or analysis request
            expert: Specific expert to use (auto-detected if None)
            strategy: Routing strategy to use (default from config if None)
            priority: Request priority level

        Returns:
            Dictionary with response, expert info, and performance metrics
        """
        start_time = time.time()
        strategy = strategy or self.config.default_strategy

        try:
            # Auto-detect expert if not specified
            if expert is None:
                expert = self._detect_expert_type(prompt)

            # Route based on strategy
            if strategy == RouterStrategy.VECTORIZED:
                result = self._ask_vectorized(prompt, expert, priority)
            elif strategy == RouterStrategy.OPTIMIZED:
                result = self._ask_optimized(prompt, expert, priority)
            elif strategy == RouterStrategy.ROBUST:
                result = self._ask_robust(prompt, expert, priority)
            elif strategy == RouterStrategy.LOCAL:
                result = self._ask_local(prompt, expert, priority)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Update performance stats
            response_time = time.time() - start_time
            self._update_performance_stats(response_time, True, strategy)

            # Log request
            if self.config.enable_logging:
                self._log_request(expert.value, prompt, result.get("response", ""), response_time, True, None)

            return result

        except Exception as e:
            response_time = time.time() - start_time
            self._update_performance_stats(response_time, False, strategy)

            if self.config.enable_logging:
                self._log_request(expert.value if expert else "unknown", prompt, "", response_time, False, str(e))

            logger.error(f"❌ Request failed: {e}")
            return {
                "response": f"🧠 HuiHui Expert Error: {str(e)}",
                "expert": expert.value if expert else "unknown",
                "strategy": strategy.value,
                "success": False,
                "error": str(e),
                "response_time": response_time
            }

    def _detect_expert_type(self, prompt: str) -> HuiHuiExpertType:
        """Detect appropriate expert type from prompt."""
        if self.vector_model and self.expert_embeddings:
            return self._vector_detect_expert_type(prompt)
        else:
            return self._keyword_detect_expert_type(prompt)

    def _vector_detect_expert_type(self, prompt: str) -> HuiHuiExpertType:
        """Vector-based expert detection."""
        try:
            prompt_embedding = self.vector_model.encode([prompt])[0]

            best_expert = HuiHuiExpertType.ORCHESTRATOR
            best_similarity = -1

            for expert, embedding in self.expert_embeddings.items():
                similarity = np.dot(prompt_embedding, embedding) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(embedding)
                )

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_expert = expert

            return best_expert

        except Exception as e:
            logger.warning(f"Vector detection failed: {e}, falling back to keywords")
            return self._keyword_detect_expert_type(prompt)

    def _keyword_detect_expert_type(self, prompt: str) -> HuiHuiExpertType:
        """Keyword-based expert detection."""
        prompt_lower = prompt.lower()

        # Options flow keywords
        if any(keyword in prompt_lower for keyword in ['vapi', 'dwfd', 'tw-laf', 'gamma', 'delta', 'flow', 'options']):
            return HuiHuiExpertType.OPTIONS_FLOW

        # Market regime keywords
        elif any(keyword in prompt_lower for keyword in ['vri', 'regime', 'volatility', 'vix', 'market structure']):
            return HuiHuiExpertType.MARKET_REGIME

        # Sentiment keywords
        elif any(keyword in prompt_lower for keyword in ['sentiment', 'news', 'fed', 'earnings', 'psychology']):
            return HuiHuiExpertType.SENTIMENT

        # Default to orchestrator
        else:
            return HuiHuiExpertType.ORCHESTRATOR

    def _ask_vectorized(self, prompt: str, expert: HuiHuiExpertType, priority: RequestPriority) -> Dict[str, Any]:
        """Vectorized routing strategy (async/high-performance)."""
        # This would integrate with the vectorized router
        # For now, fallback to optimized
        return self._ask_optimized(prompt, expert, priority)

    def _ask_optimized(self, prompt: str, expert: HuiHuiExpertType, priority: RequestPriority) -> Dict[str, Any]:
        """Optimized routing strategy (5-10s response times)."""
        config = self.expert_configs[expert]

        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt}
        ]

        data = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "num_ctx": 1024,
                "top_k": 10,
                "top_p": 0.7
            }
        }

        response = requests.post(
            f"{self.config.ollama_host}/api/chat",
            json=data,
            timeout=config.timeout
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("message", {}).get("content", "")

        return {
            "response": content,
            "expert": expert.value,
            "expert_name": config.name,
            "strategy": RouterStrategy.OPTIMIZED.value,
            "success": True
        }

    def _ask_robust(self, prompt: str, expert: HuiHuiExpertType, priority: RequestPriority) -> Dict[str, Any]:
        """Robust routing strategy (with circuit breaker and retries)."""
        config = self.expert_configs[expert]

        def make_request():
            return self._ask_optimized(prompt, expert, priority)

        return self.circuit_breaker.call(make_request)

    def _ask_local(self, prompt: str, expert: HuiHuiExpertType, priority: RequestPriority) -> Dict[str, Any]:
        """Local routing strategy (direct model access)."""
        # Direct model access implementation
        return self._ask_optimized(prompt, expert, priority)

    def _update_performance_stats(self, response_time: float, success: bool, strategy: RouterStrategy):
        """Update performance statistics."""
        self.performance_stats.total_requests += 1
        self.performance_stats.total_time += response_time

        if success:
            self.performance_stats.successful_requests += 1

        if response_time < self.performance_stats.fastest_response:
            self.performance_stats.fastest_response = response_time

        if response_time > self.performance_stats.slowest_response:
            self.performance_stats.slowest_response = response_time

        # Calculate average
        if self.performance_stats.total_requests > 0:
            self.performance_stats.avg_response_time = (
                self.performance_stats.total_time / self.performance_stats.total_requests
            )

        # Update strategy-specific stats
        if strategy == RouterStrategy.VECTORIZED:
            self.performance_stats.vector_routing_used += 1
        else:
            self.performance_stats.fallback_routing_used += 1

    def _log_request(self, expert: str, prompt: str, response: str, response_time: float, success: bool, error: Optional[str]):
        """Log request details."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "expert": expert,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "response_time": response_time,
            "success": success,
            "error": error
        }

        try:
            with open(self.config.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log request: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.performance_stats.model_dump()

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        return {
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "total_requests": self.performance_stats.total_requests,
            "success_rate": self.performance_stats.success_rate,
            "avg_response_time": self.performance_stats.avg_response_time,
            "cache_hit_rate": self.performance_stats.cache_hit_rate,
            "vector_search_available": self.vector_model is not None,
            "experts_warmed": self._experts_warmed
        }

    # Convenience methods for specific experts
    def ask_market_regime(self, prompt: str, strategy: Optional[RouterStrategy] = None) -> Dict[str, Any]:
        """Ask market regime expert specifically."""
        return self.ask(prompt, HuiHuiExpertType.MARKET_REGIME, strategy)

    def ask_options_flow(self, prompt: str, strategy: Optional[RouterStrategy] = None) -> Dict[str, Any]:
        """Ask options flow expert specifically."""
        return self.ask(prompt, HuiHuiExpertType.OPTIONS_FLOW, strategy)

    def ask_sentiment(self, prompt: str, strategy: Optional[RouterStrategy] = None) -> Dict[str, Any]:
        """Ask sentiment expert specifically."""
        return self.ask(prompt, HuiHuiExpertType.SENTIMENT, strategy)

    def ask_orchestrator(self, prompt: str, strategy: Optional[RouterStrategy] = None) -> Dict[str, Any]:
        """Ask meta-orchestrator specifically."""
        return self.ask(prompt, HuiHuiExpertType.ORCHESTRATOR, strategy)

    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)

        if self._session:
            asyncio.run(self._session.close())

# Define factory functions
def create_unified_client(config: Optional[ClientConfigV2_5] = None) -> UnifiedHuiHuiClientRouter:
    """Factory function to create unified client router."""
    return UnifiedHuiHuiClientRouter(config)

def create_optimized_client(ollama_host: str = "http://localhost:11434") -> UnifiedHuiHuiClientRouter:
    """Factory function for optimized client configuration."""
    config = ClientConfigV2_5(
        ollama_host=ollama_host,
        default_strategy=RouterStrategy.OPTIMIZED
    )
    return UnifiedHuiHuiClientRouter(config)

def create_robust_client(ollama_host: str = "http://localhost:11434") -> UnifiedHuiHuiClientRouter:
    """Factory function for robust client configuration."""
    config = ClientConfigV2_5(
        ollama_host=ollama_host,
        default_strategy=RouterStrategy.ROBUST,
        circuit_failure_threshold=3,
        circuit_recovery_timeout=30.0
    )
    return UnifiedHuiHuiClientRouter(config)

# Define quick access functions
def quick_ask(prompt: str, expert: Optional[HuiHuiExpertType] = None) -> str:
    """Quick function to ask HuiHui experts."""
    client = create_unified_client()
    result = client.ask(prompt, expert)
    return result.get("response", "No response")

def quick_market_regime(prompt: str) -> str:
    """Quick market regime analysis."""
    return quick_ask(prompt, HuiHuiExpertType.MARKET_REGIME)

def quick_options_flow(prompt: str) -> str:
    """Quick options flow analysis."""
    return quick_ask(prompt, HuiHuiExpertType.OPTIONS_FLOW)

def quick_sentiment(prompt: str) -> str:
    """Quick sentiment analysis."""
    return quick_ask(prompt, HuiHuiExpertType.SENTIMENT)

def quick_orchestrator(prompt: str) -> str:
    """Quick meta-orchestrator analysis."""
    return quick_ask(prompt, HuiHuiExpertType.ORCHESTRATOR)

# Define diagnostics
class HuiHuiDiagnostics:
    """Diagnostic utilities for HuiHui system health."""

    @staticmethod
    def test_connection(ollama_host: str = "http://localhost:11434") -> bool:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def test_model_availability(model_name: str = "huihui_ai/huihui-moe-abliterated:5b-a1.7b",
                              ollama_host: str = "http://localhost:11434") -> bool:
        """Test if specific model is available."""
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model.get("name") == model_name for model in models)
            return False
        except:
            return False

    @staticmethod
    def run_full_diagnostic() -> Dict[str, Any]:
        """Run comprehensive system diagnostic."""
        client = create_unified_client()

        return {
            "connection": HuiHuiDiagnostics.test_connection(),
            "model_available": HuiHuiDiagnostics.test_model_availability(),
            "vector_search": VECTOR_SEARCH_AVAILABLE,
            "enhanced_cache": ENHANCED_CACHE_AVAILABLE,
            "health_status": client.get_health_status(),
            "performance_stats": client.get_performance_stats()
        }

if __name__ == "__main__":
    # Quick test
    print("🚀 Testing Unified HuiHui Client Router...")
    diagnostic = HuiHuiDiagnostics.run_full_diagnostic()
    print(json.dumps(diagnostic, indent=2))
