"""
Expert Coordinator Configuration

This module contains configuration classes for the Expert Coordinator.
"""
from typing import Dict, Any, Optional
from pydantic import Field, BaseModel, ConfigDict
from .expert_models import CoordinationMode

class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = Field(
        default=5,
        description="Number of failures before opening the circuit"
    )
    recovery_timeout: int = Field(
        default=60,
        description="Seconds before attempting to close the circuit"
    )
    expected_exception: tuple = Field(
        default=(Exception,),
        description="Exceptions that should trigger the circuit breaker"
    )

class LoadBalancingConfig(BaseModel):
    """Configuration for adaptive load balancing."""
    initial_weight: float = Field(
        default=1.0,
        description="Initial weight for new experts"
    )
    min_weight: float = Field(
        default=0.1,
        description="Minimum weight an expert can have"
    )
    max_weight: float = Field(
        default=10.0,
        description="Maximum weight an expert can have"
    )
    weight_decay: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Decay factor for expert weights per hour"
    )

class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retry attempts"
    )
    initial_delay: float = Field(
        default=0.1,
        ge=0.0,
        description="Initial delay between retries in seconds"
    )
    max_delay: float = Field(
        default=5.0,
        ge=0.0,
        description="Maximum delay between retries in seconds"
    )
    backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        description="Exponential backoff multiplier"
    )

class CoordinationStrategy(BaseModel):
    """AI-powered coordination strategy with adaptive behaviors."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    mode: CoordinationMode = Field(
        default=CoordinationMode.ADAPTIVE,
        description="Coordination strategy mode"
    )
    expert_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Weights for weighted expert selection"
    )
    timeout_seconds: float = Field(
        default=30.0, 
        ge=0.1,
        description="Global timeout for coordination"
    )
    consensus_threshold: float = Field(
        default=0.7, 
        ge=0.5, 
        le=1.0,
        description="Threshold for consensus decisions"
    )
    confidence_threshold: float = Field(
        default=0.6, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence for accepting decisions"
    )
    parallel_execution: bool = Field(
        default=True,
        description="Enable parallel execution of expert queries"
    )
    fallback_strategy: Optional[str] = Field(
        default=None,
        description="Fallback strategy name if primary fails"
    )
    priority_expert: Optional[str] = Field(
        default=None,
        description="Expert ID to prioritize in certain modes"
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration"
    )
    load_balancing: LoadBalancingConfig = Field(
        default_factory=LoadBalancingConfig,
        description="Load balancing configuration"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retry configuration"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Batch size for batch processing mode"
    )
    stream_chunk_size: int = Field(
        default=1024,
        ge=1,
        description="Chunk size for streaming responses"
    )
    
    def validate_weights(self) -> None:
        """Validate and normalize expert weights."""
        if not self.expert_weights:
            return
            
        # Normalize weights to sum to 1.0
        total = sum(self.expert_weights.values())
        if total > 0:
            self.expert_weights = {k: v/total for k, v in self.expert_weights.items()}
    
    def get_effective_weights(self, expert_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Get effective weights considering expert health and performance."""
        if not self.expert_weights:
            return {}
            
        # Apply health score to weights
        effective_weights = {}
        for expert_id, weight in self.expert_weights.items():
            if expert_id in expert_metrics:
                health_score = getattr(expert_metrics[expert_id], 'health_score', 0.5)
                effective_weights[expert_id] = weight * health_score
        
        # Normalize
        total = sum(effective_weights.values())
        if total > 0:
            return {k: v/total for k, v in effective_weights.items()}
        return {}

# Default strategy configurations
DEFAULT_STRATEGIES = {
    CoordinationMode.CONSENSUS: CoordinationStrategy(
        mode=CoordinationMode.CONSENSUS,
        consensus_threshold=0.7,
        timeout_seconds=45.0,
        parallel_execution=True
    ),
    CoordinationMode.WEIGHTED: CoordinationStrategy(
        mode=CoordinationMode.WEIGHTED,
        timeout_seconds=30.0,
        parallel_execution=True
    ),
    CoordinationMode.EMERGENCY: CoordinationStrategy(
        mode=CoordinationMode.EMERGENCY,
        timeout_seconds=10.0,
        consensus_threshold=0.5,
        parallel_execuration=True,
        fallback_strategy="fastest_available"
    )
}

__all__ = [
    'CircuitBreakerConfig',
    'LoadBalancingConfig',
    'RetryConfig',
    'CoordinationStrategy',
    'DEFAULT_STRATEGIES'
]
