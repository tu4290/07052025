"""
Expert Metrics Collection

This module handles metrics collection and reporting for the Expert Coordinator.
"""
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge
from .expert_models import CircuitState

class MetricsCollector:
    """Collects and reports metrics for the coordinator."""
    
    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            'coordinator_requests_total',
            'Total number of requests',
            ['expert', 'status']
        )
        self.request_duration = Histogram(
            'coordinator_request_duration_seconds',
            'Request duration in seconds',
            ['expert']
        )
        self.expert_load = Gauge(
            'coordinator_expert_load',
            'Current load per expert',
            ['expert']
        )
        self.circuit_breaker_state = Gauge(
            'coordinator_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['expert']
        )
        
        # Performance metrics
        self.success_rate = Gauge(
            'coordinator_success_rate',
            'Success rate of expert decisions',
            ['expert']
        )
        self.error_rate = Gauge(
            'coordinator_error_rate',
            'Error rate of expert decisions',
            ['expert']
        )
        self.response_time = Histogram(
            'coordinator_response_time_seconds',
            'Response time in seconds',
            ['expert']
        )
        
        # Coordination metrics
        self.coordination_time = Histogram(
            'coordinator_coordination_time_seconds',
            'Time spent in coordination logic',
            ['strategy']
        )
        self.consensus_quality = Gauge(
            'coordinator_consensus_quality',
            'Quality of consensus (0-1)',
            ['strategy']
        )
    
    def record_request(self, expert: str, duration: float, success: bool = True) -> None:
        """Record a request with its outcome.
        
        Args:
            expert: Expert ID
            duration: Request duration in seconds
            success: Whether the request was successful
        """
        status = 'success' if success else 'error'
        self.requests_total.labels(expert=expert, status=status).inc()
        self.request_duration.labels(expert=expert).observe(duration)
        
        if success:
            self.success_rate.labels(expert=expert).set(1.0)
            self.error_rate.labels(expert=expert).set(0.0)
        else:
            self.error_rate.labels(expert=expert).set(1.0)
    
    def update_circuit_state(self, expert: str, state: CircuitState) -> None:
        """Update circuit breaker state.
        
        Args:
            expert: Expert ID
            state: New circuit breaker state
        """
        state_value = {
            CircuitState.CLOSED: 0,
            CircuitState.HALF_OPEN: 1,
            CircuitState.OPEN: 2
        }.get(state, 0)
        self.circuit_breaker_state.labels(expert=expert).set(state_value)
    
    def update_expert_load(self, expert: str, load: int) -> None:
        """Update current load for an expert.
        
        Args:
            expert: Expert ID
            load: Current number of active requests
        """
        self.expert_load.labels(expert=expert).set(load)
    
    def record_coordination_metrics(
        self,
        strategy: str,
        duration: float,
        consensus_quality: Optional[float] = None
    ) -> None:
        """Record coordination-specific metrics.
        
        Args:
            strategy: Strategy name
            duration: Coordination duration in seconds
            consensus_quality: Quality of consensus (0-1) if applicable
        """
        self.coordination_time.labels(strategy=strategy).observe(duration)
        if consensus_quality is not None:
            self.consensus_quality.labels(strategy=strategy).set(consensus_quality)

# Create a default metrics collector instance
metrics = MetricsCollector()

__all__ = ['MetricsCollector', 'metrics']
