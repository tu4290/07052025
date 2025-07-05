"""
Expert Coordinator Data Models

This module contains all data models and schemas used by the Expert Coordinator.
"""
from enum import Enum
from typing import Dict, Optional, Deque
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import Field, ConfigDict

class CircuitState(str, Enum):
    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"

class CoordinationMode(str, Enum):
    CONSENSUS = "consensus"
    WEIGHTED = "weighted"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"
    EMERGENCY = "emergency"
    ADAPTIVE = "adaptive"

@dataclass
class AdaptiveWindow:
    """Sliding window for tracking metrics with adaptive sizing."""
    max_size: int = 100
    window: Deque[float] = field(init=False)
    _sum: float = field(init=False, default=0.0)
    
    def __post_init__(self):
        from collections import deque
        self.window = deque(maxlen=self.max_size)
    
    def add(self, value: float) -> None:
        if len(self.window) == self.max_size:
            self._sum -= self.window[0]
        self.window.append(value)
        self._sum += value
    
    def average(self) -> float:
        if not self.window:
            return 0.0
        return self._sum / len(self.window)
    
    def percentile(self, p: float) -> float:
        if not self.window:
            return 0.0
        sorted_window = sorted(self.window)
        k = (len(sorted_window) - 1) * p
        f = int(k)
        c = k - f
        return sorted_window[f] * (1 - c) + sorted_window[min(f + 1, len(sorted_window) - 1)] * c

class ExpertPerformanceMetrics:
    """Comprehensive performance metrics for each expert with adaptive learning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    expert_id: str
    accuracy_score: float = Field(ge=0.0, le=1.0, default=0.7)
    response_time_ms: float = Field(ge=0.0, default=100.0)
    confidence_reliability: float = Field(ge=0.0, le=1.0, default=0.8)
    market_condition_performance: Dict[str, float] = Field(default_factory=dict)
    recent_success_rate: float = Field(ge=0.0, le=1.0, default=0.8)
    specialization_strength: float = Field(ge=0.0, le=1.0, default=0.7)
    coordination_compatibility: float = Field(ge=0.0, le=1.0, default=0.8)
    learning_rate: float = Field(ge=0.0, default=0.1)
    circuit_state: CircuitState = CircuitState.CLOSED
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    
    def __init__(self, **data):
        super().__init__(**data)
        self.response_times = AdaptiveWindow()
        self.error_rates = AdaptiveWindow(1000)
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-1) based on multiple factors."""
        weights = {
            'accuracy': 0.3,
            'response_time': 0.2,
            'success_rate': 0.3,
            'circuit_health': 0.2
        }
        
        # Normalize response time (lower is better)
        response_time_norm = max(0, min(1, 1 - (self.response_time_ms / 5000)))
        
        # Calculate circuit health
        circuit_health = 1.0
        if self.circuit_state == CircuitState.OPEN:
            circuit_health = 0.2
        elif self.circuit_state == CircuitState.HALF_OPEN:
            circuit_health = 0.5
            
        # Calculate weighted score
        score = (
            weights['accuracy'] * self.accuracy_score +
            weights['response_time'] * response_time_norm +
            weights['success_rate'] * self.recent_success_rate +
            weights['circuit_health'] * circuit_health
        )
        
        return min(1.0, max(0.0, score))
    
    def record_success(self, response_time_ms: float) -> None:
        """Record a successful request."""
        self.response_times.add(response_time_ms)
        self.response_time_ms = self.response_times.average()
        self.successful_requests += 1
        self.total_requests += 1
        self.recent_success_rate = (
            min(1.0, self.successful_requests / max(1, self.total_requests))
        )
        self.consecutive_failures = 0
        
        # Update accuracy score (simple moving average)
        self.accuracy_score = (
            self.accuracy_score * (1 - self.learning_rate) +
            self.recent_success_rate * self.learning_rate
        )
    
    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed request."""
        self.error_rates.add(1.0)  # 1 for failure
        self.total_requests += 1
        self.consecutive_failures += 1
        self.last_failure = datetime.utcnow()
        
        # Update success rate
        self.recent_success_rate = (
            max(0.0, 1 - (self.error_rates.average() / 100.0))
        )
        
        # Update circuit state if needed
        if self.consecutive_failures > 5:  # Threshold for circuit breaking
            self.circuit_state = CircuitState.OPEN

class MarketConditionContext:
    """Advanced market condition context for coordination decisions."""
    volatility_regime: str
    market_trend: str
    options_flow_intensity: str
    sentiment_regime: str
    time_of_day: str
    market_stress_level: float = Field(ge=0.0, le=1.0)
    liquidity_condition: str
    news_impact_level: float = Field(ge=0.0, le=1.0)

@dataclass
class TradeFeedback:
    """Trade feedback data class."""
    expert_id: str
    trade_id: str
    pnl: float
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    metadata: dict = field(default_factory=dict)

# Re-export commonly used types
__all__ = [
    'CircuitState',
    'CoordinationMode',
    'AdaptiveWindow',
    'ExpertPerformanceMetrics',
    'MarketConditionContext',
    'TradeFeedback'
]
