"""
🎯 EXPERT ROUTER - DATA MODELS
==================================================================

This module contains the data models used by the ExpertRouter system.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, Deque, Any, Tuple
from collections import deque, defaultdict
from datetime import datetime

from .hui_hui_schemas import HuiHuiExpertType

class PerformanceMetrics(BaseModel):
    """Tracks performance metrics for experts over time."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_response_time: float = 0.0
    last_used: Optional[datetime] = None
    vector_routing_used: int = 0
    fallback_routing_used: int = 0
    vector_accuracy: float = 0.0
    connection_reuses: int = 0
    total_connections: int = 0
    fastest_response: float = float('inf')
    slowest_response: float = 0.0

    model_config = ConfigDict(extra='forbid')

    @property
    def connection_reuse_rate(self) -> float:
        if self.total_connections == 0:
            return 0.0
        return self.connection_reuses / self.total_connections

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries

    @property
    def average_response_time(self) -> float:
        if self.successful_queries == 0:
            return 0.0
        return self.total_response_time / self.successful_queries

    def update(self, success: bool, response_time: float, used_vector: bool = False, is_fallback: bool = False, connection_reused: bool = False) -> None:
        self.total_queries += 1
        if success:
            self.successful_queries += 1
            self.total_response_time += response_time
            self.fastest_response = min(self.fastest_response, response_time)
            self.slowest_response = max(self.slowest_response, response_time)
        else:
            self.failed_queries += 1
        if used_vector:
            self.vector_routing_used += 1
            if success:
                self.vector_accuracy = ((self.vector_accuracy * (self.vector_routing_used - 1) + 1) / self.vector_routing_used)
            else:
                self.vector_accuracy = ((self.vector_accuracy * (self.vector_routing_used - 1)) / self.vector_routing_used)
        if is_fallback:
            self.fallback_routing_used += 1
        if connection_reused:
            self.connection_reuses += 1
        self.total_connections += 1
        self.last_used = datetime.utcnow()

class ExpertPerformance(BaseModel):
    """Tracks performance metrics for each expert type."""
    metrics: Dict[HuiHuiExpertType, PerformanceMetrics] = Field(default_factory=lambda: defaultdict(PerformanceMetrics))
    recent_decisions: Deque[Tuple[HuiHuiExpertType, float]] = Field(default_factory=deque)
    max_history: int = 1000

    model_config = ConfigDict(extra='forbid')

    def update_metrics(self, expert_type: HuiHuiExpertType, success: bool, response_time: float) -> None:
        self.metrics[expert_type].update(success, response_time)
        self.recent_decisions.append((expert_type, datetime.utcnow().timestamp()))
        if len(self.recent_decisions) > self.max_history:
            self.recent_decisions.popleft()

    def get_recent_usage(self, window_seconds: int = 3600) -> Dict[HuiHuiExpertType, int]:
        cutoff = datetime.utcnow().timestamp() - window_seconds
        counts = defaultdict(int)
        for expert_type, timestamp in reversed(self.recent_decisions):
            if timestamp < cutoff:
                break
            counts[expert_type] += 1
        return dict(counts)

class RoutingDecision(BaseModel):
    """Represents a routing decision made by the router."""
    expert_type: HuiHuiExpertType
    confidence: float
    strategy_used: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra='forbid',
        json_encoders={HuiHuiExpertType: lambda v: v.value}
    )

    def __str__(self) -> str:
        return f"RoutingDecision(expert_type={self.expert_type.value}, confidence={self.confidence:.2f}, strategy={self.strategy_used})"