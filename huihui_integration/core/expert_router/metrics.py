"""
🎯 EXPERT ROUTER - METRICS AND MONITORING
==================================================================

This module provides metrics collection and monitoring capabilities
for the ExpertRouter system.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Try to import Prometheus metrics if available
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, start_http_server, generate_latest, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Define dummy classes for when Prometheus is not available
    class _DummyMetric:
        def __init__(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def dec(self, *args, **kwargs):
            pass
        def time(self):
            return _DummyContextManager()
    
    class _DummyContextManager:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    Counter = Gauge = Histogram = Summary = _DummyMetric


logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For HISTOGRAM
    quantiles: Optional[List[Tuple[float, float]]] = None  # For SUMMARY

class RouterMetrics:
    """
    Collects and reports metrics for the ExpertRouter system.
    
    This class provides a unified interface for collecting metrics that works
    with multiple backends (Prometheus, logging, etc.).
    """
    
    # Default metric definitions
    METRIC_DEFINITIONS = [
        # Router metrics
        MetricDefinition(
            name="router_requests_total",
            metric_type=MetricType.COUNTER,
            description="Total number of routing requests",
            labels=["strategy"]
        ),
        MetricDefinition(
            name="router_request_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Duration of routing requests in seconds",
            labels=["strategy"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        ),
        MetricDefinition(
            name="router_errors_total",
            metric_type=MetricType.COUNTER,
            description="Total number of routing errors",
            labels=["strategy", "error_type"]
        ),
        
        # Expert metrics
        MetricDefinition(
            name="expert_requests_total",
            metric_type=MetricType.COUNTER,
            description="Total number of expert requests",
            labels=["expert_type", "strategy"]
        ),
        MetricDefinition(
            name="expert_request_duration_seconds",
            metric_type=MetricType.SUMMARY,
            description="Duration of expert requests in seconds",
            labels=["expert_type", "success"],
            quantiles=[(0.5, 0.05), (0.9, 0.01), (0.99, 0.001)]
        ),
        MetricDefinition(
            name="expert_errors_total",
            metric_type=MetricType.COUNTER,
            description="Total number of expert errors",
            labels=["expert_type", "error_type"]
        ),
        MetricDefinition(
            name="cache_hits_total",
            metric_type=MetricType.COUNTER,
            description="Total number of cache hits",
            labels=["cache_type"]
        ),
        MetricDefinition(
            name="cache_misses_total",
            metric_type=MetricType.COUNTER,
            description="Total number of cache misses",
            labels=["cache_type"]
        ),
        MetricDefinition(
            name="cache_operations_total",
            metric_type=MetricType.COUNTER,
            description="Total cache operations",
            labels=["operation", "cache_type"]
        ),
        MetricDefinition(
            name="cache_size_bytes",
            metric_type=MetricType.GAUGE,
            description="Current size of the cache in bytes",
            labels=["cache_type"]
        ),
        MetricDefinition(
            name="cache_hit_ratio",
            metric_type=MetricType.GAUGE,
            description="Cache hit ratio (hits/requests)",
            labels=["cache_type"]
        ),
        MetricDefinition(
            name="enhanced_cache_operations_total",
            metric_type=MetricType.COUNTER,
            description="Total operations on the enhanced cache",
            labels=["operation", "status"]
        ),
        MetricDefinition(
            name="enhanced_cache_size_bytes",
            metric_type=MetricType.GAUGE,
            description="Current size of the enhanced cache in bytes"
        ),
        MetricDefinition(
            name="adaptive_learning_updates_total",
            metric_type=MetricType.COUNTER,
            description="Total number of adaptive learning model updates"
        ),
        MetricDefinition(
            name="adaptive_learning_accuracy",
            metric_type=MetricType.GAUGE,
            description="Current accuracy of the adaptive learning model"
        ),
        MetricDefinition(
            name="adaptive_learning_feedback_total",
            metric_type=MetricType.COUNTER,
            description="Total feedback received for adaptive learning",
            labels=["expert", "success"]
        ),
        MetricDefinition(
            name="adaptive_learning_exploration_rate",
            metric_type=MetricType.GAUGE,
            description="Current exploration rate for adaptive learning"
        ),
        MetricDefinition(
            name="adaptive_learning_model_update_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="Duration of adaptive learning model updates in seconds",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        ),
        MetricDefinition(
            name="active_strategies",
            metric_type=MetricType.GAUGE,
            description="Number of active routing strategies",
            labels=[]
        )
    ]
    
    def __init__(
        self,
        enable_prometheus: bool = True,
        prometheus_port: int = 8000,
        namespace: str = "expert_router",
        **kwargs
    ):
        """
        Initialize the metrics collector.
        
        Args:
            enable_prometheus: Whether to enable Prometheus metrics
            prometheus_port: Port to expose Prometheus metrics on
            namespace: Namespace for Prometheus metrics
            **kwargs: Additional arguments
        """
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port
        self.namespace = namespace
        self.metrics: Dict[str, Any] = {}
        self._initialized = False
        self._start_time = time.monotonic()
        
        # Initialize metrics
        self._init_metrics()
        
        # Start Prometheus server if enabled
        if self.enable_prometheus:
            try:
                start_http_server(self.prometheus_port)
                logger.info(f"Started Prometheus metrics server on port {self.prometheus_port}")
            except Exception as e:
                logger.error(f"Failed to start Prometheus server: {str(e)}")
                self.enable_prometheus = False
    
    def _init_metrics(self) -> None:
        """Initialize all metrics."""
        if self._initialized:
            return
        
        for definition in self.METRIC_DEFINITIONS:
            metric_name = f"{self.namespace}_{definition.name}"

            # Unregister metric if it already exists to prevent errors on re-initialization
            if self.enable_prometheus and metric_name in REGISTRY._names_to_collectors:
                REGISTRY.unregister(REGISTRY._names_to_collectors[metric_name])

            if self.enable_prometheus:
                # Initialize Prometheus metrics
                if definition.metric_type == MetricType.COUNTER:
                    self.metrics[metric_name] = Counter(
                        metric_name,
                        definition.description,
                        labelnames=definition.labels
                    )
                elif definition.metric_type == MetricType.GAUGE:
                    self.metrics[metric_name] = Gauge(
                        metric_name,
                        definition.description,
                        labelnames=definition.labels
                    )
                elif definition.metric_type == MetricType.HISTOGRAM:
                    self.metrics[metric_name] = Histogram(
                        metric_name,
                        definition.description,
                        labelnames=definition.labels,
                        buckets=definition.buckets or Histogram.DEFAULT_BUCKETS
                    )
                elif definition.metric_type == MetricType.SUMMARY:
                    self.metrics[metric_name] = Summary(
                        metric_name,
                        definition.description,
                        labelnames=definition.labels
                    )
            else:
                # Initialize simple in-memory metrics
                self.metrics[metric_name] = {
                    "type": definition.metric_type,
                    "description": definition.description,
                    "labels": definition.labels,
                    "data": {}
                }
        
        # Add uptime gauge
        if self.enable_prometheus:
            self.uptime_gauge = Gauge(
                f"{self.namespace}_uptime_seconds",
                "Uptime of the router in seconds"
            )
        
        self._initialized = True
    
    def _get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> Any:
        """
        Get a metric by name with optional labels.
        
        Args:
            name: Name of the metric (without namespace prefix)
            labels: Optional labels to filter by
            
        Returns:
            The metric value or None if not found
        """
        full_name = f"{self.namespace}_{name}"
        
        if full_name not in self.metrics:
            logger.warning(f"Unknown metric: {full_name}")
            return None
        
        metric = self.metrics[full_name]
        
        if self.enable_prometheus:
            # For Prometheus metrics, we can't directly get the value
            # without implementing a custom collector, so we return the metric object
            return metric.labels(**labels) if labels else metric
        else:
            # For in-memory metrics, return the value
            if not labels:
                return metric["data"].get("", {})
                
            # Build a key from the labels
            label_key = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return metric["data"].get(label_key, 0.0)
    
    def _update_metric(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        operation: str = "inc"
    ) -> None:
        """
        Update a metric.
        
        Args:
            name: Name of the metric (without namespace prefix)
            value: Value to add/set
            labels: Optional labels
            operation: Operation to perform ('inc', 'set', 'observe')
        """
        full_name = f"{self.namespace}_{name}"
        
        if full_name not in self.metrics:
            logger.warning(f"Unknown metric: {full_name}")
            return
        
        metric = self.metrics[full_name]
        labels = labels or {}
        
        if self.enable_prometheus:
            # Update Prometheus metric
            metric = metric.labels(**labels)
            
            if operation == "inc":
                metric.inc(value)
            elif operation == "dec":
                metric.dec(value)
            elif operation == "set":
                metric.set(value)
            elif operation == "observe":
                metric.observe(value)
            else:
                logger.warning(f"Unknown operation: {operation}")
        else:
            # Update in-memory metric
            label_key = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            
            if operation == "inc":
                metric["data"][label_key] = metric["data"].get(label_key, 0) + value
            elif operation == "dec":
                metric["data"][label_key] = metric["data"].get(label_key, 0) - value
            elif operation in ("set", "observe"):
                metric["data"][label_key] = value
            else:
                logger.warning(f"Unknown operation: {operation}")
    
    # High-level metric methods
    
    def record_router_request(self, strategy: str, duration: float) -> None:
        """Record a router request."""
        self._update_metric(
            "router_requests_total",
            labels={"strategy": strategy}
        )
        self._update_metric(
            "router_request_duration_seconds",
            value=duration,
            labels={"strategy": strategy},
            operation="observe"
        )
    
    def record_router_error(self, strategy: str, error_type: str) -> None:
        """Record a router error."""
        self._update_metric(
            "router_errors_total",
            labels={
                "strategy": strategy,
                "error_type": error_type
            }
        )
    
    def record_expert_request(
        self,
        expert_type: str,
        strategy: str,
        duration: float,
        success: bool = True
    ) -> None:
        """Record an expert request."""
        self._update_metric(
            "expert_requests_total",
            labels={
                "expert_type": expert_type,
                "strategy": strategy
            }
        )
        self._update_metric(
            "expert_request_duration_seconds",
            value=duration,
            labels={
                "expert_type": expert_type,
                "success": str(success).lower()
            },
            operation="observe"
        )
    
    def record_expert_error(self, expert_type: str, error_type: str) -> None:
        """Record an expert error."""
        self._update_metric(
            "expert_errors_total",
            labels={
                "expert_type": expert_type,
                "error_type": error_type
            }
        )
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Record a cache hit."""
        self._update_metric(
            "cache_hits_total",
            labels={"cache_type": cache_type}
        )
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Record a cache miss."""
        self._update_metric(
            "cache_misses_total",
            labels={"cache_type": cache_type}
        )
    
    def record_cache_operation(self, operation: str, cache_type: str = "default") -> None:
        """Record a cache operation.
        
        Args:
            operation: Type of operation (get, set, delete, etc.)
            cache_type: Type of cache (embedding, response, etc.)
        """
        self._update_metric("cache_operations_total", labels={"operation": operation, "cache_type": cache_type})
    
    def record_enhanced_cache_operation(self, operation: str, status: str = "success") -> None:
        """Record an enhanced cache operation.
        
        Args:
            operation: Type of operation (get_embedding, set_embedding, etc.)
            status: Status of the operation (success, error, miss, etc.)
        """
        self._update_metric(
            "enhanced_cache_operations_total", 
            labels={"operation": operation, "status": status}
        )
    
    def update_enhanced_cache_size(self, size_bytes: int) -> None:
        """Update the enhanced cache size metric.
        
        Args:
            size_bytes: Current size of the enhanced cache in bytes
        """
        self._update_metric("enhanced_cache_size_bytes", value=size_bytes)
        
    def record_adaptive_learning_update(self, accuracy: Optional[float] = None) -> None:
        """Record an adaptive learning model update.
        
        Args:
            accuracy: Optional accuracy metric from the update
        """
        self._update_metric("adaptive_learning_updates_total")
        if accuracy is not None:
            self._update_metric("adaptive_learning_accuracy", value=accuracy)
    
    def record_adaptive_learning_feedback(self, expert: str, success: bool) -> None:
        """Record feedback for adaptive learning.
        
        Args:
            expert: The expert that was used
            success: Whether the expert successfully handled the query
        """
        self._update_metric(
            "adaptive_learning_feedback_total",
            labels={"expert": expert, "success": str(success).lower()}
        )
    
    def update_exploration_rate(self, rate: float) -> None:
        """Update the exploration rate metric.
        
        Args:
            rate: Current exploration rate (0.0 to 1.0)
        """
        self._update_metric("adaptive_learning_exploration_rate", value=rate)
    
    def record_model_update_duration(self, duration: float) -> None:
        """Record the duration of a model update.
        
        Args:
            duration: Duration in seconds
        """
        self._update_metric("adaptive_learning_model_update_duration_seconds", value=duration)
    
    def set_cache_size(self, cache_type: str, size: int) -> None:
        """Set the cache size."""
        self._update_metric(
            "cache_size_bytes",
            value=size,
            labels={"cache_type": cache_type},
            operation="set"
        )
    
    def set_active_strategies(self, count: int) -> None:
        """Set the number of active strategies."""
        self._update_metric(
            "active_strategies",
            value=count,
            operation="set"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.
        
        Returns:
            Dictionary containing all metrics
        """
        result = {}
        
        if self.enable_prometheus and PROMETHEUS_AVAILABLE:
            # Get metrics in Prometheus text format
            from prometheus_client import generate_latest
            from prometheus_client.openmetrics.exposition import generate_latest as generate_latest_om
            
            try:
                result["prometheus"] = generate_latest(REGISTRY).decode("utf-8")
                result["openmetrics"] = generate_latest_om(REGISTRY).decode("utf-8")
            except Exception as e:
                logger.error(f"Error generating Prometheus metrics: {str(e)}")
        else:
            # Get in-memory metrics
            result = {}
            for name, metric in self.metrics.items():
                if isinstance(metric, dict):  # In-memory metric
                    result[name] = {
                        "type": metric["type"].value,
                        "description": metric["description"],
                        "data": metric["data"]
                    }
        
        # Add uptime
        result["uptime_seconds"] = time.monotonic() - self._start_time
        
        return result
    
    def get_metric_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """
        Get the current value of a metric.
        
        Args:
            name: Name of the metric (without namespace prefix)
            labels: Optional labels to filter by
            
        Returns:
            The metric value or None if not found
        """
        return self._get_metric(name, labels)
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {}
        
        for name, metric in self.metrics.items():
            if self.enable_prometheus and not isinstance(metric, dict):
                # For Prometheus metrics, we can't easily get the current value
                # without implementing a custom collector
                summary[name] = {"type": "prometheus_metric"}
            else:
                # For in-memory metrics
                summary[name] = {
                    "type": metric["type"].value,
                    "description": metric["description"],
                    "count": len(metric["data"])
                }
        
        summary["uptime_seconds"] = time.monotonic() - self._start_time
        return summary
