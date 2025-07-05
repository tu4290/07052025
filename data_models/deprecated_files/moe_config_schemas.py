"""MOE (Mixture of Experts) configuration schemas.

This module defines the core configuration models for the MOE system,
using Pydantic for validation and type safety.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from .moe_detailed_config_schemas import (
    CustomLoadBalancingFactors,
    CustomRequestParameters,
    CustomResourceMetrics,
    CustomTimingMeasurements,
    CustomToolData,
    CustomIntelligenceData,
    SectorPerformanceData,
    CustomMarketContext,
    ToolSpecificMetrics,
    CustomDebugInformation,
    CustomRiskData,
    StressTestResults,
    RiskCategoryData,
    CustomAnalysisMetadata,
    CustomRecommendationAttributes
)

# ===== REQUEST AND CONTEXT MODELS =====

class RequestContext(BaseModel):
    """Context information for MOE requests."""
    request_id: str = Field(..., description="Unique request identifier")
    user_id: Optional[str] = Field(None, description="User making the request")
    session_id: Optional[str] = Field(None, description="Session identifier")
    priority: Optional[str] = Field(default=None, description="Request priority level") # "normal" is placeholder
    timeout_ms: int = Field(default=30000, description="Request timeout in milliseconds")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    source_system: Optional[str] = Field(None, description="System originating the request")
    request_type: str = Field(..., description="Type of request being made")
    market_session: Optional[str] = Field(None, description="Market session context")
    custom_parameters: Optional[CustomRequestParameters] = Field(None, description="Custom request parameters")
    model_config = ConfigDict(extra='forbid')

class LoadBalancingFactors(BaseModel):
    """Load balancing considerations for expert selection."""
    cpu_utilization: Optional[float] = Field(default=None, description="Current CPU utilization", ge=0.0, le=100.0)
    memory_utilization: Optional[float] = Field(default=None, description="Current memory utilization", ge=0.0, le=100.0)
    active_requests: Optional[int] = Field(default=None, description="Number of active requests", ge=0)
    queue_length: Optional[int] = Field(default=None, description="Request queue length", ge=0)
    response_time_avg: Optional[float] = Field(default=None, description="Average response time in ms", ge=0.0)
    error_rate: Optional[float] = Field(default=None, description="Current error rate", ge=0.0, le=100.0)
    throughput: Optional[float] = Field(default=None, description="Requests per second", ge=0.0)
    health_score: Optional[float] = Field(default=None, description="Overall health score", ge=0.0, le=1.0) # Default 1.0 is unsafe
    custom_factors: Optional[CustomLoadBalancingFactors] = Field(None, description="Custom load balancing factors")
    model_config = ConfigDict(extra='forbid')

class ResourceUtilization(BaseModel):
    """Resource utilization metrics."""
    cpu_percent: Optional[float] = Field(default=None, description="CPU usage percentage", ge=0.0, le=100.0)
    memory_mb: Optional[float] = Field(default=None, description="Memory usage in MB", ge=0.0)
    gpu_percent: Optional[float] = Field(default=None, description="GPU usage percentage", ge=0.0, le=100.0)
    disk_io_mb: Optional[float] = Field(default=None, description="Disk I/O in MB", ge=0.0)
    network_io_mb: Optional[float] = Field(default=None, description="Network I/O in MB", ge=0.0)
    thread_count: Optional[int] = Field(default=None, description="Number of active threads", ge=0)
    connection_count: Optional[int] = Field(default=None, description="Number of active connections", ge=0)
    cache_hit_rate: Optional[float] = Field(default=None, description="Cache hit rate percentage", ge=0.0, le=100.0)
    custom_metrics: Optional[CustomResourceMetrics] = Field(None, description="Custom resource metrics")
    model_config = ConfigDict(extra='forbid')

class ResourceMetrics(BaseModel):
    """Resource metrics for system monitoring."""
    thread_count: Optional[float] = Field(None, description="Active thread count")
    file_descriptors: Optional[float] = Field(None, description="Open file descriptors")
    swap_usage: Optional[float] = Field(None, description="Swap usage in MB")
    load_average: Optional[float] = Field(None, description="System load average")
    temperature: Optional[float] = Field(None, description="System temperature")
    custom_metrics: Optional[CustomResourceMetrics] = Field(None, description="Custom resource metrics")
    model_config = ConfigDict(extra='forbid')

# ===== PERFORMANCE AND DEBUG MODELS =====

class PerformanceBreakdown(BaseModel):
    """Performance breakdown by component."""
    routing_time_ms: Optional[float] = Field(default=None, description="Time spent on routing", ge=0.0)
    expert_execution_ms: Optional[float] = Field(default=None, description="Time spent on expert execution", ge=0.0)
    consensus_time_ms: Optional[float] = Field(default=None, description="Time spent reaching consensus", ge=0.0)
    data_serialization_ms: Optional[float] = Field(default=None, description="Time spent on data serialization", ge=0.0)
    network_latency_ms: Optional[float] = Field(default=None, description="Network latency", ge=0.0)
    queue_wait_ms: Optional[float] = Field(default=None, description="Time spent waiting in queue", ge=0.0)
    preprocessing_ms: Optional[float] = Field(default=None, description="Time spent on preprocessing", ge=0.0)
    postprocessing_ms: Optional[float] = Field(default=None, description="Time spent on postprocessing", ge=0.0)
    custom_timings: Optional[CustomTimingMeasurements] = Field(None, description="Custom timing measurements")
    model_config = ConfigDict(extra='forbid')

class DebugInfo(BaseModel):
    """Debug information for MOE operations."""
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")
    span_id: Optional[str] = Field(None, description="Span ID for this operation")
    log_level: str = Field(default="INFO", description="Log level for this operation") # INFO is acceptable default
    execution_path: List[str] = Field(default_factory=list, description="Execution path taken")
    decision_points: List[str] = Field(default_factory=list, description="Key decision points")
    error_details: Optional[str] = Field(None, description="Detailed error information")
    stack_trace: Optional[str] = Field(None, description="Stack trace if error occurred")
    expert_selection_reason: Optional[str] = Field(None, description="Reason for expert selection")
    consensus_details: Optional[str] = Field(None, description="Details about consensus process")
    custom_debug_data: Optional[CustomDebugInformation] = Field(None, description="Custom debug information")
    model_config = ConfigDict(extra='forbid')

# ===== RESULT AND RESPONSE MODELS =====

class ToolResultData(BaseModel):
    """Tool execution result data."""
    output: Optional[str] = Field(None, description="Tool output")
    return_code: int = Field(default=0, description="Tool return code") # 0 is acceptable default
    stdout: Optional[str] = Field(None, description="Standard output")
    stderr: Optional[str] = Field(None, description="Standard error")
    artifacts: List[str] = Field(default_factory=list, description="Generated artifacts")
    metrics: Optional[ToolSpecificMetrics] = Field(None, description="Tool-specific metrics")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated")
    custom_data: Optional[CustomToolData] = Field(None, description="Custom tool data")
    model_config = ConfigDict(extra='forbid')

class IntelligenceData(BaseModel):
    """Intelligence data from expert analysis."""
    confidence_score: Optional[float] = Field(default=None, description="Confidence in analysis", ge=0.0, le=1.0)
    data_quality: Optional[float] = Field(default=None, description="Quality of input data", ge=0.0, le=1.0)
    bias_score: Optional[float] = Field(default=None, description="Detected bias level", ge=0.0, le=1.0)
    uncertainty: Optional[float] = Field(default=None, description="Level of uncertainty", ge=0.0, le=1.0)
    validation_score: Optional[float] = Field(default=None, description="Cross-validation score", ge=0.0, le=1.0)
    custom_data: Optional[CustomIntelligenceData] = Field(None, description="Custom intelligence data")
    model_config = ConfigDict(extra='forbid')

class MarketData(BaseModel):
    """Market data and sector performance."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Data timestamp")
    market_cap: Optional[float] = Field(default=None, description="Market capitalization", ge=0.0)
    volume: Optional[int] = Field(default=None, description="Trading volume", ge=0)
    volatility: Optional[float] = Field(default=None, description="Market volatility", ge=0.0)
    trend: Optional[str] = Field(default=None, description="Market trend") # "neutral" is placeholder
    sector_data: Optional[SectorPerformanceData] = Field(None, description="Sector performance data")
    market_context: Optional[CustomMarketContext] = Field(None, description="Custom market context")
    model_config = ConfigDict(extra='forbid')

class RecommendationData(BaseModel):
    """Trading recommendation data."""
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="Buy/Sell/Hold recommendation")
    confidence: Optional[float] = Field(default=None, description="Confidence level", ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Recommendation timestamp")
    expiration: Optional[datetime] = Field(None, description="Recommendation expiration")
    risk_level: Optional[str] = Field(default=None, description="Risk assessment") # "medium" is placeholder
    attributes: Optional[CustomRecommendationAttributes] = Field(None, description="Custom recommendation attributes")
    model_config = ConfigDict(extra='forbid')

class DebugData(BaseModel):
    """Debug information for system monitoring."""
    log_level: str = Field(default="info", description="Debug log level") # "info" is acceptable default
    trace_id: Optional[str] = Field(None, description="Distributed tracing ID")
    stack_trace: Optional[str] = Field(None, description="Error stack trace")
    variables: Optional[Dict[str, Any]] = Field(default=None, description="Debug variables")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Debug context")
    debug_info: Optional[CustomDebugInformation] = Field(None, description="Custom debug information")
    model_config = ConfigDict(extra='forbid')

class MarketContext(BaseModel):
    """Market context during analysis."""
    market_session: str = Field(..., description="Current market session")
    trading_day: datetime = Field(..., description="Trading day")
    market_open: bool = Field(..., description="Whether market is open")
    volatility_regime: Optional[str] = Field(None, description="Current volatility regime")
    sector_performance: Optional[SectorPerformanceData] = Field(None, description="Sector performance data")
    market_sentiment: Optional[str] = Field(None, description="Overall market sentiment")
    economic_events: List[str] = Field(default_factory=list, description="Relevant economic events")
    news_sentiment: Optional[float] = Field(None, description="News sentiment score", ge=-1.0, le=1.0)
    custom_context: Optional[CustomMarketContext] = Field(None, description="Custom market context")
    model_config = ConfigDict(extra='forbid')

class AnalysisRecommendation(BaseModel):
    """Analysis recommendation."""
    recommendation_id: str = Field(..., description="Unique recommendation ID")
    recommendation_type: str = Field(..., description="Type of recommendation")
    action: str = Field(..., description="Recommended action")
    confidence: float = Field(..., description="Confidence in recommendation", ge=0.0, le=1.0)
    priority: Optional[str] = Field(default=None, description="Recommendation priority") # "medium" is placeholder
    rationale: str = Field(..., description="Rationale for recommendation")
    expected_outcome: Optional[str] = Field(None, description="Expected outcome")
    risk_level: Optional[str] = Field(default=None, description="Risk level") # "medium" is placeholder
    time_horizon: Optional[str] = Field(None, description="Time horizon for recommendation")
    custom_attributes: Optional[CustomRecommendationAttributes] = Field(None, description="Custom recommendation attributes")
    model_config = ConfigDict(extra='forbid')

class RiskAssessment(BaseModel):
    """Risk assessment data."""
    overall_risk_score: float = Field(..., description="Overall risk score", ge=0.0, le=1.0)
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    risk_categories: Optional[RiskCategoryData] = Field(None, description="Risk by category")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Risk mitigation strategies")
    probability_of_loss: Optional[float] = Field(default=None, description="Probability of loss", ge=0.0, le=1.0)
    potential_impact: Optional[str] = Field(default=None, description="Potential impact level") # "medium" is placeholder
    risk_tolerance: Optional[str] = Field(None, description="Risk tolerance level")
    stress_test_results: Optional[StressTestResults] = Field(None, description="Stress test results")
    custom_risk_data: Optional[CustomRiskData] = Field(None, description="Custom risk assessment data")
    model_config = ConfigDict(extra='forbid')

class AnalysisMetadata(BaseModel):
    """Analysis metadata."""
    analysis_version: str = Field(default="2.5", description="Analysis version") # "2.5" is acceptable default
    model_version: Optional[str] = Field(None, description="Model version used")
    data_version: Optional[str] = Field(None, description="Data version")
    environment: str = Field(default="production", description="Environment where analysis ran") # "production" is acceptable default
    analyst_id: Optional[str] = Field(None, description="Analyst or system ID")
    review_status: Optional[str] = Field(default=None, description="Review status") # "pending" is placeholder
    tags: List[str] = Field(default_factory=list, description="Analysis tags")
    categories: List[str] = Field(default_factory=list, description="Analysis categories")
    compliance_flags: List[str] = Field(default_factory=list, description="Compliance flags")
    custom_metadata: Optional[CustomAnalysisMetadata] = Field(None, description="Custom metadata")
    model_config = ConfigDict(extra='forbid')

class ExpertRequestData(BaseModel):
    """Data for expert request processing."""
    request_id: str = Field(..., description="Unique request identifier")
    expert_id: str = Field(..., description="Expert identifier")
    priority: int = Field(default=1, description="Request priority", ge=1, le=10) # 1 is acceptable default
    timeout_ms: int = Field(default=5000, description="Request timeout in milliseconds", ge=0) # 5000 is acceptable default
    retry_count: int = Field(default=0, description="Number of retries", ge=0) # 0 is acceptable default
    custom_params: Optional[CustomRequestParameters] = Field(None, description="Custom request parameters")
    resource_metrics: Optional[CustomResourceMetrics] = Field(None, description="Resource utilization metrics")
    timing_data: Optional[CustomTimingMeasurements] = Field(None, description="Timing measurements")
    tool_metrics: Optional[ToolSpecificMetrics] = Field(None, description="Tool-specific metrics")
    tool_data: Optional[CustomToolData] = Field(None, description="Tool execution data")
    intelligence_data: Optional[CustomIntelligenceData] = Field(None, description="Intelligence analysis data")
    sector_data: Optional[SectorPerformanceData] = Field(None, description="Sector performance data")
    market_context: Optional[CustomMarketContext] = Field(None, description="Market context data")
    debug_info: Optional[CustomDebugInformation] = Field(None, description="Debug information")
    risk_data: Optional[CustomRiskData] = Field(None, description="Risk assessment data")
    stress_test_results: Optional[StressTestResults] = Field(None, description="Stress test results")
    risk_category_data: Optional[RiskCategoryData] = Field(None, description="Risk category data")
    analysis_metadata: Optional[CustomAnalysisMetadata] = Field(None, description="Analysis metadata")
    recommendation_attributes: Optional[CustomRecommendationAttributes] = Field(None, description="Recommendation attributes")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Request creation timestamp")
    model_config = ConfigDict(extra='forbid')

# Export all models
__all__ = [
    'RequestContext', 'LoadBalancingFactors', 'ResourceUtilization',
    'PerformanceBreakdown', 'DebugInfo', 'ToolResultData',
    'IntelligenceData', 'MarketData', 'RecommendationData',
    'DebugData', 'MarketContext', 'AnalysisRecommendation',
    'RiskAssessment', 'AnalysisMetadata', 'ExpertRequestData'
]