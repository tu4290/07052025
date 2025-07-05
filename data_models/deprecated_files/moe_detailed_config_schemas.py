"""Detailed configuration schemas for MOE (Mixture of Experts) system components.

This module defines Pydantic models to replace Dict[str, Any] patterns
in MOE configuration schemas, providing better type safety and validation.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class CustomRequestParameters(BaseModel):
    """Custom request parameters for MOE requests."""
    timeout_override: Optional[str] = Field(None, description="Timeout override setting")
    priority_boost: Optional[str] = Field(None, description="Priority boost setting")
    routing_hint: Optional[str] = Field(None, description="Routing hint for expert selection")
    cache_policy: Optional[str] = Field(None, description="Cache policy override")
    debug_mode: Optional[str] = Field(None, description="Debug mode setting")
    additional_params: Optional[Dict[str, str]] = Field(default=None, description="Additional custom parameters")
    model_config = ConfigDict(extra='forbid')


class CustomLoadBalancingFactors(BaseModel):
    """Custom load balancing factors for expert selection."""
    expert_affinity: Optional[float] = Field(None, description="Expert affinity score")
    geographic_preference: Optional[float] = Field(None, description="Geographic preference weight")
    cost_factor: Optional[float] = Field(None, description="Cost consideration factor")
    reliability_score: Optional[float] = Field(None, description="Reliability score")
    specialization_match: Optional[float] = Field(None, description="Specialization match score")
    additional_factors: Optional[Dict[str, float]] = Field(default=None, description="Additional load balancing factors")
    model_config = ConfigDict(extra='forbid')


class CustomResourceMetrics(BaseModel):
    """Custom resource utilization metrics."""
    thread_count: Optional[float] = Field(None, description="Active thread count")
    file_descriptors: Optional[float] = Field(None, description="Open file descriptors")
    swap_usage: Optional[float] = Field(None, description="Swap usage in MB")
    load_average: Optional[float] = Field(None, description="System load average")
    temperature: Optional[float] = Field(None, description="System temperature")
    additional_metrics: Optional[Dict[str, float]] = Field(default=None, description="Additional resource metrics")
    model_config = ConfigDict(extra='forbid')


class CustomTimingMeasurements(BaseModel):
    """Custom timing measurements for performance breakdown."""
    authentication_ms: Optional[float] = Field(None, description="Authentication time")
    authorization_ms: Optional[float] = Field(None, description="Authorization time")
    validation_ms: Optional[float] = Field(None, description="Input validation time")
    transformation_ms: Optional[float] = Field(None, description="Data transformation time")
    caching_ms: Optional[float] = Field(None, description="Caching operations time")
    additional_timings: Optional[Dict[str, float]] = Field(default=None, description="Additional timing measurements")
    model_config = ConfigDict(extra='forbid')


class CustomDebugInformation(BaseModel):
    """Custom debug information for request processing."""
    request_fingerprint: Optional[str] = Field(None, description="Request fingerprint")
    expert_pool_state: Optional[str] = Field(None, description="Expert pool state")
    routing_algorithm: Optional[str] = Field(None, description="Routing algorithm used")
    fallback_triggered: Optional[str] = Field(None, description="Fallback trigger status")
    cache_status: Optional[str] = Field(None, description="Cache status")
    additional_debug: Optional[Dict[str, str]] = Field(default=None, description="Additional debug information")
    model_config = ConfigDict(extra='forbid')


class ToolSpecificMetrics(BaseModel):
    """Tool-specific metrics for tool execution results."""
    execution_time_ms: Optional[float] = Field(None, description="Tool execution time")
    memory_peak_mb: Optional[float] = Field(None, description="Peak memory usage")
    cpu_time_ms: Optional[float] = Field(None, description="CPU time consumed")
    io_operations: Optional[float] = Field(None, description="Number of I/O operations")
    cache_hits: Optional[float] = Field(None, description="Number of cache hits")
    additional_metrics: Optional[Dict[str, float]] = Field(default=None, description="Additional tool metrics")
    model_config = ConfigDict(extra='forbid')


class CustomToolData(BaseModel):
    """Custom tool execution data."""
    tool_version: Optional[str] = Field(None, description="Tool version used")
    configuration: Optional[str] = Field(None, description="Tool configuration")
    environment_vars: Optional[str] = Field(None, description="Environment variables")
    working_directory: Optional[str] = Field(None, description="Working directory")
    user_context: Optional[str] = Field(None, description="User context")
    additional_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional custom tool data")
    model_config = ConfigDict(extra='forbid')


class CustomIntelligenceData(BaseModel):
    """Custom intelligence analysis data."""
    model_confidence: Optional[str] = Field(None, description="Model confidence details")
    data_quality_score: Optional[str] = Field(None, description="Data quality assessment")
    bias_indicators: Optional[str] = Field(None, description="Bias indicators found")
    uncertainty_measures: Optional[str] = Field(None, description="Uncertainty measurements")
    validation_results: Optional[str] = Field(None, description="Validation results")
    additional_intelligence: Optional[Dict[str, Any]] = Field(default=None, description="Additional intelligence data")
    model_config = ConfigDict(extra='forbid')


class SectorPerformanceData(BaseModel):
    """Sector performance data for market context."""
    technology: Optional[float] = Field(None, description="Technology sector performance")
    healthcare: Optional[float] = Field(None, description="Healthcare sector performance")
    financials: Optional[float] = Field(None, description="Financials sector performance")
    energy: Optional[float] = Field(None, description="Energy sector performance")
    utilities: Optional[float] = Field(None, description="Utilities sector performance")
    additional_sectors: Optional[Dict[str, float]] = Field(default=None, description="Additional sector performance")
    model_config = ConfigDict(extra='forbid')


class CustomMarketContext(BaseModel):
    """Custom market context data."""
    vix_level: Optional[str] = Field(None, description="VIX volatility level")
    yield_curve_shape: Optional[str] = Field(None, description="Yield curve shape")
    dollar_strength: Optional[str] = Field(None, description="Dollar strength indicator")
    commodity_trends: Optional[str] = Field(None, description="Commodity trends")
    geopolitical_events: Optional[str] = Field(None, description="Geopolitical events")
    additional_context: Optional[Dict[str, Any]] = Field(default=None, description="Additional market context")
    model_config = ConfigDict(extra='forbid')


class CustomRecommendationAttributes(BaseModel):
    """Custom recommendation attributes."""
    target_price: Optional[str] = Field(None, description="Target price for recommendation")
    stop_loss: Optional[str] = Field(None, description="Stop loss level")
    position_size: Optional[str] = Field(None, description="Recommended position size")
    holding_period: Optional[str] = Field(None, description="Recommended holding period")
    market_conditions: Optional[str] = Field(None, description="Required market conditions")
    additional_attributes: Optional[Dict[str, Any]] = Field(default=None, description="Additional recommendation attributes")
    model_config = ConfigDict(extra='forbid')


class RiskCategoryData(BaseModel):
    """Risk assessment data by category."""
    market_risk: Optional[float] = Field(None, description="Market risk score")
    credit_risk: Optional[float] = Field(None, description="Credit risk score")
    liquidity_risk: Optional[float] = Field(None, description="Liquidity risk score")
    operational_risk: Optional[float] = Field(None, description="Operational risk score")
    regulatory_risk: Optional[float] = Field(None, description="Regulatory risk score")
    additional_categories: Optional[Dict[str, float]] = Field(default=None, description="Additional risk categories")
    model_config = ConfigDict(extra='forbid')


class StressTestResults(BaseModel):
    """Stress test results data."""
    market_crash_scenario: Optional[float] = Field(None, description="Market crash scenario result")
    interest_rate_shock: Optional[float] = Field(None, description="Interest rate shock result")
    liquidity_crisis: Optional[float] = Field(None, description="Liquidity crisis result")
    volatility_spike: Optional[float] = Field(None, description="Volatility spike result")
    correlation_breakdown: Optional[float] = Field(None, description="Correlation breakdown result")
    additional_scenarios: Optional[Dict[str, float]] = Field(default=None, description="Additional stress test scenarios")
    model_config = ConfigDict(extra='forbid')


class CustomRiskData(BaseModel):
    """Custom risk assessment data."""
    risk_model_version: Optional[str] = Field(None, description="Risk model version used")
    calculation_method: Optional[str] = Field(None, description="Risk calculation method")
    time_horizon: Optional[str] = Field(None, description="Risk assessment time horizon")
    confidence_interval: Optional[str] = Field(None, description="Confidence interval used")
    benchmark_comparison: Optional[str] = Field(None, description="Benchmark comparison")
    additional_risk_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional risk data")
    model_config = ConfigDict(extra='forbid')


class CustomAnalysisMetadata(BaseModel):
    """Custom metadata for analysis results."""
    data_sources_count: Optional[int] = Field(None, description="Number of data sources")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in megabytes")
    algorithm_version: Optional[str] = Field(None, description="Algorithm version used")
    validation_score: Optional[float] = Field(None, description="Validation score")
    additional_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    model_config = ConfigDict(extra='forbid')