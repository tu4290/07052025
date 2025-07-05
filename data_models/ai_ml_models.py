"""
AI/ML Models for EOTS v2.5

Consolidated from: ai_adaptations.py, ai_predictions.py, moe_schemas_v2_5.py,
learning_schemas.py, hui_hui_schemas.py, huihui_models.py, performance_schemas.py
"""

# Standard library imports
from typing import Optional, Dict, Any, List, Literal, TYPE_CHECKING, Deque, Tuple, Union
from datetime import datetime, timedelta, timezone
from enum import Enum
from collections import deque, defaultdict

# Third-party imports
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator, FieldValidationInfo
from typing_extensions import Literal

# EOTS Pydantic Models
from .expert_analysis_schemas import (
    ExpertAnalysisRequest,
    ExpertAnalysisResponse,
    MarketRegimeAnalysisDetails,
    OptionsFlowAnalysisDetails,
    SentimentAnalysisDetails
)


# =============================================================================
# FROM ai_adaptations.py
# =============================================================================
"""
Pydantic models for AI model adaptations and performance.
"""

class AIAdaptationV2_5(BaseModel):
    """Tracks AI model adaptations and adjustments for market conditions."""
    id: Optional[int] = Field(None, description="Auto-generated database ID", ge=1)
    adaptation_type: Literal['signal_enhancement', 'threshold_adjustment', 'model_calibration', 'feature_engineering']
    adaptation_name: str = Field(..., min_length=3, max_length=100)
    adaptation_description: Optional[str] = Field(None, max_length=1000)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    adaptation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    implementation_status: Literal['PENDING', 'ACTIVE', 'INACTIVE', 'DEPRECATED', 'TESTING'] = "PENDING"
    market_context: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    parent_model_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AIAdaptationPerformanceV2_5(BaseModel):
    """Tracks performance metrics for AI model adaptations over time."""
    adaptation_id: int = Field(..., ge=1)
    symbol: str = Field(..., max_length=20)
    time_period_days: int = Field(..., ge=1)
    total_applications: int = Field(..., ge=0)
    successful_applications: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    avg_improvement: float
    adaptation_score: float = Field(..., ge=0.0, le=1.0)
    performance_trend: Literal['IMPROVING', 'STABLE', 'DECLINING', 'UNKNOWN']
    avg_processing_time_ms: float = Field(..., ge=0.0)
    market_conditions: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# FROM ai_predictions.py
# =============================================================================
"""
Pydantic models for AI-driven predictions and performance tracking.
"""

class AIPredictionV2_5(BaseModel):
    """Represents an AI-generated prediction for market analysis and trading decisions."""
    id: Optional[int] = Field(None, description="Auto-generated database ID", ge=1)
    symbol: str = Field(..., description="Trading symbol", max_length=20)
    prediction_type: Literal['price', 'direction', 'volatility', 'eots_direction', 'sentiment']
    prediction_value: Optional[float] = Field(None, description="Predicted numerical value")
    prediction_direction: Literal['UP', 'DOWN', 'NEUTRAL']
    confidence_score: float = Field(..., description="Model's confidence (0.0 to 1.0)", ge=0.0, le=1.0)
    time_horizon: str = Field(..., description="Time frame for the prediction (e.g., '1H', '1D')")
    prediction_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_timestamp: datetime = Field(..., description="When the prediction should be evaluated")
    actual_value: Optional[float] = Field(None, description="Actual observed value")
    actual_direction: Optional[Literal['UP', 'DOWN', 'NEUTRAL']] = Field(None, description="Actual observed direction")
    prediction_accurate: Optional[bool] = Field(None, description="Whether the prediction was accurate")
    accuracy_score: Optional[float] = Field(None, description="Quantitative accuracy score (0.0 to 1.0)", ge=0.0, le=1.0)
    model_version: str = Field(default="v2.5")
    model_name: Optional[str] = Field(None)
    market_context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AIPredictionPerformanceV2_5(BaseModel):
    """Tracks and analyzes the performance of AI predictions over time."""
    symbol: str = Field(..., max_length=20)
    model_name: str = Field(default="default")
    time_period_days: int = Field(..., ge=1, le=3650)
    total_predictions: int = Field(..., ge=0)
    correct_predictions: int = Field(..., ge=0)
    incorrect_predictions: int = Field(..., ge=0)
    pending_predictions: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    avg_accuracy_score: float = Field(..., ge=0.0, le=1.0)
    learning_score: float = Field(..., ge=0.0, le=1.0)
    performance_trend: Literal['IMPROVING', 'STABLE', 'DECLINING', 'UNKNOWN']
    precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AIPredictionRequestV2_5(BaseModel):
    """Request model for creating new AI predictions."""
    symbol: str = Field(..., max_length=20)
    prediction_type: Literal['price', 'direction', 'volatility', 'eots_direction', 'sentiment']
    prediction_value: Optional[float] = Field(None, ge=0.0)
    prediction_direction: Optional[Literal['UP', 'DOWN', 'NEUTRAL']]
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    model_name: Optional[str] = Field(None, max_length=100)
    time_horizon: str = Field(..., pattern=r'^\d+[mhdwMqy]$')
    target_timestamp: datetime
    market_context: Dict[str, Any] = Field(default_factory=dict)
    request_metadata: Dict[str, Any] = Field(default_factory=dict)

class AIPredictionSummaryV2_5(BaseModel):
    """Unified summary of AI predictions and their performance."""
    symbol: str = Field(..., description="Trading symbol")
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latest_prediction: Optional[AIPredictionV2_5] = Field(None, description="The most recent prediction for the symbol")
    overall_performance: Optional[AIPredictionPerformanceV2_5] = Field(None, description="Aggregated performance metrics")
    active_predictions_count: int = Field(..., description="Number of active predictions")
    pending_predictions_count: int = Field(..., description="Number of predictions awaiting outcome")
    prediction_quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score of predictions")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Average confidence of active predictions")
    optimization_recommendations: List[str] = Field(default_factory=list, description="Recommendations for model optimization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# =============================================================================
# FROM moe_schemas_v2_5.py
# =============================================================================
# Pydantic-first data models for the MOE system

class RequestContext(BaseModel):
    """Context information for MOE requests including priority, user info, and system state."""
    request_id: str = Field(..., description="Unique identifier for the request")
    user_id: Optional[str] = Field(None, description="User making the request")
    priority: int = Field(default=5, ge=1, le=10, description="Request priority (1=highest, 10=lowest)")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    session_id: Optional[str] = Field(None, description="Session identifier")
    request_type: str = Field(default="analysis", description="Type of request")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")

class LoadBalancingFactors(BaseModel):
    """Factors considered for load balancing across MOE experts."""
    cpu_load: float = Field(default=0.0, ge=0.0, le=100.0, description="Current CPU load percentage")
    memory_usage: float = Field(default=0.0, ge=0.0, le=100.0, description="Current memory usage percentage")
    active_requests: int = Field(default=0, ge=0, description="Number of active requests")
    queue_length: int = Field(default=0, ge=0, description="Length of request queue")
    response_time_avg: float = Field(default=0.0, ge=0.0, description="Average response time in seconds")
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Recent error rate")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

class ResourceUtilization(BaseModel):
    """Resource utilization metrics for system monitoring."""
    memory_usage_mb: float = Field(default=0.0, ge=0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="CPU usage percentage")
    disk_io_rate: float = Field(default=0.0, ge=0.0, description="Disk I/O rate in MB/s")
    network_io_rate: float = Field(default=0.0, ge=0.0, description="Network I/O rate in MB/s")
    gpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU usage percentage if available")
    timestamp: datetime = Field(default_factory=datetime.now, description="Measurement timestamp")

class PerformanceBreakdown(BaseModel):
    """Detailed performance breakdown for analysis and optimization."""
    total_time_ms: float = Field(default=0.0, ge=0.0, description="Total processing time in milliseconds")
    preprocessing_time_ms: float = Field(default=0.0, ge=0.0, description="Data preprocessing time")
    analysis_time_ms: float = Field(default=0.0, ge=0.0, description="Core analysis time")
    postprocessing_time_ms: float = Field(default=0.0, ge=0.0, description="Result postprocessing time")
    network_time_ms: float = Field(default=0.0, ge=0.0, description="Network communication time")
    queue_wait_time_ms: float = Field(default=0.0, ge=0.0, description="Time spent waiting in queue")
    bottleneck_component: Optional[str] = Field(None, description="Component causing performance bottleneck")

class DebugInfo(BaseModel):
    """Debug information for troubleshooting and development."""
    debug_level: str = Field(default="INFO", description="Debug level (DEBUG, INFO, WARN, ERROR)")
    execution_path: List[str] = Field(default_factory=list, description="Execution path through system components")
    intermediate_results: Dict[str, Any] = Field(default_factory=dict, description="Intermediate calculation results")
    error_details: Optional[str] = Field(None, description="Detailed error information if applicable")
    stack_trace: Optional[str] = Field(None, description="Stack trace for errors")
    system_state: Dict[str, Any] = Field(default_factory=dict, description="System state at time of execution")

class ToolResultData(BaseModel):
    """Result data from tool execution with comprehensive metadata."""
    result_type: str = Field(..., description="Type of result (analysis, prediction, recommendation, etc.)")
    data: Any = Field(..., description="Primary result data")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the result")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Quality assessment of the result")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used for the result")
    processing_notes: List[str] = Field(default_factory=list, description="Notes about processing steps")
    validation_status: str = Field(default="pending", description="Validation status of the result")

class IntelligenceData(BaseModel):
    """Intelligence analysis data with structured insights and recommendations."""
    primary_insights: List[str] = Field(default_factory=list, description="Primary intelligence insights")
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting evidence for insights")
    confidence_levels: Dict[str, float] = Field(default_factory=dict, description="Confidence levels for each insight")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    opportunities: List[str] = Field(default_factory=list, description="Identified opportunities")
    market_implications: List[str] = Field(default_factory=list, description="Market implications of the intelligence")
    time_sensitivity: str = Field(default="medium", description="Time sensitivity of the intelligence")

class MarketContext(BaseModel):
    """Comprehensive market context for analysis and decision making."""
    market_state: str = Field(..., description="Current market state (bull, bear, sideways, volatile)")
    volatility_regime: str = Field(default="normal", description="Current volatility regime")
    liquidity_conditions: str = Field(default="normal", description="Current liquidity conditions")
    sentiment_indicators: Dict[str, float] = Field(default_factory=dict, description="Various sentiment indicators")
    economic_indicators: Dict[str, float] = Field(default_factory=dict, description="Relevant economic indicators")
    technical_levels: Dict[str, float] = Field(default_factory=dict, description="Key technical levels")
    event_calendar: List[str] = Field(default_factory=list, description="Upcoming market events")

class AnalysisRecommendation(BaseModel):
    """Structured analysis recommendation with rationale and risk assessment."""
    recommendation_id: str = Field(..., description="Unique identifier for the recommendation")
    recommendation_type: str = Field(..., description="Type of recommendation (trade, position, strategy)")
    action: str = Field(..., description="Recommended action (buy, sell, hold, adjust)")
    rationale: str = Field(..., description="Detailed rationale for the recommendation")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the recommendation")
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    time_horizon: str = Field(..., description="Recommended time horizon")
    target_price: Optional[float] = Field(None, description="Target price if applicable")
    stop_loss: Optional[float] = Field(None, description="Stop loss level if applicable")

class RiskAssessment(BaseModel):
    """Comprehensive risk assessment for trading and investment decisions."""
    overall_risk_score: float = Field(..., ge=0.0, le=10.0, description="Overall risk score (0=low, 10=high)")
    market_risk: float = Field(default=5.0, ge=0.0, le=10.0, description="Market risk component")
    liquidity_risk: float = Field(default=5.0, ge=0.0, le=10.0, description="Liquidity risk component")
    volatility_risk: float = Field(default=5.0, ge=0.0, le=10.0, description="Volatility risk component")
    concentration_risk: float = Field(default=5.0, ge=0.0, le=10.0, description="Concentration risk component")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    mitigation_strategies: List[str] = Field(default_factory=list, description="Risk mitigation strategies")
    max_drawdown_estimate: Optional[float] = Field(None, description="Estimated maximum drawdown")

class AnalysisMetadata(BaseModel):
    """Metadata for analysis results including data quality and processing information."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    analysis_type: str = Field(..., description="Type of analysis performed")
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Quality score of input data")
    completeness_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Completeness of analysis")
    data_sources: List[str] = Field(default_factory=list, description="Data sources used")
    processing_steps: List[str] = Field(default_factory=list, description="Processing steps performed")
    validation_checks: Dict[str, bool] = Field(default_factory=dict, description="Validation checks performed")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    limitations: List[str] = Field(default_factory=list, description="Analysis limitations")

# ===== ENUMS =====

class ExpertStatus(str, Enum):
    """Status of an expert in the MOE system."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class RoutingStrategy(str, Enum):
    """Strategy for routing requests to experts."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_MATCHED = "capability_matched"

class ConsensusStrategy(str, Enum):
    """Strategy for reaching consensus among experts."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPERT_RANKING = "expert_ranking"
    UNANIMOUS = "unanimous"

class AgreementLevel(str, Enum):
    """Level of agreement among experts."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CONFLICTING = "conflicting"
    INSUFFICIENT_DATA = "insufficient_data"

class HealthStatus(str, Enum):
    """Health status of the MOE system."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

# ===== CORE MOE SCHEMAS =====

class MOEExpertRegistryV2_5(BaseModel):
    """Registry entry for an expert in the MOE system."""
    
    # Expert identification
    expert_id: str = Field(..., description="Unique identifier for the expert")
    expert_name: str = Field(..., description="Human-readable name for the expert")
    expert_type: str = Field(..., description="Type/category of the expert")
    version: str = Field(default="2.5", description="Expert version")
    
    # Expert configuration
    capabilities: List[str] = Field(default_factory=list, description="List of expert capabilities")
    specializations: List[str] = Field(default_factory=list, description="Expert specializations")
    supported_tasks: List[str] = Field(default_factory=list, description="Tasks this expert can handle")
    
    # Performance metrics
    accuracy_score: float = Field(default=0.0, description="Expert accuracy score", ge=0.0, le=1.0)
    confidence_score: float = Field(default=0.0, description="Expert confidence score", ge=0.0, le=1.0)
    response_time_ms: float = Field(default=0.0, description="Average response time in milliseconds", ge=0.0)
    success_rate: float = Field(default=0.0, description="Success rate percentage", ge=0.0, le=100.0)
    
    # Resource requirements
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB", ge=0.0)
    cpu_usage_percent: float = Field(default=0.0, description="CPU usage percentage", ge=0.0, le=100.0)
    gpu_required: bool = Field(default=False, description="Whether GPU is required")
    
    # Status and health
    status: ExpertStatus = Field(default=ExpertStatus.INACTIVE, description="Current expert status")
    health_score: float = Field(default=1.0, description="Health score", ge=0.0, le=1.0)
    last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Expert tags for categorization")
    
    model_config = ConfigDict(extra='forbid')

class MOEGatingNetworkV2_5(BaseModel):
    """Gating network for routing decisions in the MOE system."""
    
    # Routing decision
    selected_experts: List[str] = Field(..., description="List of selected expert IDs")
    routing_strategy: RoutingStrategy = Field(..., description="Strategy used for routing")
    routing_confidence: float = Field(..., description="Confidence in routing decision", ge=0.0, le=1.0)
    
    # Expert weights
    expert_weights: Dict[str, float] = Field(default_factory=dict, description="Weights assigned to each expert")
    capability_scores: Dict[str, float] = Field(default_factory=dict, description="Capability scores for each expert")
    
    # Decision metadata
    decision_timestamp: datetime = Field(default_factory=datetime.now, description="When routing decision was made")
    processing_time_ms: float = Field(default=0.0, description="Time taken for routing decision", ge=0.0)
    
    # Context information
    request_context: RequestContext = Field(default_factory=RequestContext, description="Context information for the request")
    load_balancing_factors: LoadBalancingFactors = Field(default_factory=LoadBalancingFactors, description="Load balancing considerations")
    
    @field_validator('capability_scores')
    def validate_capability_scores(cls, v):
        """Ensure all capability scores are between 0 and 1."""
        for expert_id, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Capability score for {expert_id} must be between 0 and 1")
        return v
    
    model_config = ConfigDict(extra='forbid')

class MOEExpertResponseV2_5(BaseModel):
    """Response from an individual expert in the MOE system."""
    
    # Expert identification
    expert_id: str = Field(..., description="ID of the responding expert")
    expert_name: str = Field(..., description="Name of the responding expert")
    
    # Response data
    response_data: ToolResultData = Field(..., description="Expert's response data")
    confidence_score: float = Field(..., description="Expert's confidence in response", ge=0.0, le=1.0)
    
    # Performance metrics
    processing_time_ms: float = Field(..., description="Time taken to generate response", ge=0.0)
    memory_used_mb: float = Field(default=0.0, description="Memory used for this response", ge=0.0)
    
    # Quality indicators
    quality_score: float = Field(default=0.0, description="Quality score of the response", ge=0.0, le=1.0)
    uncertainty_score: float = Field(default=0.0, description="Uncertainty in the response", ge=0.0, le=1.0)
    
    # Status and errors
    success: bool = Field(default=True, description="Whether the response was successful")
    error_message: Optional[str] = Field(None, description="Error message if response failed")
    warnings: List[str] = Field(default_factory=list, description="Any warnings generated")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    version: str = Field(default="2.5", description="Response schema version")
    
    model_config = ConfigDict(extra='forbid')

class MOEUnifiedResponseV2_5(BaseModel):
    """Unified response from the MOE system combining multiple expert responses."""
    
    # Request identification
    request_id: str = Field(..., description="Unique identifier for the request")
    request_type: str = Field(..., description="Type of request processed")
    
    # Consensus and aggregation
    consensus_strategy: ConsensusStrategy = Field(..., description="Strategy used for consensus")
    agreement_level: AgreementLevel = Field(..., description="Level of agreement among experts")
    final_confidence: float = Field(..., description="Final confidence score", ge=0.0, le=1.0)
    
    # Expert responses
    expert_responses: List[MOEExpertResponseV2_5] = Field(..., description="Individual expert responses")
    participating_experts: List[str] = Field(..., description="List of expert IDs that participated")
    
    # Unified result
    unified_response: ToolResultData = Field(..., description="Unified response data")
    response_quality: float = Field(default=0.0, description="Overall response quality", ge=0.0, le=1.0)
    
    # Performance metrics
    total_processing_time_ms: float = Field(..., description="Total processing time", ge=0.0)
    expert_coordination_time_ms: float = Field(default=0.0, description="Time spent coordinating experts", ge=0.0)
    consensus_time_ms: float = Field(default=0.0, description="Time spent reaching consensus", ge=0.0)
    
    # System health
    system_health: HealthStatus = Field(default=HealthStatus.HEALTHY, description="Overall system health")
    resource_utilization: ResourceUtilization = Field(default_factory=ResourceUtilization, description="Resource utilization metrics")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    version: str = Field(default="2.5", description="Response schema version")
    
    # Debugging and analysis
    debug_info: DebugInfo = Field(default_factory=DebugInfo, description="Debug information")
    performance_breakdown: PerformanceBreakdown = Field(default_factory=PerformanceBreakdown, description="Performance breakdown by component")
    
    model_config = ConfigDict(extra='forbid')


# Forward declarations for HuiHui schemas - will be defined later in the file
if TYPE_CHECKING:
    HuiHuiMarketRegimeSchema = Any
    HuiHuiOptionsFlowSchema = Any
    HuiHuiSentimentSchema = Any
else:
    HuiHuiMarketRegimeSchema = Any
    HuiHuiOptionsFlowSchema = Any
    HuiHuiSentimentSchema = Any

class HuiHuiUnifiedExpertResponse(BaseModel):
    """PYDANTIC-FIRST: Unified response schema for all HuiHui experts."""

    # Common metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: str = Field(..., description="Expert type used")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

    # Usage tracking
    tokens_used: Optional[int] = Field(None, description="Tokens used in analysis")
    cache_hit: bool = Field(default=False, description="Whether result was from cache")

    # Expert-specific data (only one should be populated)
    market_regime_data: Optional[Any] = Field(None, description="Market Regime Expert data")
    options_flow_data: Optional[Any] = Field(None, description="Options Flow Expert data")
    sentiment_data: Optional[Any] = Field(None, description="Sentiment Expert data")

    # General response fields
    success: bool = Field(..., description="True if analysis was successful")
    message: str = Field(..., description="Response message or error details")

    model_config = ConfigDict(extra='allow')

    @model_validator(mode='after')
    @classmethod
    def validate_expert_data(cls, values):
        """Ensure exactly one expert data field is populated."""
        expert_data_fields = [
            values.market_regime_data,
            values.options_flow_data,
            values.sentiment_data
        ]
        populated_fields = [f for f in expert_data_fields if f is not None]
        if len(populated_fields) > 1:
            raise ValueError("Only one expert data field (market_regime_data, options_flow_data, or sentiment_data) should be populated.")
        return values

class MCPToolResultV2_5(BaseModel):
    """Result from an MCP tool execution."""
    tool_id: str = Field(..., description="Unique identifier for the tool")
    tool_name: str = Field(..., description="Name of the tool")
    execution_timestamp: datetime = Field(default_factory=datetime.now, description="When the tool was executed")
    execution_time_ms: float = Field(..., description="Time taken to execute the tool", ge=0.0)
    success: bool = Field(..., description="Whether the tool execution was successful")
    result_data: ToolResultData = Field(..., description="Tool execution result data")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    performance_metrics: PerformanceBreakdown = Field(default_factory=PerformanceBreakdown, description="Tool performance metrics")
    resource_usage: ResourceUtilization = Field(default_factory=ResourceUtilization, description="Resource usage during execution")
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata, description="Additional tool execution metadata")

class MCPIntelligenceResultV2_5(BaseModel):
    """Result from MCP intelligence analysis."""
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    analysis_type: str = Field(..., description="Type of intelligence analysis performed")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the analysis was performed")
    symbol: str = Field(..., description="Symbol being analyzed")
    confidence_score: float = Field(..., description="Confidence in the analysis result", ge=0.0, le=1.0)
    intelligence_data: IntelligenceData = Field(..., description="Intelligence analysis data")
    market_context: MarketContext = Field(..., description="Market context during analysis")
    tools_used: List[MCPToolResultV2_5] = Field(default_factory=list, description="Tools used in analysis")
    performance_metrics: PerformanceBreakdown = Field(default_factory=PerformanceBreakdown, description="Analysis performance metrics")
    recommendations: List[AnalysisRecommendation] = Field(default_factory=list, description="Analysis recommendations")
    risk_assessment: RiskAssessment = Field(default_factory=RiskAssessment, description="Risk assessment data")
    metadata: AnalysisMetadata = Field(default_factory=AnalysisMetadata, description="Additional analysis metadata")
    success: bool = Field(..., description="Whether the analysis was successful")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")

# Export all schemas
__all__ = [
    'ExpertStatus', 'RoutingStrategy', 'ConsensusStrategy', 'AgreementLevel', 'HealthStatus',
    'MOEExpertRegistryV2_5', 'MOEGatingNetworkV2_5', 'MOEExpertResponseV2_5', 'MOEUnifiedResponseV2_5',
    'HuiHuiUnifiedExpertResponse', 'MCPToolResultV2_5', 'MCPIntelligenceResultV2_5',
    'AIAdaptationV2_5', 'AIAdaptationPerformanceV2_5',
    'AIPredictionV2_5', 'AIPredictionPerformanceV2_5', 'AIPredictionRequestV2_5', 'AIPredictionSummaryV2_5',
    'LearningInsightV2_5', 'UnifiedLearningResult', 'MarketPattern', 'PatternThresholds',
    'PerformanceInterval', 'PerformanceMetricType', 'PerformanceMetricV2_5', 'SystemPerformanceV2_5',
    'BacktestPerformanceV2_5', 'ExecutionMetricsV2_5', 'PerformanceReportV2_5',
    'AdaptiveLearningResult', 'RecursiveIntelligenceResult', 'MarketIntelligencePattern',
    'HuiHuiExpertType', 'HuiHuiModelConfigV2_5', 'HuiHuiExpertConfigV2_5', 'HuiHuiAnalysisRequestV2_5',
    'HuiHuiAnalysisResponseV2_5', 'HuiHuiUsageRecordV2_5', 'HuiHuiPerformanceMetricsV2_5',
    'HuiHuiEnsembleConfigV2_5', 'HuiHuiUserFeedbackV2_5', 'HuiHuiMarketRegimeSchema',
    'HuiHuiOptionsFlowSchema', 'HuiHuiSentimentSchema',

    # Real-time intelligence summary
    'RealTimeIntelligenceSummaryV2_5',
]

class RealTimeIntelligenceSummaryV2_5(BaseModel):
    intelligence_active: bool = Field(..., description="Whether intelligence is active")
    intelligence_score: float = Field(..., description="Overall intelligence score")
    sentiment_regime: str = Field(..., description="Current sentiment regime")
    primary_narrative: str = Field(..., description="Primary market narrative")
    top_signal: Optional[str] = Field(None, description="Top intelligence signal")
    diabolical_insight: str = Field(..., description="Top diabolical insight")
    confidence_level: str = Field(..., description="Confidence level as string")
    timestamp: datetime = Field(..., description="Timestamp of the summary")

# =============================================================================
# FROM learning_schemas.py
# =============================================================================
# from learning_config_schemas import (
#     MarketContextData, AdaptationSuggestion, PerformanceMetricsSnapshot,
#     LearningInsightData, ExpertAdaptationSummary, ConfidenceUpdateData,
#     OptimizationRecommendation, LearningMetadata, IntelligenceLayerData,
#     MetaLearningData, PatternFeatures, MarketPrediction, PatternMetaAnalysis
# )

# ⚠️ CRITICAL TODO: AI MODEL PLACEHOLDERS NEED HEAVY RE-EXAMINATION
# These placeholder classes were temporarily converted to fail-fast but still need:
# 1. Complete redesign based on actual AI system requirements
# 2. Integration with real AI/ML model schemas
# 3. Validation against actual AI system outputs
# 4. Removal of any remaining placeholder patterns
# FAIL-FAST ARCHITECTURE: No placeholder classes with fake data allowed!
class MarketContextData(BaseModel):
    """Market context data - FAIL FAST ON MISSING DATA."""
    market_state: str = Field(..., description="Market state - REQUIRED (e.g., 'volatile', 'trending', 'ranging')")

    @field_validator('market_state')
    @classmethod
    def validate_no_fake_market_state(cls, v: str) -> str:
        """CRITICAL: Reject fake or placeholder market state values."""
        fake_states = ['normal', 'default', 'unknown', 'placeholder', 'n/a']
        if v.lower().strip() in fake_states:
            raise ValueError(f"CRITICAL: Market state '{v}' is a placeholder - provide real market analysis!")
        return v

class AdaptationSuggestion(BaseModel):
    """Placeholder for adaptation suggestion."""
    suggestion: str = Field(default="", description="Adaptation suggestion")

class PerformanceMetricsSnapshot(BaseModel):
    """Placeholder for performance metrics snapshot."""
    accuracy: float = Field(default=0.0, description="Accuracy metric")

class LearningInsightData(BaseModel):
    """Placeholder for learning insight data."""
    insights: List[str] = Field(default_factory=list, description="Learning insights")

class ExpertAdaptationSummary(BaseModel):
    """Placeholder for expert adaptation summary."""
    adaptations: List[str] = Field(default_factory=list, description="Expert adaptations")

class ConfidenceUpdateData(BaseModel):
    """Placeholder for confidence update data."""
    confidence_score: float = Field(default=0.0, description="Confidence score")

class OptimizationRecommendation(BaseModel):
    """Placeholder for optimization recommendation."""
    recommendation: str = Field(default="", description="Optimization recommendation")

class LearningMetadata(BaseModel):
    """Placeholder for learning metadata."""
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Learning metadata")

class IntelligenceLayerData(BaseModel):
    """Placeholder for intelligence layer data."""
    layer_data: Dict[str, Any] = Field(default_factory=dict, description="Intelligence layer data")

class MetaLearningData(BaseModel):
    """Placeholder for meta learning data."""
    meta_data: Dict[str, Any] = Field(default_factory=dict, description="Meta learning data")

class PatternFeatures(BaseModel):
    """Placeholder for pattern features."""
    features: List[str] = Field(default_factory=list, description="Pattern features")

class MarketPrediction(BaseModel):
    """Placeholder for market prediction."""
    prediction: str = Field(default="", description="Market prediction")

class PatternMetaAnalysis(BaseModel):
    """Placeholder for pattern meta analysis."""
    analysis: Dict[str, Any] = Field(default_factory=dict, description="Pattern meta analysis")

class LearningInsightV2_5(BaseModel):
    """Represents a single learning insight generated by the HuiHui system."""
    insight_id: str = Field(..., description="Unique identifier for the insight.")
    insight_type: str = Field(..., description="Category of the insight (e.g., 'performance_anomaly', 'pattern_discovery').")
    insight_description: str = Field(..., description="Detailed description of the insight.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence level of the insight (0.0 to 1.0).")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp when the insight was generated.")
    market_context: MarketContextData = Field(default_factory=MarketContextData, description="Market conditions or context relevant to the insight.")
    suggested_adaptation: AdaptationSuggestion = Field(default_factory=AdaptationSuggestion, description="Suggested adaptation parameters based on the insight.")
    performance_metrics_pre: PerformanceMetricsSnapshot = Field(default_factory=PerformanceMetricsSnapshot, description="Performance metrics before the insight was applied.")
    performance_metrics_post: PerformanceMetricsSnapshot = Field(default_factory=PerformanceMetricsSnapshot, description="Expected performance metrics after the insight is applied.")
    integration_priority: int = Field(5, ge=1, le=10, description="Priority for integrating this insight (1=highest, 10=lowest).")
    integration_complexity: int = Field(5, ge=1, le=10, description="Complexity of applying the suggested adaptation (1=low, 10=high).")
    adaptation_type: Optional[str] = Field(None, description="Specific type of adaptation suggested (e.g., 'parameter_adjustment', 'model_retraining').")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered during insight generation or processing.")

class UnifiedLearningResult(BaseModel):
    """Comprehensive result of a unified learning cycle, summarizing insights and adaptations."""
    symbol: str = Field(..., description="Symbol for which learning was performed (or 'SYSTEM' for system-wide).")
    analysis_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of this learning analysis.")
    learning_insights: LearningInsightData = Field(default_factory=LearningInsightData, description="Summarized learning insights.")
    performance_improvements: PerformanceMetricsSnapshot = Field(default_factory=PerformanceMetricsSnapshot, description="Quantified performance improvements from adaptations.")
    expert_adaptations: ExpertAdaptationSummary = Field(default_factory=ExpertAdaptationSummary, description="Summary of adaptations made by expert systems.")
    confidence_updates: ConfidenceUpdateData = Field(default_factory=ConfidenceUpdateData, description="Updates on confidence scores over time.")
    next_learning_cycle: datetime = Field(..., description="Timestamp for the next scheduled learning cycle.")
    learning_cycle_type: str = Field(..., description="Type of learning cycle (e.g., 'daily', 'weekly', 'continuous').")
    lookback_period_days: int = Field(..., ge=1, description="Number of days considered for this learning cycle.")
    performance_improvement_score: float = Field(..., ge=0.0, le=1.0, description="Overall score indicating performance improvement (0.0 to 1.0).")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score of the learning system.")
    optimization_recommendations: List[OptimizationRecommendation] = Field(default_factory=list, description="Specific recommendations for further optimization.")
    eots_schema_compliance: bool = Field(True, description="Indicates if the learning result is compliant with EOTS schemas.")
    learning_metadata: LearningMetadata = Field(default_factory=LearningMetadata, description="Additional metadata about the learning process.")

class MarketPattern(BaseModel):
    """Represents a detected market pattern."""
    pattern_id: str = Field(..., description="Unique pattern identifier")
    symbol: str = Field(..., description="Trading symbol")
    pattern_type: str = Field(..., description="Type of pattern detected")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Pattern confidence")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Historical success rate")
    market_conditions: MarketContextData = Field(description="Market conditions when pattern occurred")
    eots_metrics: Dict[str, float] = Field(description="EOTS metrics at pattern time")
    outcome: Optional[str] = Field(None, description="Pattern outcome if known")
    learning_weight: float = Field(default=1.0, description="Weight for learning algorithm")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of pattern detection.")

class PatternThresholds(BaseModel):
    """Defines thresholds for various market pattern detections."""
    volatility_expansion: float = Field(..., description="Threshold for volatility expansion pattern.")
    trend_continuation: float = Field(..., description="Threshold for trend continuation pattern.")
    accumulation: float = Field(..., description="Threshold for accumulation pattern.")
    distribution: float = Field(..., description="Threshold for distribution pattern.")
    consolidation: float = Field(..., description="Threshold for consolidation pattern.")
    significant_pos_thresh: float = Field(..., description="Significant positive threshold.")
    dwfd_strong_thresh: float = Field(..., description="DWFD strong threshold.")
    moderate_confidence_thresh: float = Field(..., description="Moderate confidence threshold.")

class AdaptiveLearningResult(BaseModel):
    """Pydantic model for adaptive learning results."""
    learning_iteration: int = Field(description="Current learning iteration")
    patterns_analyzed: int = Field(description="Number of patterns analyzed")
    accuracy_improvement: float = Field(description="Accuracy improvement percentage")
    new_insights: List[str] = Field(description="New insights discovered")
    confidence_evolution: float = Field(description="Evolution in confidence scoring")
    adaptation_score: float = Field(ge=0.0, le=10.0, description="Overall adaptation score")

class RecursiveIntelligenceResult(BaseModel):
    """Pydantic model for recursive intelligence analysis."""
    analysis_depth: int = Field(description="Depth of recursive analysis")
    intelligence_layers: List[IntelligenceLayerData] = Field(description="Layers of intelligence")
    convergence_score: float = Field(description="Analysis convergence score")
    recursive_insights: List[str] = Field(description="Insights from recursive analysis")
    meta_learning_data: MetaLearningData = Field(description="Meta-learning information")

class MarketIntelligencePattern(BaseModel):
    """Represents a market intelligence pattern detected by the system."""
    pattern_id: str = Field(..., description="Unique identifier for the pattern")
    pattern_type: str = Field(..., description="Type of intelligence pattern")
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Pattern detection timestamp")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Pattern confidence score")
    market_conditions: MarketContextData = Field(..., description="Market conditions when pattern was detected")
    intelligence_metrics: Dict[str, float] = Field(..., description="Intelligence metrics at pattern time")
    pattern_features: PatternFeatures = Field(..., description="Key features of the pattern")
    historical_accuracy: float = Field(..., ge=0.0, le=1.0, description="Historical accuracy of similar patterns")
    prediction: Optional[MarketPrediction] = Field(None, description="Pattern-based market prediction")
    validation_metrics: Dict[str, float] = Field(default_factory=dict, description="Pattern validation metrics")
    meta_analysis: PatternMetaAnalysis = Field(default_factory=PatternMetaAnalysis, description="Meta-analysis of the pattern")


# FROM hui_hui_schemas.py
"""
Pydantic models for the HuiHui AI Expert System integration in EOTS v2.5.

This module defines the data structures used for the HuiHui AI expert system,
including expert configurations, analysis requests/responses, and performance tracking.
"""
# from hui_hui_config_schemas import (
#     AnalysisContext, RequestMetadata, EOTSPrediction, TradingRecommendation, PerformanceByCondition
# )

# FAIL-FAST ARCHITECTURE: No placeholder classes with fake data allowed!
class AnalysisContext(BaseModel):
    """Analysis context - FAIL FAST ON MISSING DATA."""
    context_data: Dict[str, Any] = Field(..., min_items=1, description="Analysis context - REQUIRED and must not be empty")

    @field_validator('context_data')
    @classmethod
    def validate_context_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """CRITICAL: Ensure context data is not empty - empty dict indicates missing analysis."""
        if not v:
            raise ValueError("CRITICAL: context_data cannot be empty - this indicates missing analysis context!")
        return v

class RequestMetadata(BaseModel):
    """Request metadata - FAIL FAST ON MISSING DATA."""
    request_id: str = Field(..., min_length=1, description="Request ID - REQUIRED and must not be empty")

    @field_validator('request_id')
    @classmethod
    def validate_request_id_not_empty(cls, v: str) -> str:
        """CRITICAL: Ensure request ID is not empty or placeholder."""
        if not v.strip() or v.lower() in ['', 'default', 'placeholder', 'unknown']:
            raise ValueError("CRITICAL: request_id cannot be empty or placeholder - provide real request ID!")
        return v

class EOTSPrediction(BaseModel):
    """Placeholder for EOTS prediction."""
    prediction: str = Field(default="", description="EOTS prediction")

class TradingRecommendation(BaseModel):
    """Placeholder for trading recommendation."""
    recommendation: str = Field(default="", description="Trading recommendation")

class PerformanceByCondition(BaseModel):
    """Placeholder for performance by condition."""
    condition: str = Field(default="", description="Market condition")
    performance: float = Field(default=0.0, description="Performance metric")

class HuiHuiExpertType(str, Enum):
    """Defines the different types of experts available in the HuiHui AI system."""
    MARKET_REGIME = "market_regime"; OPTIONS_FLOW = "options_flow"; SENTIMENT = "sentiment"; ORCHESTRATOR = "orchestrator"
    VOLATILITY = "volatility"; LIQUIDITY = "liquidity"; RISK = "risk"; EXECUTION = "execution"

class HuiHuiModelConfigV2_5(BaseModel):
    """Configuration settings for individual HuiHui AI models controlling generation behavior and integration."""
    expert_type: HuiHuiExpertType = Field(default=HuiHuiExpertType.ORCHESTRATOR, description="Type of expert this configuration applies to.")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Controls randomness in model outputs (0=deterministic, 1.0=creative).")
    max_tokens: int = Field(default=4000, ge=100, le=8000, description="Maximum number of tokens to generate in the response.")
    enable_eots_integration: bool = Field(default=True, description="Whether to integrate with EOTS for additional context and data enrichment.")
    context_budget: int = Field(default=4000, ge=1000, le=8000, description="Number of tokens to reserve for context in the prompt.")
    timeout_seconds: int = Field(default=90, ge=30, le=300, description="Maximum time in seconds to wait for a response from the model.")
    model_config = ConfigDict(extra='forbid')


class HuiHuiExpertConfigV2_5(BaseModel):
    """Configuration for individual HuiHui experts, defining their behavior and capabilities."""
    expert_name: str = Field(..., min_length=3, max_length=100, description="Unique identifier for this expert configuration.")
    specialist_id: str = Field(..., description="ID of the specialist model to use for this expert (e.g., 'expert-market-regime-v2').")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Controls randomness in this expert's outputs (0=deterministic, 1.0=creative).")
    keywords: List[str] = Field(default_factory=list, description="List of keywords that trigger this expert's activation in the ensemble.")
    eots_metrics: List[str] = Field(default_factory=list, description="List of EOTS metrics this expert should have access to for analysis.")
    performance_weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weight of this expert's opinion in ensemble decisions (0.0 to 1.0).")
    is_active: bool = Field(default=True, description="Whether this expert is currently active and should be included in the ensemble.")
    model_config = ConfigDict(extra='forbid')


class HuiHuiAnalysisRequestV2_5(BaseModel):
    """Request model for submitting analysis tasks to the HuiHui AI system."""
    symbol: str = Field(..., description="Trading symbol to analyze (e.g., 'SPY', 'QQQ').")
    analysis_type: str = Field(..., description="Type of analysis to perform (e.g., 'market_regime', 'flow_analysis').")
    bundle_data: Optional["ProcessedDataBundleV2_5"] = Field(None, description="Processed market data bundle for analysis.")
    context: AnalysisContext = Field(default_factory=AnalysisContext, description="Additional context for the analysis (e.g., market conditions, recent news).")
    expert_preference: Optional[HuiHuiExpertType] = Field(None, description="Preferred expert type for this analysis (optional).")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this request was created (UTC).")
    request_metadata: RequestMetadata = Field(default_factory=RequestMetadata, description="Additional metadata about this request (e.g., request ID, user ID).")
    model_config = ConfigDict(extra='forbid')


class HuiHuiAnalysisResponseV2_5(BaseModel):
    """Response model containing the results of a HuiHui AI analysis."""
    expert_used: HuiHuiExpertType = Field(..., description="Type of expert that generated this response.")
    analysis_content: str = Field(..., description="Detailed analysis content in markdown format with embedded visualizations.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Expert's confidence in the analysis (0.0 to 1.0).")
    processing_time: float = Field(..., ge=0.0, description="Time taken to generate the analysis in seconds.")
    insights: List[str] = Field(default_factory=list, description="Key insights extracted from the analysis.")
    eots_predictions: Optional[List[EOTSPrediction]] = Field(None, description="Relevant EOTS predictions that informed this analysis.")
    recommendations: Optional[List[TradingRecommendation]] = Field(None, description="Trading recommendations derived from the analysis.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this response was generated (UTC).")
    model_config = ConfigDict(extra='forbid')


class HuiHuiUsageRecordV2_5(BaseModel):
    """Tracks usage and performance metrics for individual HuiHui expert invocations."""
    expert_used: HuiHuiExpertType = Field(..., description="Type of expert that was used for this analysis.")
    symbol: str = Field(..., description="Trading symbol that was analyzed (e.g., 'SPY', 'QQQ').")
    processing_time: float = Field(..., ge=0.0, description="Time taken to process the request in seconds.")
    success: bool = Field(..., description="Whether the request was processed successfully.")
    error_message: Optional[str] = Field(None, description="Error message if the request failed, None if successful.")
    market_condition: str = Field(default="normal", description="Prevailing market condition during analysis (e.g., 'bull', 'bear', 'range').")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this usage record was created (UTC).")
    model_config = ConfigDict(extra='forbid')


class HuiHuiPerformanceMetricsV2_5(BaseModel):
    """Aggregated performance metrics for a specific HuiHui expert type over time."""
    expert_type: HuiHuiExpertType = Field(..., description="Type of expert these metrics apply to.")
    total_requests: int = Field(default=0, ge=0, description="Total number of requests processed by this expert type.")
    successful_requests: int = Field(default=0, ge=0, description="Number of successfully processed requests by this expert type.")
    average_processing_time: float = Field(default=0.0, ge=0.0, description="Average processing time in seconds across all requests.")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Ratio of successful requests to total requests (0.0 to 1.0).")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When these metrics were last calculated (UTC).")
    performance_by_market_condition: Dict[str, PerformanceByCondition] = Field(
        default_factory=dict, 
        description="Performance metrics (success_rate, avg_processing_time) broken down by market condition (e.g., 'bull', 'bear')."
    )
    model_config = ConfigDict(extra='forbid')


class HuiHuiEnsembleConfigV2_5(BaseModel):
    """Configuration for creating and managing ensembles of multiple HuiHui experts."""
    ensemble_name: str = Field(..., description="Unique name identifying this expert ensemble configuration.")
    member_experts: List[HuiHuiExpertConfigV2_5] = Field(default_factory=list, description="List of expert configurations included in this ensemble.")
    voting_strategy: str = Field(default="weighted_average", description="Strategy for combining expert opinions (e.g., 'weighted_average', 'majority_vote').")
    min_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence score (0.0-1.0) required for the ensemble to provide a final answer.")
    fallback_expert: Optional[str] = Field(None, description="Name of the expert to use when ensemble confidence is below the threshold.")
    model_config = ConfigDict(extra='forbid')


class HuiHuiUserFeedbackV2_5(BaseModel):
    """Stores user feedback and ratings for HuiHui AI analysis results."""
    request_id: str = Field(..., description="Unique identifier of the original analysis request this feedback refers to.")
    expert_used: HuiHuiExpertType = Field(..., description="Type of expert that generated the analysis being rated.")
    rating: int = Field(..., ge=1, le=5, description="User's rating of the analysis quality (1=Poor to 5=Excellent).")
    helpful: bool = Field(..., description="Whether the analysis was helpful for the user's specific needs.")
    comments: Optional[str] = Field(None, description="Optional detailed comments from the user about the analysis.")
    suggested_improvements: List[str] = Field(default_factory=list, description="Specific suggestions from the user for improving the analysis.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the feedback was submitted (UTC).")
    model_config = ConfigDict(extra='forbid')

if TYPE_CHECKING:
    from .core_models import ProcessedDataBundleV2_5


class HuiHuiMarketRegimeSchema(BaseModel):
    """PYDANTIC-FIRST: Market Regime Expert specific data schema for database storage."""
    
    # Analysis metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: Literal["market_regime"] = Field(default="market_regime", description="Expert type")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    # Core regime analysis
    current_regime_id: int = Field(..., description="Current regime ID (1-20)")
    current_regime_name: str = Field(..., description="Current regime name")
    regime_confidence: float = Field(..., description="Regime confidence score", ge=0.0, le=1.0)
    regime_probability: float = Field(..., description="Regime probability", ge=0.0, le=1.0)
    
    # VRI 3.0 Components
    vri_3_composite: float = Field(..., description="VRI 3.0 composite score")
    volatility_regime_score: float = Field(..., description="Volatility regime component")
    flow_intensity_score: float = Field(..., description="Flow intensity component")
    regime_stability_score: float = Field(..., description="Regime stability component")
    transition_momentum_score: float = Field(..., description="Transition momentum component")
    
    # Regime characteristics
    volatility_level: str = Field(..., description="Volatility level (low/medium/high/extreme)")
    trend_direction: str = Field(..., description="Trend direction (bullish/bearish/sideways)")
    flow_pattern: str = Field(..., description="Flow pattern (accumulation/distribution/neutral)")
    risk_appetite: str = Field(..., description="Risk appetite (risk_on/risk_off/neutral)")
    
    # Transition prediction
    predicted_regime_id: Optional[int] = Field(None, description="Predicted next regime ID")
    transition_probability: float = Field(default=0.0, description="Transition probability", ge=0.0, le=1.0)
    expected_transition_timeframe: Optional[str] = Field(None, description="Expected transition timeframe")
    
    # Performance metrics
    processing_time_ms: float = Field(..., description="Processing time in milliseconds", ge=0.0)
    data_quality_score: float = Field(default=1.0, description="Data quality score", ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.0, description="Overall confidence level", ge=0.0, le=1.0)
    
    # Supporting data
    supporting_indicators: List[str] = Field(default_factory=list, description="Supporting indicators")
    conflicting_indicators: List[str] = Field(default_factory=list, description="Conflicting indicators")
    early_warning_signals: List[str] = Field(default_factory=list, description="Early warning signals")


class HuiHuiOptionsFlowSchema(BaseModel):
    """PYDANTIC-FIRST: Options Flow Expert specific data schema for database storage."""
    
    # Analysis metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: Literal["options_flow"] = Field(default="options_flow", description="Expert type")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    # Core flow metrics
    vapi_fa_z_score: float = Field(..., description="VAPI-FA Z-score")
    dwfd_z_score: float = Field(..., description="DWFD Z-score")
    tw_laf_score: float = Field(..., description="TW-LAF score")
    gib_oi_based: float = Field(..., description="GIB OI-based value")
    
    # SDAG Analysis
    sdag_multiplicative: float = Field(default=0.0, description="SDAG multiplicative methodology")
    sdag_directional: float = Field(default=0.0, description="SDAG directional methodology")
    sdag_weighted: float = Field(default=0.0, description="SDAG weighted methodology")
    sdag_volatility_focused: float = Field(default=0.0, description="SDAG volatility-focused methodology")
    
    # DAG Analysis
    dag_multiplicative: float = Field(default=0.0, description="DAG multiplicative methodology")
    dag_additive: float = Field(default=0.0, description="DAG additive methodology")
    dag_weighted: float = Field(default=0.0, description="DAG weighted methodology")
    dag_consensus: float = Field(default=0.0, description="DAG consensus methodology")
    
    # Flow classification
    flow_type: str = Field(..., description="Primary flow type")
    flow_subtype: str = Field(..., description="Flow subtype")
    flow_intensity: str = Field(..., description="Flow intensity level")
    directional_bias: str = Field(..., description="Directional bias (bullish/bearish/neutral)")
    
    # Participant analysis
    institutional_probability: float = Field(..., description="Institutional participant probability", ge=0.0, le=1.0)
    retail_probability: float = Field(..., description="Retail participant probability", ge=0.0, le=1.0)
    dealer_probability: float = Field(..., description="Dealer participant probability", ge=0.0, le=1.0)
    
    # Intelligence metrics
    sophistication_score: float = Field(..., description="Flow sophistication score", ge=0.0, le=1.0)
    information_content: float = Field(..., description="Information content score", ge=0.0, le=1.0)
    market_impact_potential: float = Field(..., description="Potential market impact", ge=0.0, le=1.0)
    
    # Gamma dynamics
    gamma_exposure_net: float = Field(..., description="Net gamma exposure")
    gamma_wall_strength: float = Field(default=0.0, description="Gamma wall strength at key levels")
    gamma_flip_levels: List[float] = Field(default_factory=list, description="Identified gamma flip levels")

    # Performance metrics
    processing_time_ms: float = Field(..., description="Processing time in milliseconds", ge=0.0)
    data_quality_score: float = Field(default=1.0, description="Data quality score", ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.0, description="Overall confidence level", ge=0.0, le=1.0)

    # Supporting data
    supporting_indicators: List[str] = Field(default_factory=list, description="Supporting flow indicators")
    conflicting_indicators: List[str] = Field(default_factory=list, description="Conflicting flow indicators")
    flow_anomalies: List[str] = Field(default_factory=list, description="Detected flow anomalies")


class HuiHuiSentimentSchema(BaseModel):
    """PYDANTIC-FIRST: Sentiment Expert specific data schema for database storage."""
    
    # Analysis metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: Literal["sentiment"] = Field(default="sentiment", description="Expert type")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    # Core sentiment metrics
    overall_sentiment_score: float = Field(..., description="Overall sentiment score", ge=-1.0, le=1.0)
    sentiment_direction: str = Field(..., description="Sentiment direction (bullish/bearish/neutral)")
    sentiment_strength: float = Field(..., description="Sentiment strength", ge=0.0, le=1.0)
    sentiment_confidence: float = Field(..., description="Sentiment confidence", ge=0.0, le=1.0)
    
    # Sentiment components
    price_action_sentiment: float = Field(default=0.0, description="Price action sentiment", ge=-1.0, le=1.0)
    volume_sentiment: float = Field(default=0.0, description="Volume sentiment", ge=-1.0, le=1.0)
    options_sentiment: float = Field(default=0.0, description="Options sentiment", ge=-1.0, le=1.0)
    news_sentiment: Optional[float] = Field(None, description="News sentiment (if available)", ge=-1.0, le=1.0)
    social_sentiment: Optional[float] = Field(None, description="Social media sentiment (if available)", ge=-1.0, le=1.0)
    
    # Behavioral analysis
    fear_greed_index: float = Field(default=0.0, description="Fear/Greed index", ge=0.0, le=100.0)
    market_psychology: str = Field(default="neutral", description="Overall market psychology assessment")
    sentiment_momentum: float = Field(default=0.0, description="Sentiment momentum indicator", ge=-1.0, le=1.0)

    # Advanced sentiment metrics
    put_call_ratio_sentiment: Optional[float] = Field(None, description="Put/call ratio sentiment indicator")
    vix_sentiment: Optional[float] = Field(None, description="VIX-based sentiment indicator")
    flow_sentiment: Optional[float] = Field(None, description="Options flow sentiment")

    # Performance metrics
    processing_time_ms: float = Field(..., description="Processing time in milliseconds", ge=0.0)
    data_quality_score: float = Field(default=1.0, description="Data quality score", ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.0, description="Overall confidence level", ge=0.0, le=1.0)

    # Supporting data
    sentiment_drivers: List[str] = Field(default_factory=list, description="Key sentiment drivers")
    sentiment_conflicts: List[str] = Field(default_factory=list, description="Conflicting sentiment signals")
    sentiment_catalysts: List[str] = Field(default_factory=list, description="Potential sentiment catalysts")

# =============================================================================
# FROM huihui_models.py
# =============================================================================
"""
EXPERT ROUTER - DATA MODELS

This module contains the data models used by the ExpertRouter system.
"""

# HuiHuiExpertType is already defined above in the file

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
        self.last_used = datetime.now(timezone.utc)

class ExpertPerformance(BaseModel):
    """Tracks performance metrics for each expert type."""
    metrics: Dict[HuiHuiExpertType, PerformanceMetrics] = Field(default_factory=lambda: defaultdict(PerformanceMetrics))
    recent_decisions: Deque[Tuple[HuiHuiExpertType, float]] = Field(default_factory=deque)
    max_history: int = 1000

    model_config = ConfigDict(extra='forbid')

    def update_metrics(self, expert_type: HuiHuiExpertType, success: bool, response_time: float) -> None:
        self.metrics[expert_type].update(success, response_time)
        self.recent_decisions.append((expert_type, datetime.now(timezone.utc).timestamp()))
        if len(self.recent_decisions) > self.max_history:
            self.recent_decisions.popleft()

    def get_recent_usage(self, window_seconds: int = 3600) -> Dict[HuiHuiExpertType, int]:
        cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
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


# =============================================================================
# FROM performance_schemas.py
# =============================================================================
"""
Performance tracking and analytics models for the Elite Options Trading System v2.5.

This module defines Pydantic models for tracking system performance, backtesting results,
and execution metrics in a consistent, type-safe manner.
"""

# Placeholder classes for missing imports
class PerformanceMetadata(BaseModel):
    """Placeholder for performance metadata."""
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Performance metadata")

class StrategyParameters(BaseModel):
    """Placeholder for strategy parameters."""
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")

class PerformanceSummary(BaseModel):
    """Placeholder for performance summary."""
    summary: Dict[str, Any] = Field(default_factory=dict, description="Performance summary")

class SystemHealthMetrics(BaseModel):
    """Placeholder for system health metrics."""
    health_score: float = Field(default=1.0, description="System health score")

class RiskMetrics(BaseModel):
    """Placeholder for risk metrics."""
    risk_score: float = Field(default=0.0, description="Risk score")

class MarketConditions(BaseModel):
    """Placeholder for market conditions."""
    condition: str = Field(default="normal", description="Market condition")

class PerformanceInterval(str, Enum):
    """Time intervals for performance metrics aggregation."""
    MINUTE_1 = "1m"; MINUTE_5 = "5m"; MINUTE_15 = "15m"; MINUTE_30 = "30m"
    HOUR_1 = "1h"; HOUR_4 = "4h"; HOUR_12 = "12h"; DAILY = "1d"; WEEKLY = "1w"; MONTHLY = "1M"

class PerformanceMetricType(str, Enum):
    """Types of performance metrics that can be tracked."""
    LATENCY = "latency"; THROUGHPUT = "throughput"; MEMORY = "memory"; CPU = "cpu"; NETWORK = "network"
    CACHE_HIT = "cache_hit"; CACHE_MISS = "cache_miss"; ERROR_RATE = "error_rate"; SUCCESS_RATE = "success_rate"
    ORDER_EXECUTION = "order_execution"; DATA_QUALITY = "data_quality"; BACKTEST_RETURN = "backtest_return"
    SHARPE_RATIO = "sharpe_ratio"; MAX_DRAWDOWN = "max_drawdown"; WIN_RATE = "win_rate"; PROFIT_FACTOR = "profit_factor"

class PerformanceMetricV2_5(BaseModel):
    """Base model for performance metrics with common fields and validation."""
    metric_type: PerformanceMetricType = Field(..., description="Type of performance metric being recorded.")
    component: str = Field(..., description="Component or subsystem this metric applies to.")
    value: float = Field(..., description="Numeric value of the metric.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this metric was recorded (UTC).")
    interval: PerformanceInterval = Field(PerformanceInterval.MINUTE_1, description="Aggregation interval for this metric.")
    metadata: PerformanceMetadata = Field(default_factory=PerformanceMetadata, description="Additional context-specific metadata.")
    model_config = ConfigDict(extra='forbid')

    @field_validator('value')
    def validate_value(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("Value must be numeric")
        if v < 0 and cls.metric_type not in [PerformanceMetricType.ERROR_RATE, PerformanceMetricType.MAX_DRAWDOWN]:
            raise ValueError(f"Negative values not allowed for {cls.metric_type}")
        return float(v)

class SystemPerformanceV2_5(BaseModel):
    """System-level performance metrics and health indicators."""
    cpu_usage_pct: float = Field(..., ge=0.0, le=100.0, description="Current CPU usage percentage (0-100).")
    memory_usage_pct: float = Field(..., ge=0.0, le=100.0, description="Current memory usage percentage (0-100).")
    disk_usage_pct: float = Field(..., ge=0.0, le=100.0, description="Current disk usage percentage (0-100).")
    network_latency_ms: float = Field(..., ge=0.0, description="Average network latency in milliseconds.")
    active_processes: int = Field(..., ge=0, description="Number of active processes.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this snapshot was taken (UTC).")
    model_config = ConfigDict(extra='forbid')

class BacktestPerformanceV2_5(BaseModel):
    """Comprehensive backtest performance metrics and statistics."""
    strategy_name: str = Field(..., description="Name or identifier of the backtested strategy.")
    start_date: datetime = Field(..., description="Backtest start date (UTC).")
    end_date: datetime = Field(..., description="Backtest end date (UTC).")
    total_return_pct: float = Field(..., description="Total return percentage over the backtest period.")
    annualized_return_pct: float = Field(..., description="Annualized return percentage.")
    annualized_volatility_pct: float = Field(..., ge=0.0, description="Annualized volatility percentage.")
    sharpe_ratio: Optional[float] = Field(None, description="Risk-adjusted return metric (higher is better).")
    sortino_ratio: Optional[float] = Field(None, description="Risk-adjusted return focusing on downside volatility.")
    max_drawdown_pct: float = Field(..., le=0.0, description="Maximum peak-to-trough decline (negative percentage).")
    win_rate_pct: float = Field(..., ge=0.0, le=100.0, description="Percentage of winning trades.")
    profit_factor: float = Field(..., ge=0.0, description="Gross profit divided by gross loss.")
    total_trades: int = Field(..., ge=0, description="Total number of trades executed.")
    avg_trade_duration: timedelta = Field(..., description="Average duration of trades.")
    params: StrategyParameters = Field(default_factory=StrategyParameters, description="Strategy parameters used in this backtest.")
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def validate_dates(self) -> 'BacktestPerformanceV2_5':
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
        return self

class ExecutionMetricsV2_5(BaseModel):
    """Detailed metrics for trade execution quality and performance."""
    order_id: str = Field(..., description="Unique identifier for the order.")
    symbol: str = Field(..., description="Traded symbol.")
    order_type: str = Field(..., description="Type of order (e.g., 'market', 'limit').")
    side: str = Field(..., description="'buy' or 'sell'.")
    quantity: float = Field(..., gt=0, description="Number of contracts/shares.")
    target_price: float = Field(..., gt=0, description="Target or limit price.")
    avg_fill_price: float = Field(..., gt=0, description="Average fill price.")
    slippage: float = Field(..., description="Difference between target and actual fill price.")
    slippage_pct: float = Field(..., description="Slippage as percentage of target price.")
    execution_time_ms: int = Field(..., ge=0, description="Time to execute the order in milliseconds.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the order was executed (UTC).")
    model_config = ConfigDict(extra='forbid')

class PerformanceReportV2_5(BaseModel):
    """Comprehensive performance report combining multiple metrics and analyses."""
    report_id: str = Field(..., description="Unique identifier for this report.")
    start_time: datetime = Field(..., description="Start of the reporting period (UTC).")
    end_time: datetime = Field(..., description="End of the reporting period (UTC).")
    system_metrics: List[SystemPerformanceV2_5] = Field(default_factory=list, description="System performance snapshots.")
    backtest_results: List[BacktestPerformanceV2_5] = Field(default_factory=list, description="Backtest performance results.")
    execution_metrics: List[ExecutionMetricsV2_5] = Field(default_factory=list, description="Trade execution metrics.")
    custom_metrics: Dict[str, List[PerformanceMetricV2_5]] = Field(default_factory=dict, description="Additional custom metrics.")
    summary: PerformanceSummary = Field(default_factory=PerformanceSummary, description="Aggregated summary statistics.")
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def validate_report_period(self) -> 'PerformanceReportV2_5':
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
        return self
    quantity: float = Field(..., gt=0, description="Number of contracts/shares.")
    target_price: float = Field(..., gt=0, description="Target or limit price.")
    avg_fill_price: float = Field(..., gt=0, description="Average fill price.")
    slippage: float = Field(..., description="Difference between target and actual fill price.")
    slippage_pct: float = Field(..., description="Slippage as percentage of target price.")
    execution_time_ms: int = Field(..., ge=0, description="Time to execute the order in milliseconds.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the order was executed (UTC).")
    model_config = ConfigDict(extra='forbid')

class PerformanceReportV2_5(BaseModel):
    """Comprehensive performance report combining multiple metrics and analyses."""
    report_id: str = Field(..., description="Unique identifier for this report.")
    start_time: datetime = Field(..., description="Start of the reporting period (UTC).")
    end_time: datetime = Field(..., description="End of the reporting period (UTC).")
    system_metrics: List[SystemPerformanceV2_5] = Field(default_factory=list, description="System performance snapshots.")
    backtest_results: List[BacktestPerformanceV2_5] = Field(default_factory=list, description="Backtest performance results.")
    execution_metrics: List[ExecutionMetricsV2_5] = Field(default_factory=list, description="Trade execution metrics.")
    custom_metrics: Dict[str, List[PerformanceMetricV2_5]] = Field(default_factory=dict, description="Additional custom metrics.")
    summary: PerformanceSummary = Field(default_factory=PerformanceSummary, description="Aggregated summary statistics.")
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def validate_report_period(self) -> 'PerformanceReportV2_5':
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
        return self
