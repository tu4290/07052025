# MOE (Mixture of Experts) Schemas V2.5
# Pydantic-first data models for the MOE system

from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from data_models.hui_hui_schemas import HuiHuiExpertType, HuiHuiMarketRegimeSchema, HuiHuiOptionsFlowSchema, HuiHuiSentimentSchema
from data_models.moe_config_schemas import (
    RequestContext, LoadBalancingFactors, ResourceUtilization,
    PerformanceBreakdown, DebugInfo, ToolResultData,
    IntelligenceData, MarketContext, AnalysisRecommendation,
    RiskAssessment, AnalysisMetadata
)

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
    
    class Config:
        extra = 'forbid'

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
    
    class Config:
        extra = 'forbid'

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
    
    class Config:
        extra = 'forbid'

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
    
    class Config:
        extra = 'forbid'


class HuiHuiUnifiedExpertResponse(BaseModel):
    """PYDANTIC-FIRST: Unified response schema for all HuiHui experts."""
    
    # Common metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    expert_type: HuiHuiExpertType = Field(..., description="Expert type used")
    ticker: str = Field(..., description="Analyzed ticker")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    # Usage tracking
    tokens_used: Optional[int] = Field(None, description="Tokens used in analysis")
    cache_hit: bool = Field(default=False, description="Whether result was from cache")
    
    # Expert-specific data (only one should be populated)
    market_regime_data: Optional[HuiHuiMarketRegimeSchema] = Field(None, description="Market Regime Expert data")
    options_flow_data: Optional[HuiHuiOptionsFlowSchema] = Field(None, description="Options Flow Expert data")
    sentiment_data: Optional[HuiHuiSentimentSchema] = Field(None, description="Sentiment Expert data")
    
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
        if len(populated_fields) != 1:
            raise ValueError("Exactly one expert data field (market_regime_data, options_flow_data, or sentiment_data) must be populated.")
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
    'HuiHuiUnifiedExpertResponse', 'MCPToolResultV2_5', 'MCPIntelligenceResultV2_5'
]