"""
Expert & AI Configuration Models for EOTS v2.5

This module contains expert system configurations, MOE (Mixture of Experts) settings,
and AI-related configurations.

Extracted from configuration_models.py for better modularity.
"""

# Standard library imports
from typing import Dict, List, Optional, Any
from enum import Enum

# Third-party imports
from pydantic import BaseModel, Field, ConfigDict, field_validator


# =============================================================================
# EXPERT SYSTEM ENUMS
# =============================================================================

class ModelProvider(str, Enum):
    """Supported LLM model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    AZURE = "azure"

class SecurityLevel(str, Enum):
    """Security levels for API access."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ENTERPRISE = "enterprise"

class ExpertType(str, Enum):
    """Types of experts in the system."""
    MARKET_REGIME = "market_regime"
    OPTIONS_FLOW = "options_flow"
    SENTIMENT = "sentiment"
    VOLATILITY = "volatility"
    RISK = "risk"
    EXECUTION = "execution"


# =============================================================================
# PLACEHOLDER CLASSES FOR MISSING IMPORTS
# =============================================================================

class CustomApiKeys(BaseModel):
    """Placeholder for custom API keys."""
    custom_keys: Optional[Dict[str, str]] = Field(default=None, description="Custom API keys")
    

class CustomModelConfig(BaseModel):
    """Placeholder for custom model configuration."""
    model_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom model parameters")

class CustomEndpoints(BaseModel):
    """Placeholder for custom endpoints."""
    endpoints: Optional[Dict[str, str]] = Field(default=None, description="Custom endpoints")

class CustomRateLimits(BaseModel):
    """Placeholder for custom rate limits."""
    limits: Optional[Dict[str, int]] = Field(default=None, description="Custom rate limits")

class CustomSecuritySettings(BaseModel):
    """Placeholder for custom security settings."""
    security_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom security parameters")

class CustomPerformanceSettings(BaseModel):
    """Placeholder for custom performance settings."""
    performance_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom performance parameters")

class CustomIntegrationSettings(BaseModel):
    """Placeholder for custom integration settings."""
    integration_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom integration parameters")

class CustomAgentSettings(BaseModel):
    """Placeholder for custom agent settings."""
    agent_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom agent parameters")

class CustomLearningSettings(BaseModel):
    """Placeholder for custom learning settings."""
    learning_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom learning parameters")

class CustomSafetySettings(BaseModel):
    """Placeholder for custom safety settings."""
    safety_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom safety parameters")

class CustomInsightSettings(BaseModel):
    """Placeholder for custom insight settings."""
    insight_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom insight parameters")

class ThresholdTypes(str, Enum):
    """Types of thresholds."""
    CONFIDENCE = "confidence"
    PERFORMANCE = "performance"
    RISK = "risk"

class CustomThresholdSettings(BaseModel):
    """Placeholder for custom threshold settings."""
    thresholds: Optional[Dict[str, float]] = Field(default=None, description="Custom thresholds")


# =============================================================================
# API & MODEL CONFIGURATION
# =============================================================================

class ApiKeyConfig(BaseModel):
    """Configuration for API keys."""
    openai_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_key: Optional[str] = Field(None, description="Anthropic API key")
    azure_key: Optional[str] = Field(None, description="Azure OpenAI API key")
    huggingface_key: Optional[str] = Field(None, description="HuggingFace API key")
    custom_keys: CustomApiKeys = Field(default_factory=CustomApiKeys, description="Custom API keys")
    
class ModelConfig(BaseModel):
    """Configuration for AI models - FAIL FAST ON MISSING CONFIG."""
    provider: ModelProvider = Field(..., description="Model provider - REQUIRED")
    model_name: str = Field(..., min_length=1, description="Name of the model - REQUIRED")
    temperature: float = Field(..., ge=0.0, le=2.0, description="Model temperature - REQUIRED from config")
    max_tokens: int = Field(..., ge=1, description="Maximum tokens - REQUIRED from config")
    timeout_seconds: int = Field(..., ge=1, description="Request timeout - REQUIRED from config")
    custom_config: CustomModelConfig = Field(..., description="Custom model configuration - REQUIRED")

    @field_validator('model_name')
    @classmethod
    def validate_model_name_not_placeholder(cls, v: str) -> str:
        """CRITICAL: Reject placeholder model names that indicate missing config."""
        placeholder_names = ['model', 'default', 'placeholder', 'gpt', 'claude', 'ai_model']
        if v.lower().strip() in placeholder_names:
            raise ValueError(f"CRITICAL: Model name '{v}' appears to be a placeholder - provide real model name from config!")
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature_reasonable(cls, v: float) -> float:
        """CRITICAL: Validate temperature is reasonable for financial AI."""
        if v > 1.0:
            import warnings
            warnings.warn(f"WARNING: Temperature {v} is high for financial AI - verify this is intentional!")
        return v
    
class EndpointConfig(BaseModel):
    """Configuration for API endpoints."""
    base_url: str = Field(..., description="Base URL for the API")
    api_version: Optional[str] = Field(None, description="API version")
    custom_endpoints: CustomEndpoints = Field(default_factory=CustomEndpoints, description="Custom endpoints")
    
class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""
    requests_per_minute: int = Field(default=60, ge=1, description="Requests per minute limit")
    requests_per_hour: int = Field(default=3600, ge=1, description="Requests per hour limit")
    burst_limit: int = Field(default=10, ge=1, description="Burst request limit")
    custom_limits: CustomRateLimits = Field(default_factory=CustomRateLimits, description="Custom rate limits")
    

# =============================================================================
# EXPERT SYSTEM CONFIGURATION
# =============================================================================

class ExpertSystemConfig(BaseModel):
    """Configuration for individual expert systems."""
    expert_type: ExpertType = Field(..., description="Type of expert")
    enabled: bool = Field(default=True, description="Whether this expert is enabled")
    model_config: ModelConfig = Field(..., description="Model configuration for this expert")
    api_keys: ApiKeyConfig = Field(default_factory=ApiKeyConfig, description="API keys for this expert")
    endpoints: EndpointConfig = Field(..., description="Endpoint configuration")
    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig, description="Rate limiting configuration")
    security_level: SecurityLevel = Field(default=SecurityLevel.MEDIUM, description="Security level")
    
    # Custom settings for different expert types
    custom_security: CustomSecuritySettings = Field(default_factory=CustomSecuritySettings, description="Custom security settings")
    custom_performance: CustomPerformanceSettings = Field(default_factory=CustomPerformanceSettings, description="Custom performance settings")
    custom_integration: CustomIntegrationSettings = Field(default_factory=CustomIntegrationSettings, description="Custom integration settings")
    custom_agent: CustomAgentSettings = Field(default_factory=CustomAgentSettings, description="Custom agent settings")
    custom_learning: CustomLearningSettings = Field(default_factory=CustomLearningSettings, description="Custom learning settings")
    custom_safety: CustomSafetySettings = Field(default_factory=CustomSafetySettings, description="Custom safety settings")
    custom_insights: CustomInsightSettings = Field(default_factory=CustomInsightSettings, description="Custom insight settings")
    custom_thresholds: CustomThresholdSettings = Field(default_factory=CustomThresholdSettings, description="Custom threshold settings")
    
    model_config = ConfigDict(extra='forbid')


# =============================================================================
# ADAPTIVE LEARNING CONFIGURATION
# =============================================================================

class AnalyticsEngineConfigV2_5(BaseModel):
    """Configuration for the core analytics engine components."""
    metrics_calculation_enabled: bool = Field(True, description="Enable/disable all metric calculations.")
    market_regime_analysis_enabled: bool = Field(True, description="Enable/disable market regime analysis.")
    signal_generation_enabled: bool = Field(True, description="Enable/disable signal generation.")
    key_level_identification_enabled: bool = Field(True, description="Enable/disable key level identification.")
    
    model_config = ConfigDict(extra='forbid')

class AdaptiveLearningConfigV2_5(BaseModel):
    """Configuration for the Adaptive Learning Integration module."""
    auto_adaptation: bool = Field(True, description="Enable/disable automatic application of adaptations.")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence score for an insight to trigger adaptation.")
    pattern_discovery_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence score for an insight to be considered a valid pattern discovery.")
    adaptation_frequency_minutes: int = Field(60, ge=1, description="How often (in minutes) the system checks for new adaptations.")
    analytics_engine: AnalyticsEngineConfigV2_5 = Field(default_factory=lambda: AnalyticsEngineConfigV2_5(), description="Nested configuration for the analytics engine within adaptive learning.")
    
    # Additional config fields from config file
    enabled: Optional[bool] = Field(True, description="Enable adaptive learning")
    learning_rate: Optional[float] = Field(0.01, description="Learning rate for adaptation")
    adaptation_threshold: Optional[float] = Field(0.1, description="Adaptation threshold")

    model_config = ConfigDict(extra='forbid')


# =============================================================================
# PREDICTION CONFIGURATION
# =============================================================================

class ConfidenceCalibration(BaseModel):
    """Placeholder for confidence calibration."""
    calibration_factor: float = Field(default=1.0, description="Calibration factor")
    
class PredictionConfigV2_5(BaseModel):
    """Configuration for the AI Predictions Manager module."""
    enabled: bool = Field(True, description="Enable/disable AI predictions.")
    model_name: str = Field("default_prediction_model", description="Name of the primary prediction model to use.")
    prediction_interval_seconds: int = Field(300, ge=60, description="How often (in seconds) to generate new predictions.")
    max_data_age_seconds: int = Field(120, ge=10, description="Maximum age of market data (in seconds) to be considered fresh for predictions.")
    confidence_calibration: ConfidenceCalibration = Field(default_factory=ConfidenceCalibration, description="Parameters for calibrating confidence scores based on signal strength.")
    success_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum performance score for a prediction to be considered successful.")
    
    # Additional config fields from config file
    min_confidence: Optional[float] = Field(0.6, description="Minimum confidence threshold")

    model_config = ConfigDict(extra='forbid')


# =============================================================================
# MOE (MIXTURE OF EXPERTS) CONFIGURATION
# =============================================================================

# Placeholder classes for MOE detailed configurations
class CustomLoadBalancingFactors(BaseModel):
    """Placeholder for custom load balancing factors."""
    custom_factors: Optional[Dict[str, Any]] = Field(default=None, description="Custom load balancing factors")

class CustomRequestParameters(BaseModel):
    """Placeholder for custom request parameters."""
    custom_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom request parameters")

class RequestContext(BaseModel):
    """Context information for MOE requests."""
    request_id: str = Field(..., description="Unique request identifier")
    user_id: Optional[str] = Field(None, description="User making the request")
    session_id: Optional[str] = Field(None, description="Session identifier")
    priority: str = Field(default="normal", description="Request priority level")
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
    health_score: Optional[float] = Field(default=None, description="Overall health score", ge=0.0, le=1.0)
    custom_factors: Optional[CustomLoadBalancingFactors] = Field(None, description="Custom load balancing factors")

    model_config = ConfigDict(extra='forbid')

class MOERoutingStrategy(str, Enum):
    """Routing strategies for MOE system."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_MATCHED = "capability_matched"

class MOEConsensusStrategy(str, Enum):
    """Consensus strategies for MOE system."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPERT_RANKING = "expert_ranking"
    UNANIMOUS = "unanimous"

class MOEExpertConfig(BaseModel):
    """Configuration for individual experts in MOE system."""
    expert_id: str = Field(..., description="Unique expert identifier")
    expert_type: ExpertType = Field(..., description="Type of expert")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Expert weight in ensemble")
    enabled: bool = Field(default=True, description="Whether expert is enabled")
    model_config: ModelConfig = Field(..., description="Model configuration")
    performance_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum performance threshold")

    model_config = ConfigDict(extra='forbid')

class MOESystemConfig(BaseModel):
    """Configuration for the MOE (Mixture of Experts) system."""
    enabled: bool = Field(default=True, description="Enable/disable MOE system")
    routing_strategy: MOERoutingStrategy = Field(default=MOERoutingStrategy.PERFORMANCE_BASED, description="Strategy for routing requests to experts")
    consensus_strategy: MOEConsensusStrategy = Field(default=MOEConsensusStrategy.CONFIDENCE_WEIGHTED, description="Strategy for reaching consensus among experts")
    min_experts: int = Field(default=2, ge=1, description="Minimum number of experts to consult")
    max_experts: int = Field(default=5, ge=1, description="Maximum number of experts to consult")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold for decisions")
    timeout_seconds: int = Field(default=30, ge=1, description="Timeout for expert responses")
    experts: List[MOEExpertConfig] = Field(default_factory=list, description="List of expert configurations")

    @field_validator('max_experts')
    @classmethod
    def validate_expert_limits(cls, v: int, info: Any) -> int:
        if 'min_experts' in info.data and v < info.data['min_experts']:
            raise ValueError("max_experts must be greater than or equal to min_experts")
        return v

    model_config = ConfigDict(extra='forbid')