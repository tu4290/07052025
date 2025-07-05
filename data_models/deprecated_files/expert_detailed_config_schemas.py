"""Detailed configuration schemas for expert system components.

This module defines Pydantic models to replace Dict[str, Any] patterns
in expert configuration schemas, providing better type safety and validation.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class CustomApiKeys(BaseModel):
    """Custom API key configurations."""
    openai_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_key: Optional[str] = Field(None, description="Anthropic API key")
    google_key: Optional[str] = Field(None, description="Google API key")
    azure_key: Optional[str] = Field(None, description="Azure API key")
    custom_provider_keys: Optional[Dict[str, str]] = Field(default=None, description="Additional provider keys")
    

class CustomModelConfig(BaseModel):
    """Custom model configuration details."""
    model_name: str = Field(..., description="Name of the custom model")
    provider: str = Field(..., description="Model provider")
    endpoint_url: Optional[str] = Field(None, description="Custom endpoint URL")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for this model")
    temperature: Optional[float] = Field(None, description="Temperature setting for this model")
    cost_per_token: Optional[float] = Field(None, description="Cost per token for this model")
    capabilities: Optional[Dict[str, bool]] = Field(default=None, description="Model capabilities")
    

class CustomEndpoints(BaseModel):
    """Custom API endpoint configurations."""
    chat_endpoint: Optional[str] = Field(None, description="Custom chat endpoint")
    embeddings_endpoint: Optional[str] = Field(None, description="Custom embeddings endpoint")
    completion_endpoint: Optional[str] = Field(None, description="Custom completion endpoint")
    streaming_endpoint: Optional[str] = Field(None, description="Custom streaming endpoint")
    additional_endpoints: Optional[Dict[str, str]] = Field(default=None, description="Additional custom endpoints")
    

class CustomRateLimits(BaseModel):
    """Custom rate limit configurations."""
    requests_per_minute: Optional[int] = Field(None, description="Custom requests per minute")
    tokens_per_minute: Optional[int] = Field(None, description="Custom tokens per minute")
    concurrent_requests: Optional[int] = Field(None, description="Custom concurrent requests limit")
    burst_limit: Optional[int] = Field(None, description="Burst limit for requests")
    provider_limits: Optional[Dict[str, int]] = Field(default=None, description="Provider-specific limits")
    

class CustomSecuritySettings(BaseModel):
    """Custom security configuration settings."""
    encryption_algorithm: Optional[str] = Field(None, description="Encryption algorithm to use")
    key_rotation_interval: Optional[int] = Field(None, description="Key rotation interval in hours")
    access_control_enabled: Optional[bool] = Field(None, description="Whether access control is enabled")
    audit_logging: Optional[bool] = Field(None, description="Whether audit logging is enabled")
    ip_whitelist: Optional[list] = Field(None, description="IP whitelist for access")
    security_headers: Optional[Dict[str, str]] = Field(default=None, description="Custom security headers")
    

class CustomPerformanceSettings(BaseModel):
    """Custom performance configuration settings."""
    connection_pool_size: Optional[int] = Field(None, description="Connection pool size")
    request_timeout: Optional[int] = Field(None, description="Request timeout in seconds")
    retry_strategy: Optional[str] = Field(None, description="Retry strategy to use")
    circuit_breaker_enabled: Optional[bool] = Field(None, description="Whether circuit breaker is enabled")
    load_balancing_strategy: Optional[str] = Field(None, description="Load balancing strategy")
    performance_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics configuration")
    

class CustomIntegrationSettings(BaseModel):
    """Custom integration configuration settings."""
    webhook_retries: Optional[int] = Field(None, description="Number of webhook retries")
    webhook_timeout: Optional[int] = Field(None, description="Webhook timeout in seconds")
    database_connection_string: Optional[str] = Field(None, description="Custom database connection")
    monitoring_endpoints: Optional[list] = Field(None, description="Monitoring endpoints")
    notification_channels: Optional[list] = Field(None, description="Notification channels")
    integration_configs: Optional[Dict[str, Any]] = Field(default=None, description="Integration-specific configs")
    

class CustomAgentSettings(BaseModel):
    """Custom agent configuration settings."""
    reasoning_strategy: Optional[str] = Field(None, description="Reasoning strategy to use")
    memory_persistence: Optional[bool] = Field(None, description="Whether to persist memory")
    context_compression: Optional[bool] = Field(None, description="Whether to compress context")
    multi_agent_coordination: Optional[bool] = Field(None, description="Multi-agent coordination enabled")
    tool_usage_limits: Optional[dict] = Field(None, description="Tool usage limits")
    agent_behaviors: Optional[Dict[str, Any]] = Field(default=None, description="Agent behavior configurations")
    

class CustomLearningSettings(BaseModel):
    """Custom learning configuration settings."""
    learning_algorithm: Optional[str] = Field(None, description="Learning algorithm to use")
    data_retention_period: Optional[int] = Field(None, description="Data retention period in days")
    model_versioning: Optional[bool] = Field(None, description="Whether to version models")
    continuous_learning: Optional[bool] = Field(None, description="Continuous learning enabled")
    feedback_processing: Optional[str] = Field(None, description="Feedback processing strategy")
    learning_configs: Optional[Dict[str, Any]] = Field(default=None, description="Learning-specific configurations")
    

class CustomSafetySettings(BaseModel):
    """Custom safety configuration settings."""
    content_moderation_level: Optional[str] = Field(None, description="Content moderation level")
    bias_detection_enabled: Optional[bool] = Field(None, description="Bias detection enabled")
    harmful_content_filters: Optional[list] = Field(None, description="Harmful content filters")
    safety_escalation_rules: Optional[dict] = Field(None, description="Safety escalation rules")
    compliance_standards: Optional[list] = Field(None, description="Compliance standards to follow")
    safety_configs: Optional[Dict[str, Any]] = Field(default=None, description="Safety-specific configurations")
    

class CustomInsightSettings(BaseModel):
    """Custom insight generation settings."""
    insight_algorithms: Optional[list] = Field(None, description="Insight generation algorithms")
    data_sources: Optional[list] = Field(None, description="Data sources for insights")
    insight_validation: Optional[bool] = Field(None, description="Insight validation enabled")
    real_time_insights: Optional[bool] = Field(None, description="Real-time insight generation")
    insight_storage: Optional[str] = Field(None, description="Insight storage strategy")
    insight_configs: Optional[Dict[str, Any]] = Field(default=None, description="Insight-specific configurations")
    

class ThresholdTypes(BaseModel):
    """Different threshold type configurations."""
    confidence_threshold: Optional[float] = Field(None, description="Confidence threshold")
    accuracy_threshold: Optional[float] = Field(None, description="Accuracy threshold")
    performance_threshold: Optional[float] = Field(None, description="Performance threshold")
    quality_threshold: Optional[float] = Field(None, description="Quality threshold")
    risk_threshold: Optional[float] = Field(None, description="Risk threshold")
    additional_thresholds: Optional[Dict[str, float]] = Field(default=None, description="Additional threshold types")
    

class CustomThresholdSettings(BaseModel):
    """Custom threshold configuration settings."""
    dynamic_adjustment: Optional[bool] = Field(None, description="Dynamic threshold adjustment")
    adjustment_frequency: Optional[int] = Field(None, description="Adjustment frequency in minutes")
    threshold_history: Optional[bool] = Field(None, description="Keep threshold history")
    threshold_alerts: Optional[bool] = Field(None, description="Threshold breach alerts")
    threshold_validation: Optional[str] = Field(None, description="Threshold validation strategy")
    threshold_configs: Optional[Dict[str, Any]] = Field(default=None, description="Threshold-specific configurations")