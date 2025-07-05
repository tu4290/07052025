"""Expert Configuration Schemas

This module defines Pydantic models for expert system configurations,
replacing Dict[str, Any] patterns with proper type-safe models.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from .expert_detailed_config_schemas import (
    CustomApiKeys, CustomModelConfig, CustomEndpoints, CustomRateLimits,
    CustomSecuritySettings, CustomPerformanceSettings, CustomIntegrationSettings,
    CustomAgentSettings, CustomLearningSettings, CustomSafetySettings,
    CustomInsightSettings, ThresholdTypes, CustomThresholdSettings
)


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


class ApiKeyConfig(BaseModel):
    """Configuration for API keys."""
    openai_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_key: Optional[str] = Field(None, description="Anthropic API key")
    azure_key: Optional[str] = Field(None, description="Azure OpenAI API key")
    huggingface_key: Optional[str] = Field(None, description="HuggingFace API key")
    custom_keys: CustomApiKeys = Field(default_factory=CustomApiKeys, description="Custom API keys")
    

class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    default_model: str = Field("gpt-4", description="Default model to use")
    fallback_model: str = Field("gpt-3.5-turbo", description="Fallback model if default fails")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(4096, gt=0, description="Maximum tokens per request")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    custom_models: Optional[Dict[str, CustomModelConfig]] = Field(default=None, description="Custom model configurations")
    

class ApiEndpointConfig(BaseModel):
    """Configuration for API endpoints."""
    base_url: str = Field("https://api.openai.com/v1", description="Base API URL")
    chat_endpoint: str = Field("/chat/completions", description="Chat completions endpoint")
    embeddings_endpoint: str = Field("/embeddings", description="Embeddings endpoint")
    timeout: int = Field(30, gt=0, description="Request timeout in seconds")
    retry_attempts: int = Field(3, ge=0, description="Number of retry attempts")
    custom_endpoints: CustomEndpoints = Field(default_factory=CustomEndpoints, description="Custom endpoint configurations")
    

class RateLimitConfig(BaseModel):
    """Configuration for API rate limits."""
    requests_per_minute: int = Field(60, gt=0, description="Requests per minute limit")
    tokens_per_minute: int = Field(90000, gt=0, description="Tokens per minute limit")
    concurrent_requests: int = Field(10, gt=0, description="Maximum concurrent requests")
    backoff_factor: float = Field(2.0, gt=0, description="Exponential backoff factor")
    max_backoff: int = Field(60, gt=0, description="Maximum backoff time in seconds")
    custom_limits: CustomRateLimits = Field(default_factory=CustomRateLimits, description="Custom rate limits")
    

class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    level: SecurityLevel = Field(SecurityLevel.MEDIUM, description="Security level")
    encrypt_keys: bool = Field(True, description="Whether to encrypt API keys")
    use_ssl: bool = Field(True, description="Whether to use SSL/TLS")
    verify_certificates: bool = Field(True, description="Whether to verify SSL certificates")
    allowed_domains: List[str] = Field(default_factory=list, description="Allowed domains for API calls")
    blocked_domains: List[str] = Field(default_factory=list, description="Blocked domains")
    custom_security: CustomSecuritySettings = Field(default_factory=CustomSecuritySettings, description="Custom security settings")
    

class PerformanceConfig(BaseModel):
    """Configuration for performance settings."""
    cache_enabled: bool = Field(True, description="Whether to enable response caching")
    cache_ttl: int = Field(3600, gt=0, description="Cache TTL in seconds")
    batch_size: int = Field(10, gt=0, description="Batch processing size")
    parallel_processing: bool = Field(True, description="Whether to enable parallel processing")
    max_workers: int = Field(4, gt=0, description="Maximum worker threads")
    memory_limit_mb: int = Field(1024, gt=0, description="Memory limit in MB")
    custom_performance: CustomPerformanceSettings = Field(default_factory=CustomPerformanceSettings, description="Custom performance settings")
    

class IntegrationConfig(BaseModel):
    """Configuration for system integrations."""
    database_enabled: bool = Field(True, description="Whether database integration is enabled")
    monitoring_enabled: bool = Field(True, description="Whether monitoring is enabled")
    logging_level: str = Field("INFO", description="Logging level")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for notifications")
    external_apis: List[str] = Field(default_factory=list, description="External APIs to integrate with")
    custom_integrations: CustomIntegrationSettings = Field(default_factory=CustomIntegrationSettings, description="Custom integration settings")
    

class AgentSettings(BaseModel):
    """Configuration for AI agent settings."""
    max_iterations: int = Field(10, gt=0, description="Maximum agent iterations")
    thinking_time: float = Field(1.0, ge=0, description="Thinking time between iterations")
    confidence_threshold: float = Field(0.8, ge=0, le=1, description="Confidence threshold for decisions")
    memory_size: int = Field(1000, gt=0, description="Agent memory size")
    learning_rate: float = Field(0.01, gt=0, description="Learning rate for adaptation")
    custom_agent_settings: CustomAgentSettings = Field(default_factory=CustomAgentSettings, description="Custom agent settings")
    

class LearningSettings(BaseModel):
    """Configuration for learning settings."""
    enabled: bool = Field(True, description="Whether learning is enabled")
    learning_mode: str = Field("adaptive", description="Learning mode")
    feedback_weight: float = Field(0.5, ge=0, le=1, description="Weight for feedback in learning")
    exploration_rate: float = Field(0.1, ge=0, le=1, description="Exploration rate for learning")
    update_frequency: int = Field(100, gt=0, description="Update frequency for learning")
    custom_learning_settings: CustomLearningSettings = Field(default_factory=CustomLearningSettings, description="Custom learning settings")
    

class SafetySettings(BaseModel):
    """Configuration for safety settings."""
    content_filter_enabled: bool = Field(True, description="Whether content filtering is enabled")
    toxicity_threshold: float = Field(0.8, ge=0, le=1, description="Toxicity detection threshold")
    max_response_length: int = Field(4096, gt=0, description="Maximum response length")
    blocked_topics: List[str] = Field(default_factory=list, description="Blocked topics")
    safety_model: str = Field("default", description="Safety model to use")
    custom_safety_settings: CustomSafetySettings = Field(default_factory=CustomSafetySettings, description="Custom safety settings")
    

class InsightGenerationSettings(BaseModel):
    """Configuration for insight generation."""
    enabled: bool = Field(True, description="Whether insight generation is enabled")
    depth_level: int = Field(3, ge=1, le=5, description="Depth level for insights")
    creativity_factor: float = Field(0.7, ge=0, le=1, description="Creativity factor for insights")
    context_window: int = Field(1000, gt=0, description="Context window for insights")
    insight_types: List[str] = Field(default_factory=lambda: ["analytical", "predictive", "strategic"], description="Types of insights to generate")
    custom_insight_settings: CustomInsightSettings = Field(default_factory=CustomInsightSettings, description="Custom insight settings")
    

class AdaptiveThresholds(BaseModel):
    """Configuration for adaptive thresholds."""
    enabled: bool = Field(True, description="Whether adaptive thresholds are enabled")
    base_threshold: float = Field(0.5, ge=0, le=1, description="Base threshold value")
    adaptation_rate: float = Field(0.1, ge=0, le=1, description="Rate of threshold adaptation")
    min_threshold: float = Field(0.1, ge=0, le=1, description="Minimum threshold value")
    max_threshold: float = Field(0.9, ge=0, le=1, description="Maximum threshold value")
    threshold_types: ThresholdTypes = Field(default_factory=ThresholdTypes, description="Different threshold types")
    custom_threshold_settings: CustomThresholdSettings = Field(default_factory=CustomThresholdSettings, description="Custom threshold settings")