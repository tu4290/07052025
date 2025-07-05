"""
Config management for EOTS (ConfigManagerV2_5) and HuiHui/MOE (HuiHuiConfigV2_5).
- EOTS config is loaded/validated via ConfigManagerV2_5 (singleton, uses EOTSConfigV2_5).
- HuiHui/MOE config is loaded/validated via HuiHuiConfigV2_5 and load_huihui_config.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

from pydantic import ValidationError, BaseModel, Field, ConfigDict
from ...data_models.expert_config_schemas import (
    ApiKeyConfig, ModelConfig, ApiEndpointConfig, RateLimitConfig,
    SecurityConfig, PerformanceConfig, IntegrationConfig,
    AgentSettings, LearningSettings, SafetySettings,
    InsightGenerationSettings, AdaptiveThresholds
)

# Import the main ConfigManagerV2_5 instead of duplicating it
from utils.config_manager_v2_5 import ConfigManagerV2_5

__all__ = [
    "ConfigManagerV2_5",
    "HuiHuiConfigV2_5",
    "load_huihui_config"
]

logger = logging.getLogger(__name__)


class LLMApiModel(BaseModel):
    enabled: Optional[bool] = None
    ollama_host: Optional[str] = None
    api_version: Optional[str] = None
    api_keys: ApiKeyConfig = Field(default_factory=ApiKeyConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    api_endpoints: ApiEndpointConfig = Field(default_factory=ApiEndpointConfig)
    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)


class AISettingsModel(BaseModel):
    enabled: Optional[bool] = None
    model_config = ConfigDict(extra='allow')
    agent_settings: AgentSettings = Field(default_factory=AgentSettings)
    learning_settings: LearningSettings = Field(default_factory=LearningSettings)
    safety_settings: SafetySettings = Field(default_factory=SafetySettings)
    insight_generation: InsightGenerationSettings = Field(default_factory=InsightGenerationSettings)
    adaptive_thresholds: AdaptiveThresholds = Field(default_factory=AdaptiveThresholds)


class HuiHuiConfigV2_5(BaseModel):
    llm_api: Optional[LLMApiModel] = None
    ai_settings: Optional[AISettingsModel] = None
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)


def load_huihui_config(
    config_path: Optional[Union[str, Path]] = None
) -> HuiHuiConfigV2_5:
    """
    Load and validate the HuiHui/MOE config from JSON file.
    Args:
        config_path: Path to huihui_config.json (default: project_root/config/huihui_config.json)
    Returns:
        Validated HuiHuiConfigV2_5 instance
    Raises:
        RuntimeError if file not found or validation fails
    """
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "huihui_config.json"
    elif not isinstance(config_path, Path):
        config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(
            f"HuiHui config file not found at {config_path}, returning default config."
        )
        return HuiHuiConfigV2_5()
    with open(str(config_path), 'r') as f:
        config_data = json.load(f)
    try:
        return HuiHuiConfigV2_5(**config_data)
    except ValidationError as e:
        logger.error(f"HuiHui config validation failed: {e}")
        raise RuntimeError(f"HuiHui config validation failed: {e}")