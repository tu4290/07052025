from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

# =============================================================================
# 1. STUB CLASSES FOR DEPENDENCIES
# Create minimal, empty classes to satisfy imports across the system.
# =============================================================================

class StubConfig(BaseModel):
    """A generic stub model that allows any extra fields."""
    model_config = ConfigDict(extra='allow')

# Stubs for Core System Configs
class SystemSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

class DataFetcherSettings(BaseModel):
    tradier_api_key: Optional[str] = None
    tradier_account_id: Optional[str] = None
    timeout_seconds: float = 30.0
    model_config = ConfigDict(extra='allow')

class DataManagementSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

class DatabaseSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

from data_models.dashboard_config_models import DashboardServerConfig

from .dashboard_config_models import DashboardServerConfig

from .dashboard_config_models import DashboardServerConfig

class VisualizationSettings(BaseModel):
    dashboard: DashboardServerConfig = Field(default_factory=DashboardServerConfig)
    model_config = ConfigDict(extra='allow')

class DashboardModeSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

class MainDashboardDisplaySettings(BaseModel):
    model_config = ConfigDict(extra='allow')

class DashboardDefaults(BaseModel):
    model_config = ConfigDict(extra='allow')

class IntradayCollectorSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

# Stubs for Expert & AI Configs
class ExpertSystemConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

class MOESystemConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

class AnalyticsEngineConfigV2_5(BaseModel):
    model_config = ConfigDict(extra='allow')

class AdaptiveLearningConfigV2_5(BaseModel):
    model_config = ConfigDict(extra='allow')

class PredictionConfigV2_5(BaseModel):
    model_config = ConfigDict(extra='allow')

# Stubs for Trading & Analytics Configs
class DataProcessorSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

class MarketRegimeEngineSettings(BaseModel):
    default_regime: str = Field("regime_unclear_or_transitioning")
    regime_evaluation_order: List[str] = Field(default_factory=list)
    regime_rules: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra='allow')

class EnhancedFlowMetricSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

class AdaptiveTradeIdeaFrameworkSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

class TradeParameterOptimizerSettings(BaseModel):
    model_config = ConfigDict(extra='allow')

class PerformanceTrackerSettingsV2_5(BaseModel):
    performance_data_directory: str = Field("data_cache_v2_5/performance_data_store", description="Directory for storing performance data.")
    model_config = ConfigDict(extra='allow')

# Stubs for Learning & Intelligence Configs
class LearningSystemConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

class HuiHuiSystemConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

class IntelligenceFrameworkConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

# Stub for Elite Config
class EliteConfig(BaseModel):
    """Minimal EliteConfig to satisfy dependencies."""
    model_config = ConfigDict(extra='allow')

# Stub for HuiHui Config
class HuiHuiConfig(BaseModel):
    """Minimal HuiHuiConfig to satisfy dependencies."""
    model_config = ConfigDict(extra='allow')


# =============================================================================
# 2. ROOT CONFIGURATION MODEL
# The main class that the system expects to find.
# =============================================================================

class EOTSConfigV2_5(BaseModel):
    """
    The root model for the EOTS v2.5 system configuration.
    This is a minimal version to allow the system to boot.
    It expects other configuration sections to be present but allows them
    to be flexible dictionaries for now.
    """
    # Use Dict[str, Any] for flexibility during this debug phase.
    # This allows the config manager to load the JSON without needing
    # every single sub-model to be perfectly defined yet.
    system_settings: SystemSettings = Field(default_factory=SystemSettings)
    data_fetcher_settings: DataFetcherSettings = Field(default_factory=DataFetcherSettings)
    data_management_settings: DataManagementSettings = Field(default_factory=DataManagementSettings)
    database_settings: Optional[DatabaseSettings] = None
    visualization_settings: VisualizationSettings = Field(default_factory=VisualizationSettings)
    data_processor_settings: DataProcessorSettings = Field(default_factory=DataProcessorSettings)
    market_regime_engine_settings: MarketRegimeEngineSettings = Field(default_factory=MarketRegimeEngineSettings)
    enhanced_flow_metric_settings: EnhancedFlowMetricSettings = Field(default_factory=EnhancedFlowMetricSettings)
    adaptive_trade_idea_framework_settings: AdaptiveTradeIdeaFrameworkSettings = Field(default_factory=AdaptiveTradeIdeaFrameworkSettings)
    trade_parameter_optimizer_settings: TradeParameterOptimizerSettings = Field(default_factory=TradeParameterOptimizerSettings)
    performance_tracker_settings: PerformanceTrackerSettingsV2_5 = Field(default_factory=PerformanceTrackerSettingsV2_5)
    expert_system_config: Optional[ExpertSystemConfig] = None
    moe_system_config: Optional[MOESystemConfig] = None
    adaptive_learning_config: AdaptiveLearningConfigV2_5 = Field(default_factory=AdaptiveLearningConfigV2_5)
    prediction_config: PredictionConfigV2_5 = Field(default_factory=PredictionConfigV2_5)
    intelligence_framework_config: IntelligenceFrameworkConfig = Field(default_factory=IntelligenceFrameworkConfig)
    huihui_settings: Optional[HuiHuiConfig] = None
    intraday_collector_settings: Optional[IntradayCollectorSettings] = None
    elite_config: Optional[EliteConfig] = None

    model_config = ConfigDict(extra='allow')

# =============================================================================
# 3. EXPORTS
# Make the essential classes available for import.
# =============================================================================

__all__ = [
    'EOTSConfigV2_5',
    'SystemSettings',
    'DataFetcherSettings',
    'DataManagementSettings',
    'DatabaseSettings',
    'VisualizationSettings',
    'DashboardModeSettings',
    'MainDashboardDisplaySettings',
    'DashboardDefaults',
    'IntradayCollectorSettings',
    'ExpertSystemConfig',
    'MOESystemConfig',
    'AnalyticsEngineConfigV2_5',
    'AdaptiveLearningConfigV2_5',
    'PredictionConfigV2_5',
    'DataProcessorSettings',
    'MarketRegimeEngineSettings',
    'EnhancedFlowMetricSettings',
    'AdaptiveTradeIdeaFrameworkSettings',
    'TradeParameterOptimizerSettings',
    'PerformanceTrackerSettingsV2_5',
    'LearningSystemConfig',
    'HuiHuiSystemConfig',
    'IntelligenceFrameworkConfig',
    'EliteConfig',
    'HuiHuiConfig',
]