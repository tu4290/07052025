# Essential imports only - but more comprehensive
from .configuration_models import (
    AdaptiveLearningConfigV2_5,
    AnalyticsEngineConfigV2_5,
    PredictionConfigV2_5,
    EOTSConfigV2_5
)
from .dashboard_config_models import DashboardServerConfig
from .core_models import (
    EOTSBaseModel,
    MarketRegimeState,
    KeyLevelV2_5,
    KeyLevelsDataV2_5,
    ProcessedDataBundleV2_5,
    FinalAnalysisBundleV2_5,
    RawOptionsContractV2_5,
    RawUnderlyingDataCombinedV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    ProcessedStrikeLevelMetricsV2_5,
    ProcessedContractMetricsV2_5,
    UnprocessedDataBundleV2_5,
    TradeParametersV2_5,
    ActiveRecommendationPayloadV2_5,
    UnifiedIntelligenceAnalysis
)
from .ai_ml_models import (
    MOEUnifiedResponseV2_5,
    MOEExpertResponseV2_5,
    ExpertAnalysisRequest,
    ExpertAnalysisResponse,
    MarketRegimeAnalysisDetails,
    OptionsFlowAnalysisDetails,
    SentimentAnalysisDetails,
    MarketPattern,
    PatternThresholds,
    RealTimeIntelligenceSummaryV2_5,
    AIAdaptationV2_5,
    AIAdaptationPerformanceV2_5,
    UnifiedLearningResult,
    HuiHuiExpertConfigV2_5,
    HuiHuiExpertType,
    MOEGatingNetworkV2_5,
    MarketIntelligencePattern,
    AdaptiveLearningResult,
    RecursiveIntelligenceResult,
    MCPIntelligenceResultV2_5,
    MCPToolResultV2_5
)
from .ai_dashboard_models import (
    MarketCompassModel,
    MarketCompassSegment,
    PanelConfigModel,
    ComponentStatus,
    PanelType,
    CompassTheme
)
from .huihui_learning_schemas import (
    ExpertKnowledgeBase,
    ExpertConfiguration,
    ExpertPerformanceHistory,
    LearningCycle,
    PredictionOutcome,
    FeedbackLoop
)

# Quick temporary classes for dashboard to run
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class MainDashboardDisplaySettings(BaseModel):
    theme: str = Field(default='dark', description='Dashboard theme')
    auto_refresh: bool = Field(default=True, description='Auto refresh enabled')
    refresh_interval: int = Field(default=5, description='Refresh interval in seconds')

class DashboardModeSettings(BaseModel):
    current_mode: str = Field(default='ai_dashboard', description='Current dashboard mode')
    available_modes: list = Field(default_factory=list, description='Available dashboard modes')

# Aliases for backward compatibility
RoutingDecision = MOEGatingNetworkV2_5
