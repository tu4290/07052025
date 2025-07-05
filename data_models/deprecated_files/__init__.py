"""
Data Models for EOTS v2.5

This package contains all Pydantic models used throughout the system,
now organized into 6 consolidated modules for better maintainability.

CONSOLIDATED STRUCTURE:
- core_models: Base types, system state, raw/processed data, bundles, advanced metrics
- configuration_models: All configuration schemas and settings
- ai_ml_models: AI/ML, MOE, learning, and performance models  
- trading_market_models: Trading, market context, signals, recommendations
- dashboard_ui_models: Dashboard and UI component models
- validation_utils: Validation utilities and helper functions
"""

# Import all models from consolidated files to maintain backward compatibility
from .core_models import *
from .configuration_models import *
from .core_system_config import *
# from .ai_ml_models import *  # Commented out to avoid syntax error from truncated ai_ml_models.py
from .trading_market_models import *
from .dashboard_ui_models import *
from .validation_utils import *
from .impact_analysis_schemas import *
from .context_schemas import *
from .learning_schemas import *

__all__ = [
    # From core_models
    "DataFrameSchema", "PandasDataFrame", "SystemStateV2_5", "AISystemHealthV2_5", "AuditLogEntry",
    "RawOptionsContractV2_5", "RawUnderlyingDataV2_5", "RawUnderlyingDataCombinedV2_5", "UnprocessedDataBundleV2_5",
    "ProcessedContractMetricsV2_5", "ProcessedStrikeLevelMetricsV2_5", "ProcessedUnderlyingAggregatesV2_5", "ProcessedDataBundleV2_5",
    "FinalAnalysisBundleV2_5", "UnifiedIntelligenceAnalysis", "AdvancedOptionsMetricsV2_5", "NormalizationParams",

    # From configuration_models
    "EOTSConfigV2_5", "AnalyticsEngineConfigV2_5", "AdaptiveLearningConfigV2_5", "MarketRegimeEngineSettings", "IntradayCollectorSettings",
    "TickerContextAnalyzerSettings",

    # From core_system_config
    "DashboardModeSettings", "DashboardModeCollection", "VisualizationSettings",

    # From ai_ml_models (which might re-export some learning_schemas models, set will deduplicate)
    "AIAdaptationV2_5", "AIAdaptationPerformanceV2_5", "AIPredictionV2_5", "AIPredictionPerformanceV2_5", "AIPredictionRequestV2_5",
    "ExpertStatus", "RoutingStrategy", "ConsensusStrategy", "AgreementLevel", "HealthStatus",
    "MOEExpertRegistryV2_5", "MOEGatingNetworkV2_5", "MOEExpertResponseV2_5", "MOEUnifiedResponseV2_5",
    "LearningInsightV2_5", "UnifiedLearningResult", "MarketPattern", "PatternThresholds", "PerformanceInterval", "PerformanceMetricType", "PerformanceMetricV2_5",
    "SystemPerformanceV2_5", "BacktestPerformanceV2_5", "ExecutionMetricsV2_5", "PerformanceReportV2_5", "PerformanceMetrics",
    "HuiHuiExpertType", "HuiHuiModelConfigV2_5", "HuiHuiExpertConfigV2_5", "HuiHuiAnalysisRequestV2_5",
    "HuiHuiAnalysisResponseV2_5", "HuiHuiUsageRecordV2_5", "HuiHuiPerformanceMetricsV2_5", "HuiHuiEnsembleConfigV2_5", "HuiHuiUserFeedbackV2_5",
    "MarketIntelligencePattern", "MCPIntelligenceResultV2_5", "MCPToolResultV2_5", "AdaptiveLearningResult", "RecursiveIntelligenceResult",
    "SentimentDataV2_5", "NewsArticleV2_5", "AIPredictionMetricsV2_5", "LearningBatchV2_5", "EnhancedLearningMetricsV2_5",
    
    # From trading_market_models
    "SignalPayloadV2_5", "KeyLevelV2_5", "KeyLevelsDataV2_5",
    "TradeParametersV2_5", "ActiveRecommendationPayloadV2_5", "ATIFSituationalAssessmentProfileV2_5", "ATIFStrategyDirectivePayloadV2_5", "ATIFManagementDirectiveV2_5",
    "ConsolidatedAnalysisRequest", "SuperiorTradeIntelligence",
    
    # From dashboard_ui_models
    "DashboardModeType", "ChartType", "DashboardModeUIDetail", "ChartLayoutConfigV2_5", "ControlPanelParametersV2_5",
    "DashboardConfigV2_5", "ComponentComplianceV2_5", "DashboardStateV2_5", "DashboardServerConfig",

    # From impact_analysis_schemas
    "EliteImpactResult",

    # From context_schemas
    "MarketRegimeState",
    "TickerContextDictV2_5",
    "TimeOfDayDefinitions",
    "DynamicThresholdsV2_5",
    "TickerProfile",
    "MarketContext",
    "CorrelationAnalysis",
    "VolatilityProfile",
    "TickerContextAnalysis",

    # From learning_schemas (explicitly adding all, including newly moved ones)
    "EOTSLearningContext",
    "EOTSPredictionOutcome",
    "AILearningPattern",
    "AILearningInsight",
    "AIPredictionRecord",
    "AILearningStats",
    # (Existing models like LearningInsightV2_5, UnifiedLearningResult etc. from learning_schemas are also included via its star import)
]

# Ensure __all__ has unique, sorted entries
__all__ = sorted(list(set(__all__)))