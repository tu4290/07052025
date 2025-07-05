"""
Configuration Models for EOTS v2.5 - Modular Structure

This module provides a unified interface to all configuration models
by importing from the specialized configuration modules.

Modular structure:
- core_system_config.py: Core system, dashboard, and data management
- expert_ai_config.py: Expert systems, MOE, and AI configurations  
- trading_analytics_config.py: Trading parameters, performance, and analytics
- learning_intelligence_config.py: Learning systems, HuiHui, and intelligence

This maintains backward compatibility while providing better organization.
"""

# =============================================================================
# IMPORTS FROM MODULAR CONFIGURATION FILES
# =============================================================================

# Core System Configuration
from .core_system_config import (
    # Dashboard & UI
    RegimeIndicatorConfig,
    FlowGaugeConfig,
    GibGaugeConfig,
    MiniHeatmapConfig,
    RecommendationsTableConfig,
    TickerContextConfig,
    DashboardDefaults,
    DashboardServerConfig,
    SignalActivationSettings,
    DashboardModeSettings,
    MainDashboardDisplaySettings,
    DashboardModeCollection,
    VisualizationSettings,
    
    # System Settings
    SystemSettings,
    
    # Data Management
    ApiKeysSettings,
    ConvexValueAuthSettings,
    DataFetcherSettings,
    DataManagementSettings,
    DatabaseSettings,
    
    # Control Panel & Collector
    ControlPanelParametersV2_5,
    IntradayCollectorSettings,
)

# Expert & AI Configuration
from .expert_ai_config import (
    # Enums
    ModelProvider,
    SecurityLevel,
    ExpertType,
    
    # API & Model Configuration
    ApiKeyConfig,
    ModelConfig,
    EndpointConfig,
    RateLimitConfig,
    
    # Expert System Configuration
    ExpertSystemConfig,
    
    # Adaptive Learning
    AnalyticsEngineConfigV2_5,
    AdaptiveLearningConfigV2_5,
    
    # Prediction Configuration
    ConfidenceCalibration,
    PredictionConfigV2_5,
    
    # MOE Configuration
    RequestContext,
    LoadBalancingFactors,
    MOERoutingStrategy,
    MOEConsensusStrategy,
    MOEExpertConfig,
    MOESystemConfig,
)

# Trading & Analytics Configuration
from .trading_analytics_config import (
    # Data Processor
    DataProcessorFactors,
    IVContextParameters,
    DataProcessorSettings,
    
    # Market Regime Engine
    RegimeRuleConditions,
    RegimeRules,
    MarketRegimeEngineSettings,
    
    # Enhanced Flow Metrics
    VAPIFAParameters,
    DWFDParameters,
    TWLAFParameters,
    EnhancedFlowMetricSettings,
    
    # Signal Integration
    SignalIntegrationParameters,
    ConvictionMappingParameters,
    
    # Trading Strategy
    StrategySpecificRule,
    IntelligentRecommendationManagementRules,
    ContractSelectionFilters,
    StopLossCalculationRules,
    ProfitTargetCalculationRules,
    
    # Performance Tracking
    PerformanceMetadata,
    StrategyParameters,
    PerformanceTrackerSettingsV2_5,
    
    # Adaptive Metrics
    TickerSpecificParameters,
    AdaptiveMetricParameters,
)

# Learning & Intelligence Configuration
from .learning_intelligence_config import (
    # Learning System
    MarketContextData,
    AdaptationSuggestion,
    PerformanceMetricsSnapshot,
    LearningInsightData,
    ExpertAdaptationSummary,
    ConfidenceUpdateData,
    LearningSystemConfig,
    
    # HuiHui AI System
    AnalysisContext,
    RequestMetadata,
    EOTSPrediction,
    TradingRecommendation,
    PerformanceByCondition,
    HuiHuiSystemConfig,
    
    # Intelligence Framework
    IntelligenceFrameworkConfig,
)

from pydantic import BaseModel, Field, ConfigDict, field_validator, FieldValidationInfo

# Elite Intelligence Configuration
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_analytics_engine', 'eots_metrics'))
    from elite_intelligence import EliteConfig
except ImportError:
    # Fallback if elite_intelligence module not available
    class EliteConfig(BaseModel):
        """Fallback EliteConfig model"""
        enable_elite_regime_detection: bool = Field(True, description="Enable elite regime detection")
        elite_regime_threshold: float = Field(0.7, description="Elite regime threshold")
        model_config = ConfigDict(extra='allow')


# =============================================================================
# ADDITIONAL CONFIGURATION MODELS
# =============================================================================

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict, field_validator, FieldValidationInfo

class RegimeContextWeightMultipliers(BaseModel):
    """Multipliers for regime context weighting - FAIL FAST ON MISSING CONFIG."""
    bullish_multiplier: float = Field(..., gt=0.0, description="Bullish regime multiplier - REQUIRED from config")
    bearish_multiplier: float = Field(..., gt=0.0, description="Bearish regime multiplier - REQUIRED from config")
    neutral_multiplier: float = Field(..., gt=0.0, description="Neutral regime multiplier - REQUIRED from config")

    @field_validator('bullish_multiplier', 'bearish_multiplier', 'neutral_multiplier')
    @classmethod
    def validate_multipliers_reasonable(cls, v: float, info: FieldValidationInfo) -> float:
        """CRITICAL: Validate multipliers are reasonable for financial calculations."""
        if v <= 0.0:
            raise ValueError(f"CRITICAL: {info.field_name} must be positive!")
        if v > 10.0:
            import warnings
            warnings.warn(f"WARNING: {info.field_name} = {v} is very high - verify this is intentional!")
        return v
    
    model_config = ConfigDict(extra='forbid')

class LearningParams(BaseModel):
    """Parameters for learning systems."""
    performance_tracker_query_lookback: int = Field(90, ge=1, description="Lookback days for performance tracking")
    learning_rate_for_signal_weights: float = Field(0.05, ge=0, le=1, description="Learning rate for signal weights")
    learning_rate_for_target_adjustments: float = Field(0.02, ge=0, le=1, description="Learning rate for target adjustments")
    min_trades_for_statistical_significance: int = Field(20, ge=1, description="Minimum trades for statistical significance")
    
    model_config = ConfigDict(extra='forbid')

class AdaptiveTradeIdeaFrameworkSettings(BaseModel):
    """Settings for the Adaptive Trade Idea Framework (ATIF)."""
    min_conviction_to_initiate_trade: float = Field(2.5, ge=0, le=5, description="Minimum conviction score to initiate trade")
    signal_integration_params: SignalIntegrationParameters = Field(default_factory=SignalIntegrationParameters, description="Signal integration parameters")
    regime_context_weight_multipliers: RegimeContextWeightMultipliers = Field(default_factory=RegimeContextWeightMultipliers, description="Regime context weight multipliers")
    conviction_mapping_params: ConvictionMappingParameters = Field(default_factory=ConvictionMappingParameters, description="Conviction mapping parameters")
    strategy_specificity_rules: List[StrategySpecificRule] = Field(default_factory=list, description="Strategy-specific rules")
    intelligent_recommendation_management_rules: IntelligentRecommendationManagementRules = Field(default_factory=IntelligentRecommendationManagementRules, description="Intelligent recommendation management rules")
    learning_params: LearningParams = Field(default_factory=LearningParams, description="Learning parameters")
    
    # Additional config fields from config file
    enabled: Optional[bool] = Field(True, description="Enable ATIF framework")
    min_conviction_threshold: Optional[float] = Field(2.5, description="Minimum conviction threshold (alternative field name)")
    # Removed dictionary fields - use proper Pydantic models instead of Dict[str, Any]
    # signal_integration_parameters and conviction_mapping_parameters should be defined as proper Pydantic models

    model_config = ConfigDict(extra='allow')

class TradeParameterOptimizerSettings(BaseModel):
    """Settings for the Trade Parameter Optimizer."""
    contract_selection_filters: ContractSelectionFilters = Field(default_factory=ContractSelectionFilters, description="Contract selection filters")
    entry_price_logic: str = Field("MID_PRICE", description="Entry price logic")
    stop_loss_calculation_rules: StopLossCalculationRules = Field(default_factory=StopLossCalculationRules, description="Stop loss calculation rules")
    profit_target_calculation_rules: ProfitTargetCalculationRules = Field(default_factory=ProfitTargetCalculationRules, description="Profit target calculation rules")
    
    def to_dict(self) -> Dict[str, Any]: # Keep for now, or use model_dump()
        # Ensure nested Pydantic models are also converted if this method is kept
        return self.model_dump()

    # Additional config fields from config file
    enabled: Optional[bool] = Field(True, description="Enable trade parameter optimizer")
    optimization_interval_seconds: Optional[int] = Field(300, description="Optimization interval in seconds")

    model_config = ConfigDict(extra='allow')


# =============================================================================
# ROOT CONFIGURATION MODEL
# =============================================================================

class EOTSConfigV2_5(BaseModel):
    """
    The root model for the EOTS v2.5 system configuration.
    Provides a unified interface to all configuration sections.
    """
    # Core System Settings - TIER 2: VALIDATED DEFAULTS (System-level, not trading-critical)
    system_settings: SystemSettings = Field(
        default_factory=SystemSettings,
        description="System settings - uses reasonable defaults if not in config"
    )
    data_fetcher_settings: DataFetcherSettings = Field(..., description="Data fetcher settings - REQUIRED from config")
    data_management_settings: DataManagementSettings = Field(
        default_factory=DataManagementSettings,
        description="Data management settings - uses reasonable defaults if not in config"
    )
    database_settings: Optional[DatabaseSettings] = Field(None, description="Optional database settings")
    visualization_settings: VisualizationSettings = Field(
        default_factory=VisualizationSettings,
        description="Visualization settings - uses reasonable defaults if not in config"
    )

    @field_validator('system_settings')
    @classmethod
    def validate_system_settings(cls, v: SystemSettings) -> SystemSettings:
        """INFO: Using default system settings - verify these are appropriate for your environment."""
        import warnings
        warnings.warn("INFO: Using default system settings - review config file to customize if needed.")
        return v
    
    # Data Processing & Analytics - TIER 1: FAIL-FAST FOR TRADING-CRITICAL CONFIG
    data_processor_settings: DataProcessorSettings = Field(..., description="Data processor settings - REQUIRED from config (affects all calculations)")
    market_regime_engine_settings: MarketRegimeEngineSettings = Field(..., description="Market regime engine settings - REQUIRED from config (trading-critical)")
    enhanced_flow_metric_settings: EnhancedFlowMetricSettings = Field(..., description="Enhanced flow metric settings - REQUIRED from config (trading-critical)")

    @field_validator('market_regime_engine_settings')
    @classmethod
    def validate_regime_engine_config(cls, v: MarketRegimeEngineSettings) -> MarketRegimeEngineSettings:
        """CRITICAL: Ensure market regime engine configuration is explicitly provided."""
        import warnings
        warnings.warn("CRITICAL: Verify market regime engine settings match your trading environment!")
        return v

    @field_validator('enhanced_flow_metric_settings')
    @classmethod
    def validate_flow_metrics_config(cls, v: EnhancedFlowMetricSettings) -> EnhancedFlowMetricSettings:
        """CRITICAL: Ensure flow metrics configuration is explicitly provided."""
        import warnings
        warnings.warn("CRITICAL: Verify flow metrics settings are calibrated for current market conditions!")
        return v
    adaptive_metric_parameters: AdaptiveMetricParameters = Field(default_factory=AdaptiveMetricParameters)
    
    # Trading & Strategy - TIER 1: FAIL-FAST FOR TRADING-CRITICAL CONFIG
    adaptive_trade_idea_framework_settings: AdaptiveTradeIdeaFrameworkSettings = Field(..., description="ATIF settings - REQUIRED from config (trading-critical)")
    trade_parameter_optimizer_settings: TradeParameterOptimizerSettings = Field(..., description="TPO settings - REQUIRED from config (trading-critical)")

    @field_validator('adaptive_trade_idea_framework_settings')
    @classmethod
    def validate_atif_config_not_default(cls, v: AdaptiveTradeIdeaFrameworkSettings) -> AdaptiveTradeIdeaFrameworkSettings:
        """CRITICAL: Ensure ATIF configuration is explicitly provided, not defaults."""
        # Add validation logic here when ATIF model is fully defined
        import warnings
        warnings.warn("CRITICAL: Verify ATIF settings are appropriate for your trading strategy!")
        return v

    @field_validator('trade_parameter_optimizer_settings')
    @classmethod
    def validate_tpo_config_not_default(cls, v: TradeParameterOptimizerSettings) -> TradeParameterOptimizerSettings:
        """CRITICAL: Ensure TPO configuration is explicitly provided, not defaults."""
        # Add validation logic here when TPO model is fully defined
        import warnings
        warnings.warn("CRITICAL: Verify TPO settings are appropriate for your risk management!")
        return v
    performance_tracker_settings: PerformanceTrackerSettingsV2_5 = Field(default_factory=PerformanceTrackerSettingsV2_5)
    
    # AI & Intelligence - TIER 1: FAIL-FAST FOR TRADING-CRITICAL AI CONFIG
    expert_system_config: Optional[ExpertSystemConfig] = Field(None, description="Expert system configuration - optional")
    moe_system_config: Optional[MOESystemConfig] = Field(None, description="MOE system configuration - optional")
    adaptive_learning_config: AdaptiveLearningConfigV2_5 = Field(..., description="Adaptive learning config - REQUIRED from config (affects trading intelligence)")
    prediction_config: PredictionConfigV2_5 = Field(..., description="Prediction config - REQUIRED from config (affects trading predictions)")
    intelligence_framework_config: IntelligenceFrameworkConfig = Field(..., description="Intelligence framework config - REQUIRED from config (affects trading intelligence)")

    @field_validator('adaptive_learning_config')
    @classmethod
    def validate_adaptive_learning_config(cls, v: AdaptiveLearningConfigV2_5) -> AdaptiveLearningConfigV2_5:
        """CRITICAL: Ensure adaptive learning configuration is explicitly provided."""
        import warnings
        warnings.warn("CRITICAL: Verify adaptive learning settings are appropriate for your trading strategy!")
        return v

    @field_validator('prediction_config')
    @classmethod
    def validate_prediction_config(cls, v: PredictionConfigV2_5) -> PredictionConfigV2_5:
        """CRITICAL: Ensure prediction configuration is explicitly provided."""
        import warnings
        warnings.warn("CRITICAL: Verify prediction settings are calibrated for current market conditions!")
        return v
    
    # Optional Components
    intraday_collector_settings: Optional[IntradayCollectorSettings] = Field(None, description="Intraday collector settings")

    # Additional Configuration Sections - TIER 3: SMART DEFAULTS (System-level, reasonable defaults)
    strategy_settings: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Strategy settings - will use reasonable defaults if not provided in config"
    )
    ticker_context_analyzer_settings: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Ticker context analyzer settings - will use reasonable defaults if not provided in config"
    )
    key_level_identifier_settings: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Key level identifier settings - will use reasonable defaults if not provided in config"
    )
    heatmap_generation_settings: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Heatmap generation settings - will use reasonable defaults if not provided in config"
    )
    symbol_specific_overrides: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Symbol-specific overrides - will use reasonable defaults if not provided in config"
    )
    performance_tracker_settings_v2_5: Optional[PerformanceTrackerSettingsV2_5] = Field(
        default_factory=PerformanceTrackerSettingsV2_5,
        description="Performance tracker settings v2.5 - will use reasonable defaults if not provided in config"
    )
    elite_config: Optional[EliteConfig] = Field(
        default_factory=EliteConfig,
        description="Elite configuration - will use reasonable defaults if not provided in config"
    )
    time_of_day_definitions: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Time of day definitions - will use reasonable defaults if not provided in config"
    )

    def to_dict(self) -> Dict[str, Any]: # Keep for now, or use model_dump()
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True) # arbitrary_types_allowed for EliteConfig fallback if it's not Pydantic


# =============================================================================
# BACKWARD COMPATIBILITY EXPORTS
# =============================================================================

# Export all models for backward compatibility
__all__ = [
    # Root configuration
    'EOTSConfigV2_5',
    
    # Core system models
    'SystemSettings', 'DataFetcherSettings', 'DataManagementSettings', 'DatabaseSettings',
    'VisualizationSettings', 'DashboardModeSettings', 'MainDashboardDisplaySettings', 'DashboardDefaults',
    'IntradayCollectorSettings',
    
    # Expert & AI models
    'ExpertSystemConfig', 'MOESystemConfig', 'AnalyticsEngineConfigV2_5', 'AdaptiveLearningConfigV2_5', 'PredictionConfigV2_5',
    
    # Trading & analytics models
    'DataProcessorSettings', 'MarketRegimeEngineSettings', 'EnhancedFlowMetricSettings',
    'AdaptiveTradeIdeaFrameworkSettings', 'TradeParameterOptimizerSettings', 'PerformanceTrackerSettingsV2_5',
    
    # Learning & intelligence models
    'LearningSystemConfig', 'HuiHuiSystemConfig', 'IntelligenceFrameworkConfig',
    
    # Additional models
    'RegimeContextWeightMultipliers', 'LearningParams', 'EliteConfig',
]