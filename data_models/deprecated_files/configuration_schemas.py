""""
Pydantic models defining the structure for various configuration sections
of the EOTS v2.5 system, including system settings, dashboard modes,
coefficients, and parameters for different analytical components.
These models are primarily used by ConfigManagerV2_5 to validate and provide
access to the system's configuration.
"""
from pydantic import BaseModel
from pydantic import Field
from pydantic import FilePath
from pydantic import field_validator, ConfigDict, FieldValidationInfo
from typing import List, Dict, Any, Optional, Union
from data_models.ui_component_schemas import (
    RegimeIndicatorConfig, FlowGaugeConfig, GibGaugeConfig, 
    MiniHeatmapConfig, RecommendationsTableConfig, TickerContextConfig,
    DashboardServerConfig, SignalActivationSettings
)
from data_models.configuration_detailed_schemas import (
    DataProcessorFactors, IVContextParameters, RegimeRules, VAPIFAParameters,
    DWFDParameters, TWLAFParameters, SignalIntegrationParameters, ConvictionMappingParameters,
    StrategySpecificRule, IntelligentRecommendationManagementRules, TickerSpecificParameters,
    UGCHParameters, SGDHPParameters, IVSDHParameters, ADAGSettings, ESDAGSettings,
    DTDPISettings, VRI2Settings, EnhancedHeatmapSettings, ContractSelectionFilters,
    StopLossCalculationRules, ProfitTargetCalculationRules, ConfidenceCalibration,
    TimeOfDayDefinitions, StrategyParameters,
    RegimeContextWeightMultipliers, EnhancedHeatmapSettingsDetailed, SymbolOverrideSettings,
    ApiKeysSettings
)

# --- TimeOfDayDefinitions moved to context_schemas.py as it's used by TickerContextAnalyzer ---
# from .context_schemas import TimeOfDayDefinitions (if it were here)

# --- Dashboard Specific Configuration Models ---
class DashboardModeSettings(BaseModel):
    """Defines settings for a single dashboard mode."""
    label: str = Field(..., description="Display label for the mode in UI selectors.")
    module_name: str = Field(..., description="Python module name (e.g., 'main_dashboard_display_v2_5') to import for this mode's layout and callbacks.")
    charts: List[str] = Field(default_factory=list, description="List of chart/component identifier names expected to be displayed in this mode.")
    
    model_config = ConfigDict(extra='forbid')


class MainDashboardDisplaySettings(BaseModel):
    """Settings specific to components on the main dashboard display."""
    regime_indicator: RegimeIndicatorConfig = Field(default_factory=RegimeIndicatorConfig, description="Configuration for the Market Regime indicator display.")
    flow_gauge: FlowGaugeConfig = Field(default_factory=FlowGaugeConfig, description="Configuration for flow gauge visualizations.")
    gib_gauge: GibGaugeConfig = Field(default_factory=GibGaugeConfig, description="Configuration for GIB gauge visualizations.")
    mini_heatmap: MiniHeatmapConfig = Field(default_factory=MiniHeatmapConfig, description="Default settings for mini-heatmap components.")
    recommendations_table: RecommendationsTableConfig = Field(default_factory=RecommendationsTableConfig, description="Configuration for the ATIF recommendations table.")
    ticker_context: TickerContextConfig = Field(default_factory=TickerContextConfig, description="Settings for ticker context display area.")
    
    model_config = ConfigDict(extra='forbid')


class DashboardModeCollection(BaseModel):
    """Defines the collection of all available dashboard modes."""
    main: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Main Dashboard", module_name="main_dashboard_display_v2_5",
        charts=["regime_display", "flow_gauges", "gib_gauges", "recommendations_table"]
    ))
    flow: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Flow Analysis", module_name="flow_mode_display_v2_5",
        charts=["net_value_heatmap_viz", "net_cust_delta_flow_viz", "net_cust_gamma_flow_viz", "net_cust_vega_flow_viz"]
    ))
    # ... other default modes from original schema (structure, timedecay, advanced)
    structure: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Structure & Positioning", module_name="structure_mode_display_v2_5", 
        charts=["mspi_components", "sai_ssi_displays"]
    ))
    timedecay: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Time Decay & Pinning", module_name="time_decay_mode_display_v2_5",
        charts=["tdpi_displays", "vci_strike_charts"]
    ))
    advanced: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Advanced Flow Metrics", module_name="advanced_flow_mode_display_v2_5",
        charts=["vapi_gauges", "dwfd_gauges", "tw_laf_gauges"]
    ))
    
    model_config = ConfigDict(extra='forbid')


class VisualizationSettings(BaseModel):
    """Overall visualization and dashboard settings."""
    dashboard_refresh_interval_seconds: int = Field(60, ge=10, description="Interval in seconds between automatic dashboard data refreshes.")
    max_table_rows_signals_insights: int = Field(10, ge=1, description="Maximum number of rows to display in signals and insights tables on the dashboard.")
    dashboard: DashboardServerConfig = Field(default_factory=DashboardServerConfig, description="Core Dash server and display settings.")
    modes_detail_config: DashboardModeCollection = Field(default_factory=lambda: DashboardModeCollection())
    main_dashboard_settings: MainDashboardDisplaySettings = Field(default_factory=lambda: MainDashboardDisplaySettings())

    model_config = ConfigDict(extra='forbid')


# --- Metric Calculation Coefficients ---
class DagAlphaCoeffs(BaseModel):
    """Coefficients for A-DAG calculation based on flow alignment."""
    aligned: float = Field(default=1.35, description="Coefficient when market flow aligns with OI structure.")
    opposed: float = Field(default=0.65, description="Coefficient when market flow opposes OI structure.")
    neutral: float = Field(default=1.0, description="Coefficient when market flow is neutral to OI structure.")
    class Config: extra = 'forbid'

class TdpiBetaCoeffs(BaseModel):
    """Coefficients for D-TDPI calculation based on Charm flow alignment."""
    aligned: float = Field(default=1.35, description="Coefficient for aligned Charm flow.")
    opposed: float = Field(default=0.65, description="Coefficient for opposed Charm flow.")
    neutral: float = Field(default=1.0, description="Coefficient for neutral Charm flow.")
    class Config: extra = 'forbid'

class VriGammaCoeffs(BaseModel): # Note: "Gamma" here refers to vri_gamma, not option gamma
    """Coefficients for VRI 2.0 related to Vanna flow proxy alignment."""
    aligned: float = Field(default=1.35, description="Coefficient for aligned Vanna flow proxy.")
    opposed: float = Field(default=0.65, description="Coefficient for opposed Vanna flow proxy.")
    neutral: float = Field(default=1.0, description="Coefficient for neutral Vanna flow proxy.")
    model_config = ConfigDict(extra='forbid')

class CoefficientsSettings(BaseModel):
    """Container for various metric calculation coefficients."""
    dag_alpha: DagAlphaCoeffs = Field(default_factory=lambda: DagAlphaCoeffs(), description="A-DAG alpha coefficients.")
    tdpi_beta: TdpiBetaCoeffs = Field(default_factory=lambda: TdpiBetaCoeffs(), description="D-TDPI beta coefficients.")
    vri_gamma: VriGammaCoeffs = Field(default_factory=lambda: VriGammaCoeffs(), description="VRI 2.0 gamma (vri_gamma) coefficients for Vanna flow.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dag_alpha': self.dag_alpha.model_dump(),
            'tdpi_beta': self.tdpi_beta.model_dump(),
            'vri_gamma': self.vri_gamma.model_dump()
        }
    model_config = ConfigDict(extra='forbid')


# --- Data Processor Settings ---
class DataProcessorSettings(BaseModel):
    """Settings for the InitialDataProcessorV2_5, including factors and coefficients."""
    factors: DataProcessorFactors = Field(default_factory=DataProcessorFactors, description="Various numerical factors used in metric calculations (e.g., tdpi_gaussian_width).")
    coefficients: CoefficientsSettings = Field(default_factory=lambda: CoefficientsSettings(), description="Collection of Greek interaction coefficients.")
    iv_context_parameters: IVContextParameters = Field(default_factory=IVContextParameters, description="Parameters for IV contextualization (e.g., vol_trend_avg_days).")
    
    def to_dict(self) -> Dict[str, Any]:
         return {
             'factors': self.factors.to_dict(),
             'coefficients': self.coefficients.to_dict(),
             'iv_context_parameters': self.iv_context_parameters.to_dict() # .to_dict() on sub-model should be .model_dump()
         }
    model_config = ConfigDict(extra='forbid')

    def to_dict(self) -> Dict[str, Any]: # Overwrite to ensure proper model_dump usage
         return self.model_dump(exclude_none=True)


# --- Market Regime Engine Settings ---
class MarketRegimeEngineSettings(BaseModel):
    """Configuration for the MarketRegimeEngineV2_5."""
    default_regime: str = Field(default="REGIME_UNCLEAR_OR_TRANSITIONING", description="Default market regime if no other rules match.")
    regime_evaluation_order: List[str] = Field(default_factory=list, description="Order in which to evaluate market regime rules (most specific first).")
    regime_rules: RegimeRules = Field(default_factory=RegimeRules, description="Rules for detecting different market regimes (e.g., 'bullish', 'bearish').")
    # TimeOfDayDefinitions is now in context_schemas.py and imported there if needed by MRE logic.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'default_regime': self.default_regime,
            'regime_evaluation_order': self.regime_evaluation_order,
            'regime_rules': self.regime_rules.to_dict()
        }
    
    class Config: extra = 'forbid'


# --- System, Fetcher, Data Management Settings ---
class SystemSettings(BaseModel):
    """General system-level settings for EOTS v2.5."""
    project_root_override: Optional[str] = Field(None, description="Absolute path to override auto-detected project root. Use null for auto-detection.")
    logging_level: str = Field("INFO", description="Minimum logging level (e.g., DEBUG, INFO, WARNING, ERROR).")
    log_to_file: bool = Field(True, description="If true, logs will be written to the file specified in log_file_path.")
    log_file_path: str = Field("logs/eots_v2_5.log", description="Relative path from project root for the log file.") # pattern=".log$" removed for simplicity
    max_log_file_size_bytes: int = Field(10485760, ge=1024, description="Maximum size of a single log file in bytes before rotation.")
    backup_log_count: int = Field(5, ge=0, description="Number of old log files to keep after rotation.")
    live_mode: bool = Field(True, description="If true, system attempts to use live data sources; affects error handling.")
    fail_fast_on_errors: bool = Field(True, description="If true, system may halt on critical data quality or API errors.")
    metrics_for_dynamic_threshold_distribution_tracking: List[str] = Field(
        default_factory=lambda: ["GIB_OI_based_Und", "VAPI_FA_Z_Score_Und", "DWFD_Z_Score_Und", "TW_LAF_Z_Score_Und"],
        description="List of underlying aggregate metric names to track historically for dynamic threshold calculations."
    )
    signal_activation: SignalActivationSettings = Field(default_factory=SignalActivationSettings, description="Master toggles for enabling or disabling specific signal generation routines or categories.")
    class Config: extra = 'forbid'

class ConvexValueAuthSettings(BaseModel):
    """Authentication settings for ConvexValue API."""
    use_env_variables: bool = Field(True, description="If true, attempts to load credentials from environment variables first.")
    auth_method: str = Field("email_password", description="Authentication method for ConvexValue API (e.g., 'email_password', 'api_key').")
    # Specific credential fields would be here if not using env variables, e.g., email: Optional[str], password: Optional[SecretStr]
    class Config: extra = 'forbid'

class DataFetcherSettings(BaseModel):
    """Settings for data fetching components."""
    convexvalue_auth: ConvexValueAuthSettings = Field(default_factory=ConvexValueAuthSettings, description="Authentication settings for ConvexValue.")
    tradier_api_key: str = Field(..., description="API Key for Tradier (sensitive, ideally from env var).")
    tradier_account_id: str = Field(..., description="Account ID for Tradier (sensitive, ideally from env var).")
    max_retries: int = Field(3, ge=0, description="Maximum number of retry attempts for a failing API call.")
    retry_delay_seconds: float = Field(5.0, ge=0, description="Base delay in seconds between API call retries.")
    # Added fields based on ValidationError
    api_keys: Optional[ApiKeysSettings] = Field(default=None, description="Optional API keys configuration if not using direct fields.")
    retry_attempts: Optional[int] = Field(3, description="Number of retry attempts for API calls.") # Defaulted from error
    retry_delay: Optional[float] = Field(5.0, description="Delay in seconds between retries.") # Defaulted from error
    timeout: Optional[float] = Field(30.0, description="Timeout in seconds for API requests.") # Defaulted from error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'convexvalue_auth': self.convexvalue_auth.model_dump(),
            'tradier_api_key': self.tradier_api_key,
            'tradier_account_id': self.tradier_account_id,
            'max_retries': self.max_retries,
            'retry_delay_seconds': self.retry_delay_seconds,
            'api_keys': self.api_keys.to_dict() if self.api_keys else None,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'timeout': self.timeout
        }
    
    class Config: extra = 'forbid'

class DataManagementSettings(BaseModel):
    """Settings related to data caching and storage."""
    data_cache_dir: str = Field("data_cache_v2_5", description="Root directory for caching temporary data.")
    historical_data_store_dir: str = Field("data_cache_v2_5/historical_data_store", description="Directory for persistent historical market and metric data.")
    performance_data_store_dir: str = Field("data_cache_v2_5/performance_data_store", description="Directory for storing trade recommendation performance data.")
    # Added fields based on ValidationError
    cache_directory: Optional[str] = Field("data_cache_v2_5", description="Cache directory path.") # Defaulted from error
    data_store_directory: Optional[str] = Field("data_cache_v2_5/data_store", description="Data store directory path.") # Defaulted from error
    cache_expiry_hours: Optional[float] = Field(24.0, description="Cache expiry in hours.") # Defaulted from error
    class Config: extra = 'forbid'


# --- Settings for Specific Analytical Components ---
class EnhancedFlowMetricSettings(BaseModel):
    """Parameters for Tier 3 Enhanced Rolling Flow Metrics (VAPI-FA, DWFD, TW-LAF)."""
    vapi_fa_params: VAPIFAParameters = Field(default_factory=VAPIFAParameters, description="Parameters specific to VAPI-FA calculation (e.g., primary_flow_interval, iv_source_key).")
    dwfd_params: DWFDParameters = Field(default_factory=DWFDParameters, description="Parameters specific to DWFD calculation (e.g., flow_interval, fvd_weight_factor).")
    tw_laf_params: TWLAFParameters = Field(default_factory=TWLAFParameters, description="Parameters specific to TW-LAF calculation (e.g., time_weights_for_intervals, spread_calculation_params).")
    # Common params (can be overridden within specific metric_params if needed)
    z_score_window: int = Field(20, ge=5, le=200, description="Default window size for Z-score normalization of enhanced flow metrics.")
    # Added fields based on ValidationError
    acceleration_calculation_intervals: Optional[List[str]] = Field(default_factory=list, description="Intervals for flow acceleration calculation.")
    time_intervals: Optional[List[int]] = Field(default_factory=list, description="Time intervals for flow metrics.")
    liquidity_weight: Optional[float] = Field(None, description="Weight for liquidity adjustment.")
    divergence_threshold: Optional[float] = Field(None, description="Threshold for flow divergence.")
    lookback_periods: Optional[List[int]] = Field(default_factory=list, description="Lookback periods for flow calculations.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vapi_fa_params': self.vapi_fa_params.to_dict(),
            'dwfd_params': self.dwfd_params.to_dict(),
            'tw_laf_params': self.tw_laf_params.to_dict(),
            'z_score_window': self.z_score_window,
            'acceleration_calculation_intervals': self.acceleration_calculation_intervals,
            'time_intervals': self.time_intervals,
            'liquidity_weight': self.liquidity_weight,
            'divergence_threshold': self.divergence_threshold,
            'lookback_periods': self.lookback_periods
        }
    
    class Config: extra = 'forbid'

class StrategySettings(BaseModel): # General strategy settings, can be expanded
    """High-level strategy settings, often used for ATIF and TPO guidance."""
    # Example: thresholds for various signals or conviction modifiers
    # This model uses 'allow' as it's a common place for various ad-hoc strategy params.
    # Better practice would be to define specific sub-models for different strategy aspects.
    class Config: extra = 'allow'

class LearningParams(BaseModel):
    """Parameters governing the ATIF's performance-based learning loop."""
    performance_tracker_query_lookback: int = Field(90, ge=1, description="Number of days of historical performance data ATIF considers for learning.")
    learning_rate_for_signal_weights: float = Field(0.05, ge=0, le=1, description="Aggressiveness (0-1) of ATIF's adjustment to signal weights based on new performance.")
    learning_rate_for_target_adjustments: float = Field(0.02, ge=0, le=1, description="Aggressiveness (0-1) of ATIF's adjustment to target parameters based on new performance.")
    min_trades_for_statistical_significance: int = Field(20, ge=1, description="Minimum number of trades for a specific setup/symbol/regime before performance significantly influences weights.")
    class Config: extra = 'forbid'

# ATIFSettingsModel appears to be an older version of AdaptiveTradeIdeaFrameworkSettings.
# The canonical AdaptiveTradeIdeaFrameworkSettings is defined in data_models/configuration_models.py
# class ATIFSettingsModel(BaseModel):
#     min_conviction_to_initiate_trade: float = Field(..., description="Minimum conviction score required to initiate a trade.")
#     signal_integration_params: dict = Field(default_factory=dict)
#     regime_context_weight_multipliers: dict = Field(default_factory=dict)
#     conviction_mapping_params: dict = Field(default_factory=dict)
#     strategy_specificity_rules: list = Field(default_factory=list)
#     intelligent_recommendation_management_rules: dict = Field(default_factory=dict)
#     learning_params: LearningParams = Field(default_factory=lambda: LearningParams())

#     class Config:
#         extra = "forbid" # Changed to forbid to enforce strictness

# AdaptiveTradeIdeaFrameworkSettings is defined and refactored in data_models/configuration_models.py
# class AdaptiveTradeIdeaFrameworkSettings(BaseModel):
#     """Comprehensive settings for the Adaptive Trade Idea Framework (ATIF)."""
#     min_conviction_to_initiate_trade: float = Field(2.5, ge=0, le=5, description="Minimum ATIF conviction score (0-5 scale) to generate a new trade recommendation.")
#     signal_integration_params: SignalIntegrationParameters = Field(default_factory=SignalIntegrationParameters, description="Parameters for how ATIF integrates and weights raw signals (e.g., base_signal_weights, performance_weighting_sensitivity).")
#     regime_context_weight_multipliers: RegimeContextWeightMultipliers = Field(default_factory=RegimeContextWeightMultipliers, description="Multipliers applied to signal weights based on current market regime.")
#     conviction_mapping_params: ConvictionMappingParameters = Field(default_factory=ConvictionMappingParameters, description="Rules and thresholds for mapping ATIF's internal assessment to a final conviction score.")
#     strategy_specificity_rules: List[StrategySpecificRule] = Field(default_factory=list, description="Rule set mapping [Assessment + Conviction + Regime + Context + IV] to specific option strategies, DTEs, and deltas.")
#     intelligent_recommendation_management_rules: IntelligentRecommendationManagementRules = Field(default_factory=IntelligentRecommendationManagementRules, description="Rules for adaptive exits, parameter adjustments, and partial position management.")
#     learning_params: LearningParams = Field(default_factory=lambda: LearningParams())

#     def to_dict(self) -> Dict[str, Any]:
#         """Convert to dictionary for backward compatibility."""
#         return self.model_dump()

#     class Config: extra = 'forbid'

class TickerContextAnalyzerSettings(BaseModel):
    """Settings for the TickerContextAnalyzerV2_5."""
    # General settings
    lookback_days: int = Field(252, description="Default lookback days for historical analysis (e.g., for IV rank).") # Approx 1 year
    correlation_window: int = Field(60, description="Window for calculating correlations if used.")
    volatility_windows: List[int] = Field(default_factory=lambda: [1, 5, 20], description="Windows for short, medium, long term volatility analysis.")
    # Example specific settings for a ticker or default profile
    SPY: TickerSpecificParameters = Field(default_factory=TickerSpecificParameters, description="Specific context analysis parameters for SPY.")
    DEFAULT_TICKER_PROFILE: TickerSpecificParameters = Field(default_factory=TickerSpecificParameters, description="Default parameters for tickers not explicitly defined.")
    # Parameters for fetching external data if used (e.g., earnings calendar)
    # use_yahoo_finance: bool = False
    # yahoo_finance_rate_limit_seconds: float = 2.0
    # Added fields based on ValidationError
    volume_threshold: Optional[int] = Field(None, description="Volume threshold for ticker context analysis.")
    use_yahoo_finance: Optional[bool] = Field(False, description="Flag to use Yahoo Finance for data.")
    yahoo_finance_rate_limit_seconds: Optional[float] = Field(2.0, description="Rate limit in seconds for Yahoo Finance API calls.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lookback_days': self.lookback_days,
            'correlation_window': self.correlation_window,
            'volatility_windows': self.volatility_windows,
            'SPY': self.SPY.to_dict(),
            'DEFAULT_TICKER_PROFILE': self.DEFAULT_TICKER_PROFILE.to_dict(),
            'volume_threshold': self.volume_threshold,
            'use_yahoo_finance': self.use_yahoo_finance,
            'yahoo_finance_rate_limit_seconds': self.yahoo_finance_rate_limit_seconds
        }
    
    class Config: extra = 'forbid'

class KeyLevelIdentifierSettings(BaseModel):
    """Settings for the KeyLevelIdentifierV2_5."""
    lookback_periods: int = Field(20, description="Lookback period for identifying significant prior S/R from A-MSPI history.")
    min_touches: int = Field(2, description="Minimum times a historical level must have been touched to be considered significant.")
    level_tolerance: float = Field(0.005, ge=0, le=0.1, description="Percentage tolerance for clustering nearby levels (e.g., 0.5% = 0.005).")
    # Thresholds for identifying levels from various sources
    nvp_support_quantile: float = Field(0.95, ge=0, le=1, description="Quantile for identifying strong NVP support levels.")
    nvp_resistance_quantile: float = Field(0.95, ge=0, le=1, description="Quantile for identifying strong NVP resistance levels.")
    # Other source-specific thresholds (e.g., for SGDHP, UGCH scores) would be defined here.
    # Added fields based on ValidationError
    volume_threshold: Optional[float] = Field(None, description="Volume threshold for key level identification.")
    oi_threshold: Optional[int] = Field(None, description="Open interest threshold for key level identification.")
    gamma_threshold: Optional[float] = Field(None, description="Gamma threshold for key level identification.")
    class Config: extra = 'forbid'

class HeatmapGenerationSettings(BaseModel):
    """Parameters for generating data for Enhanced Heatmaps."""
    ugch_params: UGCHParameters = Field(default_factory=UGCHParameters, description="Parameters for UGCH data generation, e.g., weights for each Greek.")
    sgdhp_params: SGDHPParameters = Field(default_factory=SGDHPParameters, description="Parameters for SGDHP data generation, e.g., price proximity sensitivity.")
    ivsdh_params: IVSDHParameters = Field(default_factory=IVSDHParameters, description="Parameters for IVSDH data generation.")
    # Added field based on ValidationError
    flow_normalization_window: Optional[int] = Field(None, description="Window for flow normalization in heatmap generation.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ugch_params': self.ugch_params.to_dict(),
            'sgdhp_params': self.sgdhp_params.to_dict(),
            'ivsdh_params': self.ivsdh_params.to_dict(),
            'flow_normalization_window': self.flow_normalization_window
        }
    
    class Config: extra = 'forbid'

class PerformanceTrackerSettingsV2_5(BaseModel):
    """Settings for the PerformanceTrackerV2_5 module."""
    performance_data_directory: str = Field("data_cache_v2_5/performance_data_store", description="Directory for storing performance tracking data files.")
    historical_window_days: int = Field(365, ge=1, description="Number of days of performance data to retain and consider for analysis.")
    weight_smoothing_factor: float = Field(0.1, ge=0, le=1, description="Smoothing factor for performance-based weight adjustments by ATIF (0=no new learning, 1=full new learning).")
    min_sample_size_for_stats: int = Field(10, ge=1, description="Minimum number of trade samples required for calculating reliable performance statistics for a setup/signal.")
    # confidence_threshold: float = Field(0.75, ge=0, le=1, description="Confidence threshold for performance-based adjustments - not directly used by ATIF learning_rate, but for user interpretation.")
    # update_interval_seconds: int = Field(3600, ge=1, description="Interval for batch updates or re-analysis of performance data (if applicable).")
    tracking_enabled: bool = Field(True, description="Master toggle for enabling/disabling performance tracking.")
    metrics_to_track_display: List[str] = Field(default_factory=lambda: ["returns", "sharpe_ratio", "max_drawdown", "win_rate"], description="List of performance metrics to display on dashboard.")
    # reporting_frequency: str = Field("daily", description="Frequency of generating performance reports (if applicable).")
    # benchmark_symbol: str = Field("SPY", description="Benchmark symbol for relative performance comparison.")
    # Added fields based on ValidationError
    min_sample_size: Optional[int] = Field(10, description="Minimum sample size for performance statistics.")
    confidence_threshold: Optional[float] = Field(0.75, description="Confidence threshold for performance adjustments.")
    update_interval_seconds: Optional[int] = Field(3600, description="Interval for performance data updates.")
    class Config: extra = 'forbid'

class AdaptiveMetricParameters(BaseModel): # Top-level container for all adaptive metric specific settings in config
    """Container for settings related to all Tier 2 Adaptive Metrics."""
    a_dag_settings: ADAGSettings = Field(default_factory=ADAGSettings, description="Settings for Adaptive Delta Adjusted Gamma Exposure (A-DAG).")
    e_sdag_settings: ESDAGSettings = Field(default_factory=ESDAGSettings, description="Settings for Enhanced Skew and Delta Adjusted Gamma Exposure (E-SDAG) methodologies.")
    d_tdpi_settings: DTDPISettings = Field(default_factory=DTDPISettings, description="Settings for Dynamic Time Decay Pressure Indicator (D-TDPI).")
    vri_2_0_settings: VRI2Settings = Field(default_factory=VRI2Settings, description="Settings for Volatility Regime Indicator Version 2.0 (VRI 2.0).")
    enhanced_heatmap_settings: Optional[EnhancedHeatmapSettingsDetailed] = Field(default=None, description="Settings for enhanced heatmap generation.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "a_dag_settings": self.a_dag_settings.to_dict(),
            "e_sdag_settings": self.e_sdag_settings.to_dict(),
            "d_tdpi_settings": self.d_tdpi_settings.to_dict(),
            "vri_2_0_settings": self.vri_2_0_settings.to_dict(),
            "enhanced_heatmap_settings": self.enhanced_heatmap_settings.to_dict() if self.enhanced_heatmap_settings else None
        }
    
    class Config: extra = 'forbid'


# --- Analytics Engine Configuration ---
class AnalyticsEngineConfigV2_5(BaseModel):
    """Configuration for the core analytics engine components."""
    metrics_calculation_enabled: bool = Field(True, description="Enable/disable all metric calculations.")
    market_regime_analysis_enabled: bool = Field(True, description="Enable/disable market regime analysis.")
    signal_generation_enabled: bool = Field(True, description="Enable/disable signal generation.")
    key_level_identification_enabled: bool = Field(True, description="Enable/disable key level identification.")
    # Add more specific settings as needed for the analytics engine
    class Config:
        extra = 'forbid'

# --- Adaptive Learning Configuration ---
class AdaptiveLearningConfigV2_5(BaseModel):
    """Configuration for the Adaptive Learning Integration module."""
    auto_adaptation: bool = Field(True, description="Enable/disable automatic application of adaptations.")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence score for an insight to trigger adaptation.")
    pattern_discovery_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum confidence score for an insight to be considered a valid pattern discovery.")
    adaptation_frequency_minutes: int = Field(60, ge=1, description="How often (in minutes) the system checks for new adaptations.")
    analytics_engine: AnalyticsEngineConfigV2_5 = Field(default_factory=lambda: AnalyticsEngineConfigV2_5(), description="Nested configuration for the analytics engine within adaptive learning.")
    # Add more settings relevant to adaptive learning, e.g., learning rates, model paths
    class Config:
        extra = 'forbid'

# --- Prediction Configuration ---
class PredictionConfigV2_5(BaseModel):
    """Configuration for the AI Predictions Manager module."""
    enabled: bool = Field(True, description="Enable/disable AI predictions.")
    model_name: str = Field("default_prediction_model", description="Name of the primary prediction model to use.")
    prediction_interval_seconds: int = Field(300, ge=60, description="How often (in seconds) to generate new predictions.")
    max_data_age_seconds: int = Field(120, ge=10, description="Maximum age of market data (in seconds) to be considered fresh for predictions.")
    confidence_calibration: ConfidenceCalibration = Field(default_factory=ConfidenceCalibration, description="Parameters for calibrating confidence scores based on signal strength.")
    success_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum performance score for a prediction to be considered successful.")
    # Add more settings relevant to prediction, e.g., specific model parameters, data sources
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'model_name': self.model_name,
            'prediction_interval_seconds': self.prediction_interval_seconds,
            'max_data_age_seconds': self.max_data_age_seconds,
            'confidence_calibration': self.confidence_calibration.to_dict(),
            'success_threshold': self.success_threshold
        }
    
    class Config:
        extra = 'forbid'

# --- Trade Parameter Optimizer Settings ---
class TradeParameterOptimizerSettings(BaseModel):
    """Settings for the TradeParameterOptimizerV2_5 module."""
    contract_selection_filters: ContractSelectionFilters = Field(default_factory=ContractSelectionFilters, description="Filters for selecting optimal option contracts.")
    entry_price_logic: str = Field("MID_PRICE", description="Logic for calculating entry price (e.g., 'MID_PRICE', 'LIMIT_X_CENTS_THROUGH_MID').")
    stop_loss_calculation_rules: StopLossCalculationRules = Field(default_factory=StopLossCalculationRules, description="Rules for calculating adaptive stop-losses.")
    profit_target_calculation_rules: ProfitTargetCalculationRules = Field(default_factory=ProfitTargetCalculationRules, description="Rules for calculating multi-tiered profit targets.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'contract_selection_filters': self.contract_selection_filters.to_dict(),
            'entry_price_logic': self.entry_price_logic,
            'stop_loss_calculation_rules': self.stop_loss_calculation_rules.to_dict(),
            'profit_target_calculation_rules': self.profit_target_calculation_rules.to_dict()
        }
    
    class Config:
        extra = 'forbid'


# --- Symbol Specific Overrides Structure ---
class SymbolDefaultOverridesStrategySettingsTargets(BaseModel):
    """Defines default target parameters for strategy settings."""
    target_atr_stop_loss_multiplier: float = Field(1.5, ge=0.1)
    class Config: extra = 'forbid'

class SymbolDefaultOverridesStrategySettings(BaseModel):
    """Defines default strategy settings, can be overridden per symbol."""
    targets: SymbolDefaultOverridesStrategySettingsTargets = Field(default_factory=lambda: SymbolDefaultOverridesStrategySettingsTargets())
    class Config: extra = 'forbid'

class SymbolDefaultOverrides(BaseModel):
    """Container for default settings that can be overridden by specific symbols."""
    strategy_settings: Optional[SymbolDefaultOverridesStrategySettings] = Field(default_factory=lambda: SymbolDefaultOverridesStrategySettings())
    # Can add other overridable sections here, e.g., market_regime_engine_settings
    class Config: extra = 'allow' # Allow adding other top-level setting blocks for DEFAULT

class SymbolSpecificOverrides(BaseModel):
    """
    Main container for symbol-specific configuration overrides.
    Keys are ticker symbols (e.g., "SPY", "AAPL") or "DEFAULT".
    Values are dictionaries structured like parts of the main config.
    """
    DEFAULT: Optional[SymbolDefaultOverrides] = Field(default_factory=lambda: SymbolDefaultOverrides(), description="Default override profile applied if no ticker-specific override exists.")
    SPY: Optional[SymbolOverrideSettings] = Field(default=None, description="Specific overrides for SPY.")
    # Add other commonly traded symbols as needed, e.g., AAPL: Optional[Dict[str, Any]]
    class Config: extra = 'allow' # Allows adding new ticker symbols as keys


# --- Database and Collector Settings (if used) ---
class DatabaseSettings(BaseModel):
    """Database connection settings (if a central DB is used)."""
    host: str = Field(..., description="Database host address.")
    port: int = Field(5432, description="Database port number.")
    database: str = Field(..., description="Database name.")
    user: str = Field(..., description="Database username.")
    password: str = Field(..., description="Database password (sensitive).") # Consider pydantic.SecretStr
    min_connections: int = Field(1, ge=0, description="Minimum number of connections in pool.")
    max_connections: int = Field(10, ge=1, description="Maximum number of connections in pool.")
    class Config: extra = 'forbid'

class IntradayCollectorSettings(BaseModel):
    """Settings for an intraday metrics collector service (if separate)."""
    watched_tickers: List[str] = Field(default_factory=lambda: ["SPY", "QQQ"], description="List of tickers for intraday metric collection.")
    metrics_to_collect: List[str] = Field(default_factory=lambda: ["vapi_fa", "dwfd", "tw_laf"], description="Specific metrics to collect intraday.")
    cache_dir: str = Field("cache/intraday_metrics_collector", description="Directory for intraday collector cache.")
    collection_interval_seconds: int = Field(60, ge=5, description="Interval in seconds between metric collections.")
    market_open_time: str = Field("09:30:00", description="Market open time (HH:MM:SS) for collector activity.")
    market_close_time: str = Field("16:00:00", description="Market close time (HH:MM:SS) for collector activity.")
    reset_cache_at_eod: bool = Field(True, description="Whether to wipe intraday cache at end of day.")
    # Added fields based on ValidationError
    metrics: Optional[List[str]] = Field(default_factory=list, description="List of metrics for the intraday collector.")
    reset_at_eod: Optional[bool] = Field(True, description="Whether to reset cache at EOD for intraday collector.") # Field already existed, ensuring Optional and default
    class Config: extra = 'forbid'


class TradingConfigV2_5(BaseModel):
    """Consolidated trading configuration - replaces AdaptiveTradeIdeaFrameworkSettings + StrategySettings."""

    # ATIF Configuration
    min_conviction_to_initiate_trade: float
    signal_integration_params: SignalIntegrationParameters
    regime_context_weight_multipliers: RegimeContextWeightMultipliers
    conviction_mapping_params: ConvictionMappingParameters
    strategy_specificity_rules: List[StrategySpecificRule]
    intelligent_recommendation_management_rules: IntelligentRecommendationManagementRules
    performance_tracker_query_lookback: int
    learning_rate_for_signal_weights: float
    learning_rate_for_target_adjustments: float
    min_trades_for_statistical_significance: int
    strategy_params: StrategyParameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump()

    model_config = ConfigDict(extra='allow')  # Allow additional strategy parameters

class DashboardConfigV2_5(BaseModel):
    """Consolidated dashboard configuration - replaces VisualizationSettings + DashboardModeSettings + MainDashboardDisplaySettings."""

    # Core Dashboard Settings
    refresh_interval_seconds: int = Field(default=30, description="Interval between dashboard refreshes")
    host: str = Field(default="localhost", description="Dashboard host")

# --- PORTED: ControlPanelParametersV2_5 (from deprecated_files/eots_schemas_v2_5.py) ---

class ControlPanelParametersV2_5(BaseModel):
    """PYDANTIC-FIRST: Control panel parameters with strict validation.
    
    *** STRICT VALIDATION: NO DEFAULTS, NO FALLBACKS ***
    *** ALL PARAMETERS MUST BE EXPLICITLY PROVIDED ***
    *** VALIDATION ERRORS WILL FAIL FAST ***
    """
    symbol: str = Field(
        ...,  # No default - must be provided
        description="Trading symbol (e.g., 'SPY', 'SPX')",
        min_length=1,
        max_length=10
    )
    dte_min: int = Field(
        ...,  # No default - must be provided
        description="Minimum days to expiration",
        ge=0,
        le=365
    )
    dte_max: int = Field(
        ...,  # No default - must be provided
        description="Maximum days to expiration",
        ge=0,
        le=365
    )
    price_range_percent: int = Field(
        ...,  # No default - must be provided
        description="Price range percentage for filtering",
        ge=1,
        le=100
    )
    fetch_interval_seconds: int = Field(
        ...,  # No default - must be provided
        description="Data fetch interval in seconds",
        ge=5,
        le=3600
    )
    @field_validator('dte_max')
    @classmethod
    def validate_dte_range(cls, v: int, info: Any) -> int:
        if 'dte_min' in info.data and v < info.data['dte_min']:
            raise ValueError("dte_max must be greater than or equal to dte_min")
        return v
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        if not v.isalpha():
            raise ValueError("symbol must contain only letters")
        return v.upper()
    model_config = ConfigDict(extra='forbid', frozen=True)


# --- Elite Regime Detection Configuration ---
class EliteConfig(BaseModel):
    """Elite regime detection configuration."""
    enable_elite_regime_detection: bool = Field(True, description="Enable or disable elite regime detection.")
    elite_regime_threshold: float = Field(0.7, ge=0, le=1, description="Threshold for elite regime score.")
    model_config = ConfigDict(extra='forbid')

# --- Top-Level Configuration Model ---
class EOTSConfigV2_5(BaseModel):
    """
    The root model for the EOTS v2.5 system configuration (config_v2_5.json).
    It defines all valid parameters, their types, default values, and descriptions,
    ensuring configuration integrity and providing a structured way to access settings.
    """
    system_settings: SystemSettings = Field(default_factory=SystemSettings)
    data_fetcher_settings: DataFetcherSettings
    data_management_settings: DataManagementSettings = Field(default_factory=DataManagementSettings)
    database_settings: Optional[DatabaseSettings] = Field(None, description="Optional database connection settings.")
    
    data_processor_settings: DataProcessorSettings = Field(default_factory=lambda: DataProcessorSettings())
    adaptive_metric_parameters: AdaptiveMetricParameters = Field(default_factory=lambda: AdaptiveMetricParameters())
    enhanced_flow_metric_settings: EnhancedFlowMetricSettings = Field(default_factory=lambda: EnhancedFlowMetricSettings())
    key_level_identifier_settings: KeyLevelIdentifierSettings = Field(default_factory=lambda: KeyLevelIdentifierSettings())
    heatmap_generation_settings: HeatmapGenerationSettings = Field(default_factory=lambda: HeatmapGenerationSettings())
    market_regime_engine_settings: MarketRegimeEngineSettings = Field(default_factory=lambda: MarketRegimeEngineSettings())
    
    # Strategy and ATIF settings are often complex and interlinked
    strategy_settings: StrategySettings = Field(default_factory=lambda: StrategySettings())
    adaptive_trade_idea_framework_settings: AdaptiveTradeIdeaFrameworkSettings = Field(default_factory=lambda: ATIFSettingsModel())
    trade_parameter_optimizer_settings: TradeParameterOptimizerSettings = Field(default_factory=lambda: TradeParameterOptimizerSettings())
    
    ticker_context_analyzer_settings: TickerContextAnalyzerSettings = Field(default_factory=lambda: TickerContextAnalyzerSettings())
    performance_tracker_settings_v2_5: PerformanceTrackerSettingsV2_5 = Field(default_factory=lambda: PerformanceTrackerSettingsV2_5())
    
    visualization_settings: VisualizationSettings = Field(default_factory=lambda: VisualizationSettings())
    elite_config: EliteConfig = Field(default_factory=lambda: EliteConfig(), description="Elite regime detection configuration.")
    symbol_specific_overrides: SymbolSpecificOverrides = Field(default_factory=lambda: SymbolSpecificOverrides())
    
    # TimeOfDayDefinitions is now in context_schemas.py and typically loaded into MarketRegimeEngine or TickerContextAnalyzer
    # For direct config access if needed by other general utils:
    time_of_day_definitions: Optional[TimeOfDayDefinitions] = Field(None, description="Optional: Can load TimeOfDayDefinitions directly here if not handled by MRE/TCA init. Otherwise, they use context_schemas.TimeOfDayDefinitions.")

    intraday_collector_settings: Optional[IntradayCollectorSettings] = Field(None, description="Optional settings for a separate intraday metrics collector service.")
    prediction_config: Optional[PredictionConfigV2_5] = Field(default_factory=lambda: PredictionConfigV2_5(), description="Configuration for the AI Predictions Manager.")

    def to_dict(self) -> Dict[str, Any]: # Will be replaced by model_dump()
        return self.model_dump(exclude_none=True)

    model_config = ConfigDict(
        extra='forbid',
        json_schema_extra={
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "EOTS_V2_5_Config_Schema_Root",
            "description": "Root schema for EOTS v2.5 configuration (config_v2_5.json). Defines all valid parameters, types, defaults, and descriptions for system operation."
        },
        arbitrary_types_allowed=True # Might be needed if any fields are complex non-Pydantic types
    )