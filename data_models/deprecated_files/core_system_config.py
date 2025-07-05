"""
Core System Configuration Models for EOTS v2.5

This module contains core system settings, dashboard configurations,
and basic data processing parameters.

Extracted from configuration_models.py for better modularity.
"""

# Standard library imports
from typing import List, Dict, Any, Optional, Union

# Third-party imports
from pydantic import BaseModel, Field, FilePath, field_validator, model_validator, ConfigDict, FieldValidationInfo


# =============================================================================
# DASHBOARD & UI CONFIGURATION
# =============================================================================

class RegimeIndicatorConfig(BaseModel):
    """Configuration for regime indicator display."""
    enabled: bool = Field(default=True, description="Enable regime indicator")
    update_interval: int = Field(default=30, description="Update interval in seconds")
    
    class Config:
        extra = 'forbid'

class FlowGaugeConfig(BaseModel):
    """Configuration for flow gauge display."""
    enabled: bool = Field(default=True, description="Enable flow gauge")
    gauge_type: str = Field(default="radial", description="Type of gauge display")
    
    class Config:
        extra = 'forbid'

class GibGaugeConfig(BaseModel):
    """Configuration for GIB gauge display."""
    enabled: bool = Field(default=True, description="Enable GIB gauge")
    threshold_levels: List[float] = Field(default_factory=lambda: [-2.0, -1.0, 1.0, 2.0], description="Threshold levels for gauge")
    
    class Config:
        extra = 'forbid'

class MiniHeatmapConfig(BaseModel):
    """Configuration for mini heatmap display."""
    enabled: bool = Field(default=True, description="Enable mini heatmap")
    grid_size: tuple = Field(default=(10, 10), description="Grid size for heatmap")
    
    model_config = ConfigDict(extra='forbid')

class RecommendationsTableConfig(BaseModel):
    """Configuration for recommendations table display."""
    enabled: bool = Field(default=True, description="Enable recommendations table")
    max_rows: int = Field(default=10, description="Maximum rows to display")
    
    model_config = ConfigDict(extra='forbid')

class TickerContextConfig(BaseModel):
    """Configuration for ticker context display."""
    enabled: bool = Field(default=True, description="Enable ticker context")
    show_details: bool = Field(default=True, description="Show detailed context information")
    
    model_config = ConfigDict(extra='forbid')

class DashboardDefaults(BaseModel):
    """Default settings for dashboard components - FAIL FAST ON MISSING CONFIG."""
    symbol: str = Field(..., min_length=1, description="Default symbol - REQUIRED from config")
    refresh_interval_seconds: int = Field(..., gt=0, description="Default refresh interval in seconds - REQUIRED from config")
    dte_min: int = Field(..., ge=0, description="Default minimum DTE - REQUIRED from config")
    dte_max: int = Field(..., gt=0, description="Default maximum DTE - REQUIRED from config")
    price_range_percent: float = Field(..., gt=0.0, description="Default price range percentage - REQUIRED from config")

    @field_validator('symbol')
    @classmethod
    def validate_symbol_not_placeholder(cls, v: str) -> str:
        """CRITICAL: Reject placeholder symbols that indicate missing config."""
        placeholder_symbols = ['default', 'placeholder', 'symbol', 'ticker', 'example', 'test']
        if v.lower().strip() in placeholder_symbols:
            raise ValueError(f"CRITICAL: Symbol '{v}' appears to be a placeholder - provide real trading symbol from config!")
        return v.upper()

    @model_validator(mode='after')
    def validate_dte_range_consistency(self) -> 'DashboardDefaults':
        """CRITICAL: Validate DTE range consistency."""
        if self.dte_min >= self.dte_max:
            raise ValueError("CRITICAL: dte_min must be less than dte_max!")
        return self

    model_config = ConfigDict(extra='forbid') # Changed from 'allow'

class DashboardServerConfig(BaseModel):
    """Configuration for dashboard server."""
    host: str = Field(default="localhost", description="Dashboard host")
    port: int = Field(default=8050, description="Dashboard port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Additional config fields from config file
    auto_refresh_seconds: Optional[int] = Field(default=30, description="Auto refresh interval in seconds")
    timestamp_format: Optional[str] = Field(default='%Y-%m-%d %H:%M:%S %Z', description="Timestamp format")
    defaults: Optional[DashboardDefaults] = Field(default_factory=DashboardDefaults, description="Default dashboard settings") # This is fine as DashboardDefaults has required fields
    modes_detail_config: Optional[Union[Dict[str, Any], 'DashboardModeCollection']] = Field(default=None, description="Dashboard modes configuration")
    flow_mode_settings: Optional[Dict[str, Any]] = Field(default=None, description="Flow mode settings")
    volatility_mode_settings: Optional[Dict[str, Any]] = Field(default=None, description="Volatility mode settings")
    main_dashboard_settings: Optional[Dict[str, Any]] = Field(default=None, description="Main dashboard settings")

    model_config = ConfigDict(extra='allow') # Allowing extra fields here might be a specific design choice for top-level config.

class SignalActivationSettings(BaseModel):
    """Configuration for signal activation."""
    enabled: bool = Field(default=True, description="Enable signal activation")
    auto_refresh: bool = Field(default=True, description="Auto refresh signals")

    # Additional config fields from config file
    EnableAllSignals: Optional[bool] = Field(True, description="Enable all signals")

    model_config = ConfigDict(extra='forbid') # Changed from 'allow'


# =============================================================================
# DASHBOARD MODE SETTINGS
# =============================================================================

class DashboardModeSettings(BaseModel):
    """Defines settings for a single dashboard mode."""
    label: str = Field(..., description="Display label for the mode in UI selectors.")
    module_name: str = Field(..., description="Python module name to import for this mode's layout and callbacks.")
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
    
    model_config = ConfigDict(extra='forbid') # Converted from class Config

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
    structure: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Structure & Positioning", module_name="structure_mode_display_v2_5",
        charts=["mspi_components", "sai_ssi_displays"]
    ))
    timedecay: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Time Decay & Pinning", module_name="time_decay_mode_display_v2_5",
        charts=["tdpi_displays", "vci_strike_charts"]
    ))
    advanced: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Advanced Flow Metrics", module_name="advanced_flow_mode_v2_5",
        charts=["vapi_gauges", "dwfd_gauges", "tw_laf_gauges"]
    ))
    volatility: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Volatility Deep Dive", module_name="volatility_mode_display_v2_5",
        charts=["vri_2_0_strike_profile", "vri_0dte_strike_viz", "vfi_0dte_agg_viz", "vvr_0dte_agg_viz", "vci_0dte_agg_viz"]
    ))
    ai: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="AI Intelligence Hub", module_name="ai_dashboard.ai_dashboard_display_v2_5",
        charts=["ai_market_analysis", "ai_recommendations", "ai_insights", "ai_regime_context", "ai_performance_tracker"]
    ))

    model_config = ConfigDict(extra='forbid')  # Changed from 'allow'

    @classmethod
    def model_validate(cls, obj):
        """Custom validation to handle dictionary conversion for all mode fields."""
        if isinstance(obj, dict):
            # Convert nested dictionaries to DashboardModeSettings
            converted_obj = {}
            for key, value in obj.items():
                if isinstance(value, dict):
                    # Convert dict to DashboardModeSettings
                    converted_obj[key] = DashboardModeSettings.model_validate(value)
                else:
                    converted_obj[key] = value
            return super().model_validate(converted_obj)
        return super().model_validate(obj)

class VisualizationSettings(BaseModel):
    """Overall visualization and dashboard settings."""
    dashboard_refresh_interval_seconds: int = Field(60, ge=10, description="Interval in seconds between automatic dashboard data refreshes.")
    max_table_rows_signals_insights: int = Field(10, ge=1, description="Maximum number of rows to display in signals and insights tables on the dashboard.")
    dashboard: DashboardServerConfig = Field(default_factory=DashboardServerConfig, description="Core Dash server and display settings.")
    modes_detail_config: DashboardModeCollection = Field(default_factory=lambda: DashboardModeCollection())
    main_dashboard_settings: MainDashboardDisplaySettings = Field(default_factory=lambda: MainDashboardDisplaySettings())

    @field_validator('modes_detail_config', mode='before')
    @classmethod
    def validate_modes_detail_config(cls, v):
        """Ensure modes_detail_config is properly parsed as DashboardModeCollection."""
        if isinstance(v, dict):
            # Convert dictionary to DashboardModeCollection
            return DashboardModeCollection.model_validate(v)
        return v

    # Additional flexible configuration fields
    dashboard_mode_settings: Optional[Dict[str, Any]] = Field(default=None, description="Dashboard mode configuration settings.")
    main_dashboard_display_settings: Optional[Dict[str, Any]] = Field(default=None, description="Main dashboard display configuration.")

    model_config = ConfigDict(extra='forbid') # Changed from 'allow'


# =============================================================================
# SYSTEM SETTINGS
# =============================================================================

class SystemSettings(BaseModel):
    """General system-level settings for EOTS v2.5."""
    project_root_override: Optional[str] = Field(None, description="Absolute path to override auto-detected project root. Use null for auto-detection.")
    logging_level: str = Field("INFO", description="Minimum logging level (e.g., DEBUG, INFO, WARNING, ERROR).")
    log_to_file: bool = Field(True, description="If true, logs will be written to the file specified in log_file_path.")
    log_file_path: str = Field("logs/eots_v2_5.log", description="Relative path from project root for the log file.")
    max_log_file_size_bytes: int = Field(10485760, ge=1024, description="Maximum size of a single log file in bytes before rotation.")
    backup_log_count: int = Field(5, ge=0, description="Number of old log files to keep after rotation.")
    live_mode: bool = Field(True, description="If true, system attempts to use live data sources; affects error handling.")
    fail_fast_on_errors: bool = Field(True, description="If true, system may halt on critical data quality or API errors.")
    metrics_for_dynamic_threshold_distribution_tracking: List[str] = Field(
        default_factory=lambda: ["GIB_OI_based_Und", "VAPI_FA_Z_Score_Und", "DWFD_Z_Score_Und", "TW_LAF_Z_Score_Und"],
        description="List of underlying aggregate metric names to track historically for dynamic threshold calculations."
    )
    signal_activation: SignalActivationSettings = Field(default_factory=SignalActivationSettings, description="Master toggles for enabling or disabling specific signal generation routines or categories.")
    
    model_config = ConfigDict(extra='forbid')


# =============================================================================
# DATA FETCHER & MANAGEMENT SETTINGS
# =============================================================================

class ApiKeysSettings(BaseModel):
    """Configuration for API keys."""
    keys: Optional[Dict[str, str]] = Field(default=None, description="API keys")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    model_config = ConfigDict(extra='forbid')

class ConvexValueAuthSettings(BaseModel):
    """Authentication settings for ConvexValue API."""
    use_env_variables: bool = Field(True, description="If true, attempts to load credentials from environment variables first.")
    auth_method: str = Field("email_password", description="Authentication method for ConvexValue API (e.g., 'email_password', 'api_key').")
    
    model_config = ConfigDict(extra='forbid')

class DataFetcherSettings(BaseModel):
    """Settings for data fetching components."""
    convexvalue_auth: ConvexValueAuthSettings = Field(default_factory=ConvexValueAuthSettings, description="Authentication settings for ConvexValue.")
    tradier_api_key: str = Field(..., description="API Key for Tradier (sensitive, ideally from env var).")
    tradier_account_id: str = Field(..., description="Account ID for Tradier (sensitive, ideally from env var).")
    max_retries: int = Field(3, ge=0, description="Maximum number of retry attempts for a failing API call.")
    retry_delay_seconds: float = Field(5.0, ge=0, description="Base delay in seconds between API call retries.")
    timeout_seconds: Optional[float] = Field(30.0, description="Timeout in seconds for API requests.")
    api_keys: Optional[ApiKeysSettings] = Field(default=None, description="Optional API keys configuration if not using direct fields.")
    retry_attempts: Optional[int] = Field(3, description="Number of retry attempts for API calls.")
    retry_delay: Optional[float] = Field(5.0, description="Delay in seconds between retries.")
    timeout: Optional[float] = Field(30.0, description="Timeout in seconds for API requests.")
    
    def to_dict(self) -> Dict[str, Any]: # Ensure it uses model_dump
        return self.model_dump(exclude_none=True)
    
    model_config = ConfigDict(extra='forbid') # Changed from 'allow'

class DataManagementSettings(BaseModel):
    """Settings related to data caching and storage."""
    data_cache_dir: str = Field("data_cache_v2_5", description="Root directory for caching temporary data.")
    historical_data_store_dir: str = Field("data_cache_v2_5/historical_data_store", description="Directory for persistent historical market and metric data.")
    performance_data_store_dir: str = Field("data_cache_v2_5/performance_data_store", description="Directory for storing trade recommendation performance data.")
    cache_directory: Optional[str] = Field("data_cache_v2_5", description="Cache directory path.")
    data_store_directory: Optional[str] = Field("data_cache_v2_5/data_store", description="Data store directory path.")
    cache_expiry_hours: Optional[float] = Field(24.0, description="Cache expiry in hours.")
    
    model_config = ConfigDict(extra='forbid')


# =============================================================================
# DATABASE SETTINGS
# =============================================================================

class DatabaseSettings(BaseModel):
    """Database connection settings (if a central DB is used)."""
    host: str = Field(..., description="Database host address.")
    port: int = Field(5432, description="Database port number.")
    database: str = Field(..., description="Database name.")
    user: str = Field(..., description="Database username.")
    password: str = Field(..., description="Database password (sensitive).")
    min_connections: int = Field(1, ge=0, description="Minimum number of connections in pool.")
    max_connections: int = Field(10, ge=1, description="Maximum number of connections in pool.")

    model_config = ConfigDict(extra='forbid')


# =============================================================================
# CONTROL PANEL PARAMETERS
# =============================================================================

class ControlPanelParametersV2_5(BaseModel):
    """Control panel parameters with strict validation.

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


# =============================================================================
# INTRADAY COLLECTOR SETTINGS
# =============================================================================

class IntradayCollectorSettings(BaseModel):
    """Settings for an intraday metrics collector service (if separate)."""
    watched_tickers: List[str] = Field(default_factory=lambda: ["SPY", "QQQ"], description="List of tickers for intraday metric collection.")
    metrics: List[str] = Field(default_factory=lambda: ["vapi_fa", "dwfd", "tw_laf"], description="List of metrics for the intraday collector.")
    cache_dir: str = Field("cache/intraday_metrics", description="Directory for intraday collector cache.")
    collection_interval_seconds: int = Field(5, ge=5, description="Interval in seconds between metric collections.")
    market_open_time: str = Field("09:30:00", description="Market open time (HH:MM:SS) for collector activity.")
    market_close_time: str = Field("16:00:00", description="Market close time (HH:MM:SS) for collector activity.")
    reset_at_eod: bool = Field(True, description="Whether to reset cache at EOD for intraday collector.")

    # Alternative field names for backward compatibility
    metrics_to_collect: Optional[List[str]] = Field(None, description="Alternative field name for metrics (deprecated)")
    reset_cache_at_eod: Optional[bool] = Field(None, description="Alternative field name for reset_at_eod (deprecated)")

    # Additional config fields from config file
    symbol: Optional[str] = Field("SPY", description="Primary symbol for intraday collection")
    dte_min: Optional[int] = Field(0, description="Minimum days to expiration")
    dte_max: Optional[int] = Field(5, description="Maximum days to expiration")
    fetch_interval_seconds: Optional[int] = Field(30, description="Fetch interval in seconds")

    model_config = ConfigDict(extra='forbid') # Changed from 'allow'