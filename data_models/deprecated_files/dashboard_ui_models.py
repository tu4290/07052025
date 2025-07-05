"""
Dashboard & UI Models for EOTS v2.5

Consolidated from: dashboard_schemas.py, ui_component_schemas.py
"""

# Standard library imports
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union

# Third-party imports
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# Import from core_system_config to avoid circular imports
from .core_system_config import ControlPanelParametersV2_5


# =============================================================================
# FROM dashboard_schemas.py
# =============================================================================
"""
Pydantic models for the EOTS v2.5 Dashboard and UI components.

This module defines the data structures used for dashboard configurations,
UI components, and display settings across the EOTS platform.
"""



class DashboardModeType(str, Enum):
    """Defines the different dashboard modes available in the EOTS UI."""
    STANDARD = "standard"; ADVANCED = "advanced"; AI = "ai"; VOLATILITY = "volatility"
    STRUCTURE = "structure"; TIMEDECAY = "timedecay"; CUSTOM = "custom"

 
class ChartType(str, Enum):
    """Supported chart types for the dashboard visualizations."""
    LINE = "line"; BAR = "bar"; SCATTER = "scatter"; HEATMAP = "heatmap"; GAUGE = "gauge"
    CANDLESTICK = "candlestick"; HISTOGRAM = "histogram"; PIE = "pie"; TABLE = "table"

class DashboardModeUIDetail(BaseModel):
    """Detailed configuration for a specific dashboard mode's display settings, including UI elements."""
    module_name: str = Field(..., description="Python module containing the dashboard mode implementation.")
    charts: List[str] = Field(default_factory=list, description="List of chart/component identifiers to display in this mode.")
    label: str = Field("", description="User-friendly display name for this mode.")
    description: str = Field("", description="Description of what this dashboard mode shows.")
    icon: str = Field("", description="Icon identifier for the mode selector.")
    model_config = ConfigDict(extra='allow')  # Allow flexible configuration

class ChartMargin(BaseModel):
    """Chart margin configuration."""
    t: int = Field(default=30, ge=0, description="Top margin in pixels")
    b: int = Field(default=30, ge=0, description="Bottom margin in pixels")
    l: int = Field(default=20, ge=0, description="Left margin in pixels")
    r: int = Field(default=20, ge=0, description="Right margin in pixels")
    model_config = ConfigDict(extra='allow')  # Allow flexible configuration


class ChartLayoutConfigV2_5(BaseModel):
    """Unified chart layout configuration for consistent visualization across EOTS."""
    title_text: str = Field(..., description="Chart title text")
    chart_type: ChartType = Field(..., description="Type of chart to render")
    height: int = Field(300, ge=100, le=1000, description="Chart height in pixels")
    width: Union[int, str] = Field("100%", description="Chart width in pixels or percentage")
    x_axis_title: str = Field("", description="X-axis title")
    y_axis_title: str = Field("", description="Y-axis title")
    show_legend: bool = Field(True, description="Whether to show the legend")
    margin: ChartMargin = Field(
        default_factory=ChartMargin,
        description="Chart margins in pixels"
    )
    template: str = Field("plotly_white", description="Plotly template to use")
    model_config = ConfigDict(extra='allow')  # Allow flexible configuration

    def to_plotly_layout(self) -> Dict[str, Any]:
        """Convert to Plotly layout dictionary."""
        return {
            "title": {"text": self.title_text},
            "height": self.height,
            "width": self.width if isinstance(self.width, int) else None,
            "xaxis": {"title": self.x_axis_title},
            "yaxis": {"title": self.y_axis_title},
            "showlegend": self.show_legend,
            "margin": self.margin.model_dump(),
            "template": self.template
        }

class DashboardConfigV2_5(BaseModel):
    """Main configuration for the EOTS dashboard, defining modes, charts, and component behavior."""
    available_modes: Dict[DashboardModeType, DashboardModeUIDetail] = Field(default_factory=dict, description="Available dashboard modes and their configurations.")
    default_mode: DashboardModeType = Field(DashboardModeType.STANDARD, description="Default dashboard mode loaded on startup.")
    chart_configs: Dict[str, ChartLayoutConfigV2_5] = Field(default_factory=dict, description="Configuration for individual charts by their unique identifier.")
    control_panel: Optional[ControlPanelParametersV2_5] = Field(default=None, description="Default parameters for the control panel.")

# DashboardModeSettings is defined in configuration_schemas.py
# ChartType, ChartLayoutConfigV2_5, ControlPanelParametersV2_5 moved to configuration_schemas.py

class ComponentComplianceV2_5(BaseModel):
    """Tracks compliance and performance metrics for dashboard components."""
    component_id: str = Field(..., description="Unique identifier for the dashboard component.")
    respects_filters: bool = Field(True, description="Whether this component respects global filters.")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics (e.g., load_time, render_time).")
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of last update (UTC).")
    model_config = ConfigDict(extra='allow')  # Allow flexible configuration

class DashboardStateV2_5(BaseModel):
    """Tracks the current state and user interactions with the dashboard UI."""
    current_mode: Optional[DashboardModeType] = Field(None, description="Currently active dashboard mode.")
    active_components: List[str] = Field(default_factory=list, description="List of currently active component IDs.")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User-specific UI preferences and settings.")
    last_interaction: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of last user interaction (UTC).")
    model_config = ConfigDict(extra='allow')  # Allow flexible configuration

# --- PORTED: AIHubComplianceReportV2_5 for compliance tracking (from deprecated_files/eots_schemas_v2_5.py) ---
class AIHubComplianceReportV2_5(BaseModel):
    """PYDANTIC-FIRST: AI Hub compliance reporting and validation. Tracks which dashboard components and metrics respect control panel filters (DTE, price range, refresh interval, etc.)."""
    control_panel_params: ControlPanelParametersV2_5 = Field(..., description="Current control panel parameters")
    options_contracts_filtered: int = Field(default=0, description="Number of options contracts after filtering")
    options_contracts_total: int = Field(default=0, description="Total options contracts before filtering")
    dte_filter_applied: bool = Field(default=False, description="Whether DTE filter was applied")
    price_filter_applied: bool = Field(default=False, description="Whether price range filter was applied")
    components_respecting_filters: List[str] = Field(default_factory=list, description="Dashboard components respecting filters")
    components_not_respecting_filters: List[str] = Field(default_factory=list, description="Dashboard components NOT respecting filters")
    metrics_calculated_with_filters: List[str] = Field(default_factory=list, description="Metrics calculated using filtered data")
    metrics_using_raw_data: List[str] = Field(default_factory=list, description="Metrics using unfiltered data")
    compliance_score: float = Field(default=0.0, description="Overall compliance score (0.0-1.0)", ge=0.0, le=1.0)
    compliance_status: str = Field(default="UNKNOWN", description="Compliance status", pattern="^(COMPLIANT|PARTIAL|NON_COMPLIANT|UNKNOWN)$")
    compliance_issues: List[str] = Field(default_factory=list, description="List of compliance issues found")
    compliance_recommendations: List[str] = Field(default_factory=list, description="Recommendations to improve compliance")
    report_timestamp: datetime = Field(default_factory=datetime.now, description="When compliance report was generated")
    data_timestamp: Optional[datetime] = Field(None, description="Timestamp of analyzed data")
    model_config = ConfigDict(extra='allow')  # Allow flexible configuration

    def calculate_compliance_score(self) -> float:
        """
        Calculate overall compliance score using tracking data.
        This method should be integrated with a compliance tracker for accurate scoring.
        """
        total_components = len(self.components_respecting_filters) + len(self.components_not_respecting_filters)
        if total_components == 0:
            self.compliance_score = 0.0
            self.compliance_status = "UNKNOWN"
            return 0.0
        component_score = len(self.components_respecting_filters) / total_components
        filter_score = 0.0
        if self.dte_filter_applied:
            filter_score += 0.5
        if self.price_filter_applied:
            filter_score += 0.5
        total_metrics = len(self.metrics_calculated_with_filters) + len(self.metrics_using_raw_data)
        metrics_score = 0.0
        if total_metrics > 0:
            metrics_score = len(self.metrics_calculated_with_filters) / total_metrics
        final_score = (component_score * 0.5) + (filter_score * 0.3) + (metrics_score * 0.2)
        self.compliance_score = round(final_score, 3)
        if self.compliance_score >= 0.9:
            self.compliance_status = "COMPLIANT"
        elif self.compliance_score >= 0.7:
            self.compliance_status = "PARTIAL"
        else:
            self.compliance_status = "NON_COMPLIANT"
        return self.compliance_score

    def add_compliance_issue(self, issue: str) -> None:
        """Add a compliance issue to the report."""
        if issue not in self.compliance_issues:
            self.compliance_issues.append(issue)

    def add_compliance_recommendation(self, recommendation: str) -> None:
        """Add a compliance recommendation to the report."""
        if recommendation not in self.compliance_recommendations:
            self.compliance_recommendations.append(recommendation)

# =============================================================================
# FROM ui_component_schemas.py
# =============================================================================
"""
This module defines specific Pydantic models to replace Dict[str, Any] patterns
in dashboard and UI configurations, ensuring type safety and validation.
"""


# Component Status Models
class ComponentStatus(BaseModel):
    """Status information for a system component."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status (e.g., 'healthy', 'warning', 'error', 'offline')")
    message: Optional[str] = Field(None, description="Additional status message")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class SystemComponentStatuses(BaseModel):
    """Collection of system component statuses."""
    components: List[ComponentStatus] = Field(default_factory=list, description="List of component statuses")
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format for backward compatibility."""
        return {comp.name: comp.status for comp in self.components}
    
    def add_component(self, name: str, status: str, message: Optional[str] = None, last_updated: Optional[str] = None):
        """Add or update a component status."""
        # Remove existing component with same name
        self.components = [comp for comp in self.components if comp.name != name]
        # Add new component
        self.components.append(ComponentStatus(
            name=name, 
            status=status, 
            message=message, 
            last_updated=last_updated
        ))

# Signal Activation Models
class SignalActivationSettings(BaseModel):
    """Settings for enabling/disabling signal generation routines."""
    enable_all_signals: bool = Field(True, description="Master toggle for all signal generation")
    flow_signals: bool = Field(True, description="Enable options flow signals")
    regime_signals: bool = Field(True, description="Enable market regime signals")
    sentiment_signals: bool = Field(True, description="Enable sentiment signals")
    volatility_signals: bool = Field(True, description="Enable volatility signals")
    momentum_signals: bool = Field(True, description="Enable momentum signals")
    
    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary format for backward compatibility."""
        return {
            "EnableAllSignals": self.enable_all_signals,
            "FlowSignals": self.flow_signals,
            "RegimeSignals": self.regime_signals,
            "SentimentSignals": self.sentiment_signals,
            "VolatilitySignals": self.volatility_signals,
            "MomentumSignals": self.momentum_signals
        }


class RegimeColor(str, Enum):
    """Color options for market regime indicators."""
    DEFAULT = "secondary"
    BULLISH = "success"
    BEARISH = "danger"
    NEUTRAL = "info"
    UNCLEAR = "warning"


class RegimeIndicatorConfig(BaseModel):
    """Configuration for Market Regime indicator display."""
    title: str = Field(default="Market Regime", description="Display title for the regime indicator")
    regime_colors: Dict[str, RegimeColor] = Field(
        default_factory=lambda: {
            "default": RegimeColor.DEFAULT,
            "bullish": RegimeColor.BULLISH,
            "bearish": RegimeColor.BEARISH,
            "neutral": RegimeColor.NEUTRAL,
            "unclear": RegimeColor.UNCLEAR
        },
        description="Color mapping for different market regimes"
    )
    model_config = ConfigDict(extra='forbid')





class GaugeStep(BaseModel):
    """Configuration for gauge step ranges and colors."""
    range: List[float] = Field(..., description="Range values [min, max] for this step")
    color: str = Field(..., description="Color code for this range")
    model_config = ConfigDict(extra='forbid')


class FlowGaugeConfig(BaseModel):
    """Configuration for flow gauge visualizations."""
    height: int = Field(default=200, ge=50, le=500, description="Gauge height in pixels")
    indicator_font_size: int = Field(default=16, ge=8, le=32, description="Font size for indicators")
    number_font_size: int = Field(default=24, ge=12, le=48, description="Font size for numbers")
    axis_range: List[float] = Field(default=[-3, 3], description="Axis range [min, max]")
    threshold_line_color: str = Field(default="white", description="Color for threshold lines")
    margin: ChartMargin = Field(default_factory=lambda: ChartMargin(t=60, b=40, l=20, r=20))
    steps: List[GaugeStep] = Field(
        default_factory=lambda: [
            GaugeStep(range=[-3, -2], color="#d62728"),
            GaugeStep(range=[-2, -0.5], color="#ff9896"),
            GaugeStep(range=[-0.5, 0.5], color="#aec7e8"),
            GaugeStep(range=[0.5, 2], color="#98df8a"),
            GaugeStep(range=[2, 3], color="#2ca02c")
        ],
        description="Gauge step configurations"
    )
    model_config = ConfigDict(extra='forbid')


class GibGaugeConfig(BaseModel):
    """Configuration for GIB gauge visualizations."""
    height: int = Field(default=180, ge=50, le=500, description="Gauge height in pixels")
    indicator_font_size: int = Field(default=14, ge=8, le=32, description="Font size for indicators")
    number_font_size: int = Field(default=20, ge=12, le=48, description="Font size for numbers")
    axis_range: List[float] = Field(default=[-1, 1], description="Axis range [min, max]")
    dollar_axis_range: List[float] = Field(default=[-1000000, 1000000], description="Dollar axis range [min, max]")
    threshold_line_color: str = Field(default="white", description="Color for threshold lines")
    margin: ChartMargin = Field(default_factory=lambda: ChartMargin(t=50, b=30, l=15, r=15))
    steps: List[GaugeStep] = Field(
        default_factory=lambda: [
            GaugeStep(range=[-1, -0.5], color="#d62728"),
            GaugeStep(range=[-0.5, -0.1], color="#ff9896"),
            GaugeStep(range=[-0.1, 0.1], color="#aec7e8"),
            GaugeStep(range=[0.1, 0.5], color="#98df8a"),
            GaugeStep(range=[0.5, 1], color="#2ca02c")
        ],
        description="Gauge step configurations"
    )
    dollar_steps: List[GaugeStep] = Field(
        default_factory=lambda: [
            GaugeStep(range=[-1000000, -500000], color="#d62728"),
            GaugeStep(range=[-500000, -100000], color="#ff9896"),
            GaugeStep(range=[-100000, 100000], color="#aec7e8"),
            GaugeStep(range=[100000, 500000], color="#98df8a"),
            GaugeStep(range=[500000, 1000000], color="#2ca02c")
        ],
        description="Dollar gauge step configurations"
    )
    model_config = ConfigDict(extra='forbid')


class MiniHeatmapConfig(BaseModel):
    """Configuration for mini-heatmap components."""
    height: int = Field(default=150, ge=50, le=500, description="Heatmap height in pixels")
    colorscale: str = Field(default="RdYlGn", description="Color scale for the heatmap")
    margin: ChartMargin = Field(default_factory=lambda: ChartMargin(t=50, b=30, l=40, r=40))
    model_config = ConfigDict(extra='forbid')


class TableCellStyle(BaseModel):
    """Style configuration for table cells."""
    textAlign: str = Field(default="left", description="Text alignment")
    padding: str = Field(default="5px", description="Cell padding")
    minWidth: str = Field(default="80px", description="Minimum cell width")
    width: str = Field(default="auto", description="Cell width")
    maxWidth: str = Field(default="200px", description="Maximum cell width")
    model_config = ConfigDict(extra='forbid')


class TableHeaderStyle(BaseModel):
    """Style configuration for table headers."""
    backgroundColor: str = Field(default="rgb(30, 30, 30)", description="Header background color")
    fontWeight: str = Field(default="bold", description="Header font weight")
    color: str = Field(default="white", description="Header text color")
    model_config = ConfigDict(extra='forbid')


class TableDataStyle(BaseModel):
    """Style configuration for table data."""
    backgroundColor: str = Field(default="rgb(50, 50, 50)", description="Data background color")
    color: str = Field(default="white", description="Data text color")
    model_config = ConfigDict(extra='forbid')


class RecommendationsTableConfig(BaseModel):
    """Configuration for the ATIF recommendations table."""
    title: str = Field(default="ATIF Recommendations", description="Table title")
    max_rationale_length: int = Field(default=50, ge=10, le=200, description="Maximum length for rationale text")
    page_size: int = Field(default=5, ge=1, le=50, description="Number of items per page")
    style_cell: TableCellStyle = Field(default_factory=TableCellStyle)
    style_header: TableHeaderStyle = Field(default_factory=TableHeaderStyle)
    style_data: TableDataStyle = Field(default_factory=TableDataStyle)
    model_config = ConfigDict(extra='forbid')


class TickerContextConfig(BaseModel):
    """Configuration for ticker context display area."""
    title: str = Field(default="Ticker Context", description="Display title")
    model_config = ConfigDict(extra='forbid')


class DashboardDefaults(BaseModel):
    """Default settings for dashboard components - FAIL FAST ON MISSING CONFIG."""
    symbol: str = Field(..., min_length=1, description="Default symbol to display - REQUIRED from config")
    refresh_interval_seconds: int = Field(..., gt=0, description="Default refresh interval - REQUIRED from config")
    dte_min: int = Field(..., ge=0, description="Minimum days to expiration - REQUIRED from config")
    dte_max: int = Field(..., gt=0, description="Maximum days to expiration - REQUIRED from config")
    price_range_percent: float = Field(..., gt=0.0, description="Price range percentage - REQUIRED from config")

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

    model_config = ConfigDict(extra='forbid')


class HeatmapSettings(BaseModel):
    """Settings for heatmap visualizations."""
    height: int = Field(default=500, ge=100, description="Chart height in pixels")
    colorscale: str = Field(default="RdYlGn", description="Color scale for the heatmap")
    autosize: bool = Field(default=True, description="Enable automatic sizing")
    responsive: bool = Field(default=True, description="Enable responsive design")
    model_config = ConfigDict(extra='forbid')


class FlowModeSettings(BaseModel):
    """Settings for flow mode visualizations."""
    net_value_heatmap: HeatmapSettings = Field(default_factory=lambda: HeatmapSettings(
        height=600, colorscale="RdYlGn", autosize=False, responsive=False
    ))
    sgdhp_heatmap: HeatmapSettings = Field(default_factory=lambda: HeatmapSettings(
        height=500, colorscale="RdYlGn_r", autosize=True, responsive=True
    ))
    ivsdh_heatmap: HeatmapSettings = Field(default_factory=lambda: HeatmapSettings(
        height=500, colorscale="RdYlBu", autosize=True, responsive=True
    ))
    ugch_heatmap: HeatmapSettings = Field(default_factory=lambda: HeatmapSettings(
        height=500, colorscale="Viridis", autosize=True, responsive=True
    ))
    model_config = ConfigDict(extra='forbid')


class VolatilityGaugeSettings(BaseModel):
    """Settings for volatility gauges."""
    height: int = Field(default=300, ge=100, description="Gauge height in pixels")
    indicator_font_size: int = Field(default=16, ge=8, description="Indicator font size")
    number_font_size: int = Field(default=24, ge=8, description="Number font size")
    margin: ChartMargin = Field(default_factory=lambda: ChartMargin(t=60, b=40, l=20, r=20))
    model_config = ConfigDict(extra='forbid')


class VolatilityChartSettings(BaseModel):
    """Settings for volatility charts."""
    height: int = Field(default=500, ge=100, description="Chart height in pixels")
    colorscale: str = Field(default="RdYlGn", description="Color scale for the chart")
    autosize: bool = Field(default=True, description="Enable automatic sizing")
    responsive: bool = Field(default=True, description="Enable responsive design")
    model_config = ConfigDict(extra='forbid')


class VolatilityModeSettings(BaseModel):
    """Settings for volatility mode visualizations."""
    vri_chart_height: int = Field(default=500, ge=100, description="VRI chart height")
    gauge_height: int = Field(default=300, ge=100, description="Gauge height")
    vri_2_0_chart: VolatilityChartSettings = Field(default_factory=VolatilityChartSettings)
    volatility_gauges: VolatilityGaugeSettings = Field(default_factory=VolatilityGaugeSettings)
    model_config = ConfigDict(extra='forbid')


class DashboardModeSettings(BaseModel):
    """Settings for a dashboard mode."""
    label: str = Field(..., description="Display label for the mode")
    module_name: str = Field(..., description="Module name for the mode")
    charts: List[str] = Field(default_factory=list, description="List of charts in this mode")
    model_config = ConfigDict(extra='forbid')


class ModesDetailConfig(BaseModel):
    """Configuration for all dashboard modes."""
    main: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Main Dashboard",
        module_name="main_dashboard_display_v2_5",
        charts=["regime_display", "flow_gauges", "gib_gauges", "recommendations_table"]
    ))
    flow: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Flow Analysis",
        module_name="flow_mode_display_v2_5",
        charts=["net_value_heatmap_viz", "sgdhp_heatmap_viz", "ivsdh_heatmap_viz", "ugch_heatmap_viz", "net_cust_delta_flow_viz", "net_cust_gamma_flow_viz", "net_cust_vega_flow_viz"]
    ))
    structure: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Structure & Positioning",
        module_name="structure_mode_display_v2_5",
        charts=["mspi_components_viz"]
    ))
    timedecay: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Time Decay & Pinning",
        module_name="time_decay_mode_display_v2_5",
        charts=["tdpi_displays", "vci_strike_charts"]
    ))
    advanced: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Advanced Flow",
        module_name="advanced_flow_mode_v2_5",
        charts=["vapi_gauge", "dwfd_gauge", "tw_laf_gauge"]
    ))
    volatility: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="Volatility Deep Dive",
        module_name="volatility_mode_display_v2_5",
        charts=["vri_2_0_strike_profile", "vri_0dte_strike_viz", "vfi_0dte_agg_viz", "vvr_0dte_agg_viz", "vci_0dte_agg_viz"]
    ))
    ai: DashboardModeSettings = Field(default_factory=lambda: DashboardModeSettings(
        label="AI Intelligence Hub",
        module_name="ai_dashboard.ai_dashboard_display_v2_5",
        charts=["ai_market_analysis", "ai_recommendations", "ai_insights", "ai_regime_context", "ai_performance_tracker"]
    ))
    model_config = ConfigDict(extra='forbid')


class MainDashboardSettings(BaseModel):
    """Settings for main dashboard components."""
    regime_indicator: RegimeIndicatorConfig = Field(default_factory=RegimeIndicatorConfig)
    flow_gauge: "FlowGaugeConfig" = Field(default_factory=lambda: FlowGaugeConfig())
    gib_gauge: "GibGaugeConfig" = Field(default_factory=lambda: GibGaugeConfig())
    mini_heatmap: "MiniHeatmapConfig" = Field(default_factory=lambda: MiniHeatmapConfig())
    ticker_context: "TickerContextConfig" = Field(default_factory=lambda: TickerContextConfig())
    recommendations_table: "RecommendationsTableConfig" = Field(default_factory=lambda: RecommendationsTableConfig())
    model_config = ConfigDict(extra='forbid')


class DashboardServerConfig(BaseModel):
    """Configuration for dashboard server settings."""
    host: str = Field(default="127.0.0.1", description="Server host address")
    port: int = Field(default=8050, ge=1024, le=65535, description="Server port number")
    debug: bool = Field(default=False, description="Enable debug mode")
    dev_tools_hot_reload: bool = Field(default=True, description="Enable hot reload in development")
    auto_refresh_seconds: int = Field(default=30, ge=1, description="Auto refresh interval in seconds")
    timestamp_format: str = Field(default="%Y-%m-%d %H:%M:%S %Z", description="Timestamp format string")
    defaults: DashboardDefaults = Field(default_factory=DashboardDefaults)
    modes_detail_config: ModesDetailConfig = Field(default_factory=ModesDetailConfig)
    flow_mode_settings: FlowModeSettings = Field(default_factory=FlowModeSettings)
    volatility_mode_settings: VolatilityModeSettings = Field(default_factory=VolatilityModeSettings)
    main_dashboard_settings: MainDashboardSettings = Field(default_factory=MainDashboardSettings)
    model_config = ConfigDict(extra='forbid')