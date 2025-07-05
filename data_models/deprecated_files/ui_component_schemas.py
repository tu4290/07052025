"""Pydantic models for UI component configurations.

This module defines specific Pydantic models to replace Dict[str, Any] patterns
in dashboard and UI configurations, ensuring type safety and validation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


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


class ChartMargin(BaseModel):
    """Chart margin configuration."""
    t: int = Field(default=30, ge=0, description="Top margin in pixels")
    b: int = Field(default=30, ge=0, description="Bottom margin in pixels")
    l: int = Field(default=20, ge=0, description="Left margin in pixels")
    r: int = Field(default=20, ge=0, description="Right margin in pixels")
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
    """Default settings for dashboard components."""
    symbol: str = Field(default="SPY", description="Default symbol to display")
    refresh_interval_seconds: int = Field(default=30, ge=1, description="Default refresh interval")
    dte_min: int = Field(default=0, ge=0, description="Minimum days to expiration")
    dte_max: int = Field(default=5, ge=0, description="Maximum days to expiration")
    price_range_percent: float = Field(default=5.0, ge=0, description="Price range percentage")
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
        module_name="ai_dashboard",
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