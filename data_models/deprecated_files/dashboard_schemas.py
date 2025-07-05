"""
Pydantic models for the EOTS v2.5 Dashboard and UI components.

This module defines the data structures used for dashboard configurations,
UI components, and display settings across the EOTS platform.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union 
from pydantic import BaseModel, Field, ConfigDict
from data_models.configuration_schemas import ControlPanelParametersV2_5  # Added for compliance model
from data_models.ui_component_schemas import ChartMargin

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
    model_config = ConfigDict(extra='forbid')

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
    model_config = ConfigDict(extra='forbid')

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
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last update (UTC).")
    model_config = ConfigDict(extra='forbid')

class DashboardStateV2_5(BaseModel):
    """Tracks the current state and user interactions with the dashboard UI."""
    current_mode: Optional[DashboardModeType] = Field(None, description="Currently active dashboard mode.")
    active_components: List[str] = Field(default_factory=list, description="List of currently active component IDs.")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User-specific UI preferences and settings.")
    last_interaction: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last user interaction (UTC).")
    model_config = ConfigDict(extra='forbid')

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
    model_config = ConfigDict(extra='forbid')

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