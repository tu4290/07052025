"""
Dashboard Mode Models for EOTS v2.5
====================================

This module contains all Pydantic models specifically for dashboard modes,
implementing the unified architecture pattern for consistent mode behavior.

Key Features:
- Base state models for all modes
- Control panel integration models
- Filtered data bundle structures
- Mode-specific state models

Author: EOTS v2.5 Development Team
Version: 2.5.0 (Pydantic-First Architecture)
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Union
from datetime import datetime
from datetime import datetime
import logging

# Import core models
from .core_models import FinalAnalysisBundleV2_5
from .configuration_models import ControlPanelParametersV2_5

logger = logging.getLogger(__name__)


# =============================================================================
# BASE MODELS
# =============================================================================

class BaseModeState(BaseModel):
    """
    Base state model for all dashboard modes.
    
    This provides the foundation for consistent mode behavior across
    the entire dashboard system.
    """
    target_symbol: str = Field(..., description="Target trading symbol")
    control_panel_params: ControlPanelParametersV2_5 = Field(..., description="Control panel parameters")
    analysis_bundle: FinalAnalysisBundleV2_5 = Field(..., description="Analysis data bundle")
    error_message: Optional[str] = Field(None, description="Error message if any")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    mode_enabled: bool = Field(True, description="Whether this mode is enabled")
    
    @field_validator('target_symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format"""
        if not v or not v.strip():
            raise ValueError("target_symbol cannot be empty")
        return v.upper().strip()
    
    @field_validator('warnings')
    @classmethod
    def validate_warnings(cls, v: List[str]) -> List[str]:
        """Ensure warnings are non-empty strings"""
        return [w for w in v if w and w.strip()]
    
    model_config = ConfigDict(extra='forbid', frozen=False)


class BaseModeConfig(BaseModel):
    """
    Base configuration model for all dashboard modes.
    
    Provides common configuration options that all modes should support.
    """
    enabled: bool = Field(True, description="Enable this mode")
    refresh_interval_seconds: int = Field(30, ge=1, le=3600, description="Refresh interval in seconds")
    chart_height: int = Field(400, ge=200, le=1200, description="Default chart height in pixels")
    show_timestamps: bool = Field(True, description="Show timestamps on charts")
    show_control_panel_info: bool = Field(True, description="Show control panel filter info")
    theme: str = Field("dark", description="Chart theme")
    
    model_config = ConfigDict(extra='forbid')


class FilteredDataBundle(BaseModel):
    """
    Data bundle filtered by control panel parameters.
    
    This centralizes all filtering logic and provides a consistent
    interface for filtered data across all modes.
    """
    # Filtered data collections
    filtered_strikes: List[Any] = Field(default_factory=list, description="Strikes within price range")
    filtered_contracts: List[Any] = Field(default_factory=list, description="Contracts within DTE range")
    
    # Price range information
    price_range_min: float = Field(..., description="Minimum price in range")
    price_range_max: float = Field(..., description="Maximum price in range")
    current_price: float = Field(..., description="Current underlying price")
    price_range_percent: float = Field(..., description="Price range percentage used")
    
    # DTE range information
    dte_min: int = Field(..., description="Minimum DTE used for filtering")
    dte_max: int = Field(..., description="Maximum DTE used for filtering")
    
    # Filtering statistics
    total_strikes_available: int = Field(0, description="Total strikes before filtering")
    total_contracts_available: int = Field(0, description="Total contracts before filtering")
    strikes_filtered_count: int = Field(0, description="Number of strikes after filtering")
    contracts_filtered_count: int = Field(0, description="Number of contracts after filtering")
    
    # Metadata
    filter_applied_at: datetime = Field(default_factory=datetime.now, description="When filtering was applied")
    
    @field_validator('price_range_min', 'price_range_max', 'current_price')
    @classmethod
    def validate_prices_positive(cls, v: float) -> float:
        """Ensure prices are positive"""
        if v <= 0:
            raise ValueError("Prices must be positive")
        return v
    
    @field_validator('dte_min', 'dte_max')
    @classmethod
    def validate_dte_range(cls, v: int) -> int:
        """Ensure DTE values are non-negative"""
        if v < 0:
            raise ValueError("DTE values must be non-negative")
        return v
    
    def get_filter_summary(self) -> str:
        """Get a human-readable summary of applied filters"""
        return (
            f"Symbol: {self.current_price:.2f} "
            f"| Price Range: {self.price_range_min:.2f}-{self.price_range_max:.2f} ({self.price_range_percent}%) "
            f"| DTE: {self.dte_min}-{self.dte_max} "
            f"| Strikes: {self.strikes_filtered_count}/{self.total_strikes_available} "
            f"| Contracts: {self.contracts_filtered_count}/{self.total_contracts_available}"
        )
    
    model_config = ConfigDict(extra='forbid')


# =============================================================================
# MODE-SPECIFIC STATE MODELS
# =============================================================================

class MainDashboardState(BaseModeState):
    """State model for main dashboard mode"""
    filtered_data: Optional[FilteredDataBundle] = Field(None, description="Filtered data bundle")
    # REMOVED: Dict usage - replaced with strict Pydantic models
    regime_indicator_enabled: bool = Field(default=True, description="Enable regime indicators")
    flow_gauges_enabled: bool = Field(default=True, description="Enable flow gauges")
    key_metrics_enabled: bool = Field(default=True, description="Enable key metrics")
    
    model_config = ConfigDict(extra='forbid')


class AdvancedFlowModeState(BaseModeState):
    """State model for Advanced Flow Mode"""
    filtered_data: Optional[FilteredDataBundle] = Field(None, description="Filtered data bundle")
    # REMOVED: Dict usage - replaced with strict Pydantic models
    flow_metrics_enabled: bool = Field(default=True, description="Enable advanced flow metrics")
    volume_profile_enabled: bool = Field(default=True, description="Enable volume profile")
    flow_heatmap_enabled: bool = Field(default=True, description="Enable flow heatmap")
    net_flow_analysis_enabled: bool = Field(default=True, description="Enable net flow analysis")
    
    model_config = ConfigDict(extra='forbid')


class FlowModeState(BaseModeState):
    """State model for Flow Mode"""
    filtered_data: Optional[FilteredDataBundle] = Field(None, description="Filtered data bundle")
    # REMOVED: Dict usage - replaced with strict Pydantic models
    net_value_heatmap_enabled: bool = Field(default=True, description="Enable net value heatmap")
    flow_gauges_enabled: bool = Field(default=True, description="Enable flow gauges")
    sgdhp_heatmap_enabled: bool = Field(default=True, description="Enable SGDHP heatmap")
    ivsdh_heatmap_enabled: bool = Field(default=True, description="Enable IVSDH heatmap")
    ugch_heatmap_enabled: bool = Field(default=True, description="Enable UGCH heatmap")
    
    model_config = ConfigDict(extra='forbid')


class StructureModeState(BaseModeState):
    """State model for Structure Mode"""
    filtered_data: Optional[FilteredDataBundle] = Field(None, description="Filtered data bundle")
    # REMOVED: Dict usage - replaced with strict Pydantic models
    mspi_profile_enabled: bool = Field(default=True, description="Enable MSPI profile")
    key_levels_enabled: bool = Field(default=True, description="Enable key levels")
    dealer_positioning_enabled: bool = Field(default=True, description="Enable dealer positioning")
    amspi_heatmap_enabled: bool = Field(default=True, description="Enable A-MSPI heatmap")
    esdag_charts_enabled: bool = Field(default=True, description="Enable ES-DAG charts")
    adag_strike_enabled: bool = Field(default=True, description="Enable A-DAG strike")
    asai_assi_enabled: bool = Field(default=True, description="Enable A-SAI/A-SSI")
    
    model_config = ConfigDict(extra='forbid')


class VolatilityModeState(BaseModeState):
    """State model for Volatility Mode"""
    filtered_data: Optional[FilteredDataBundle] = Field(None, description="Filtered data bundle")
    # REMOVED: Dict usage - replaced with strict Pydantic models
    volatility_surface_enabled: bool = Field(default=True, description="Enable volatility surface")
    vri_profile_enabled: bool = Field(default=True, description="Enable VRI profile")
    volatility_gauges_enabled: bool = Field(default=True, description="Enable volatility gauges")
    volatility_heatmap_enabled: bool = Field(default=True, description="Enable volatility heatmap")
    volatility_context_enabled: bool = Field(default=True, description="Enable volatility context")
    
    model_config = ConfigDict(extra='forbid')


class TimeDecayModeState(BaseModeState):
    """State model for Time Decay Mode"""
    filtered_data: Optional[FilteredDataBundle] = Field(None, description="Filtered data bundle")
    # REMOVED: Dict usage - replaced with strict Pydantic models
    decay_profile_enabled: bool = Field(default=True, description="Enable time decay profile")
    expiry_calendar_enabled: bool = Field(default=True, description="Enable expiry calendar")
    theta_analysis_enabled: bool = Field(default=True, description="Enable theta analysis")
    time_context_enabled: bool = Field(default=True, description="Enable time context")
    decay_heatmap_enabled: bool = Field(default=True, description="Enable decay heatmap")
    
    model_config = ConfigDict(extra='forbid')


class MonitoringDashboardState(BaseModeState):
    """State model for Monitoring Dashboard"""
    # REMOVED: Dict usage - replaced with strict Pydantic models
    system_health_enabled: bool = Field(default=True, description="Enable system health metrics")
    performance_metrics_enabled: bool = Field(default=True, description="Enable performance metrics")
    error_logs: Optional[List[str]] = Field(default_factory=list, description="Recent error logs")
    alert_status_enabled: bool = Field(default=True, description="Enable alert status")
    
    model_config = ConfigDict(extra='forbid')


class AIDashboardState(BaseModeState):
    """State model for AI Dashboard"""
    filtered_data: Optional[FilteredDataBundle] = Field(None, description="Filtered data bundle")
    # REMOVED: Dict usage - replaced with strict Pydantic models
    ai_recommendations_enabled: bool = Field(default=True, description="Enable AI recommendations")
    market_compass_enabled: bool = Field(default=True, description="Enable market compass")
    flow_intelligence_enabled: bool = Field(default=True, description="Enable flow intelligence")
    volatility_gamma_enabled: bool = Field(default=True, description="Enable volatility gamma panel")
    moe_system_enabled: bool = Field(default=True, description="Enable MOE system")
    
    model_config = ConfigDict(extra='forbid')


# =============================================================================
# MODE-SPECIFIC CONFIG MODELS
# =============================================================================

class MainDashboardConfig(BaseModeConfig):
    """Configuration for main dashboard mode"""
    show_regime_indicator: bool = Field(True, description="Show market regime indicator")
    show_flow_gauges: bool = Field(True, description="Show flow gauges")
    show_key_metrics: bool = Field(True, description="Show key metrics summary")
    regime_indicator_height: int = Field(150, description="Regime indicator height")
    
    model_config = ConfigDict(extra='forbid')


class AdvancedFlowModeConfig(BaseModeConfig):
    """Configuration for advanced flow mode"""
    show_volume_profile: bool = Field(True, description="Show volume profile")
    show_flow_heatmap: bool = Field(True, description="Show flow heatmap")
    heatmap_height: int = Field(500, description="Heatmap height")
    volume_profile_height: int = Field(400, description="Volume profile height")
    
    model_config = ConfigDict(extra='forbid')


class FlowModeConfig(BaseModeConfig):
    """Configuration for flow mode"""
    net_value_heatmap_height: int = Field(500, description="Net value heatmap height")
    sgdhp_heatmap_height: int = Field(500, description="SGDHP heatmap height")
    ivsdh_heatmap_height: int = Field(500, description="IVSDH heatmap height")
    ugch_heatmap_height: int = Field(500, description="UGCH heatmap height")
    show_all_heatmaps: bool = Field(True, description="Show all heatmaps")
    
    model_config = ConfigDict(extra='forbid')


class StructureModeConfig(BaseModeConfig):
    """Configuration for structure mode"""
    mspi_chart_height: int = Field(400, description="MSPI chart height")
    key_levels_table_height: int = Field(300, description="Key levels table height")
    show_dealer_positioning: bool = Field(True, description="Show dealer positioning")
    show_key_levels: bool = Field(True, description="Show key levels table")
    asai_assi_chart_height: int = Field(350, description="A-SAI/A-SSI chart height")
    
    model_config = ConfigDict(extra='forbid')


class VolatilityModeConfig(BaseModeConfig):
    """Configuration for volatility mode"""
    surface_chart_height: int = Field(500, description="Volatility surface height")
    vri_chart_height: int = Field(400, description="VRI chart height")
    show_volatility_gauges: bool = Field(True, description="Show volatility gauges")
    show_volatility_context: bool = Field(True, description="Show volatility context")
    
    model_config = ConfigDict(extra='forbid')


class TimeDecayModeConfig(BaseModeConfig):
    """Configuration for time decay mode"""
    decay_chart_height: int = Field(400, description="Decay chart height")
    expiry_calendar_height: int = Field(300, description="Expiry calendar height")
    show_theta_analysis: bool = Field(True, description="Show theta analysis")
    show_time_context: bool = Field(True, description="Show time context")
    
    model_config = ConfigDict(extra='forbid')


class MonitoringDashboardConfig(BaseModeConfig):
    """Configuration for monitoring dashboard"""
    max_error_logs: int = Field(100, description="Maximum error logs to display")
    alert_refresh_interval: int = Field(10, description="Alert refresh interval in seconds")
    show_system_health: bool = Field(True, description="Show system health metrics")
    
    model_config = ConfigDict(extra='forbid')


class AIDashboardConfig(BaseModeConfig):
    """Configuration for AI dashboard"""
    ai_recommendations_height: int = Field(400, description="AI recommendations panel height")
    market_compass_height: int = Field(500, description="Market compass height")
    flow_intelligence_height: int = Field(400, description="Flow intelligence panel height")
    volatility_gamma_height: int = Field(400, description="Volatility gamma panel height")
    show_moe_system: bool = Field(True, description="Show MOE system components")
    show_ai_insights: bool = Field(True, description="Show AI insights panel")
    
    model_config = ConfigDict(extra='forbid')


# =============================================================================
# UTILITY MODELS
# =============================================================================

class ModeValidationResult(BaseModel):
    """Result of mode input validation"""
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    model_config = ConfigDict(extra='forbid')


class ChartDataModel(BaseModel):
    """Base model for chart data"""
    chart_type: str = Field(..., description="Type of chart")
    # REMOVED: Dict usage - replaced with strict Pydantic models
    chart_type: str = Field(..., description="Chart type identifier")
    data_source: str = Field(..., description="Data source identifier")
    title: str = Field(..., description="Chart title")
    height: int = Field(400, description="Chart height")
    
    model_config = ConfigDict(extra='forbid')


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base models
    'BaseModeState',
    'BaseModeConfig', 
    'FilteredDataBundle',
    
    # Mode-specific state models
    'MainDashboardState',
    'AdvancedFlowModeState',
    'FlowModeState',
    'StructureModeState',
    'VolatilityModeState',
    'TimeDecayModeState',
    'MonitoringDashboardState',
    'AIDashboardState',
    
    # Mode-specific config models
    'MainDashboardConfig',
    'AdvancedFlowModeConfig',
    'FlowModeConfig',
    'StructureModeConfig',
    'VolatilityModeConfig',
    'TimeDecayModeConfig',
    'MonitoringDashboardConfig',
    'AIDashboardConfig',
    
    # Utility models
    'ModeValidationResult',
    'ChartDataModel',
]