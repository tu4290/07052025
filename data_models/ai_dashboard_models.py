# data_models/ai_dashboard_models.py
"""
Pydantic V2 Models for the EOTS AI Hub Dashboard
=================================================

PYDANTIC-FIRST & ZERO DICT ACCEPTANCE.

This module defines a comprehensive set of Pydantic V2 models to structure all data,
configuration, and state for the AI Hub dashboard. It ensures end-to-end type safety,
eliminates the use of dictionaries for styling and state management, and provides a
validated foundation for all UI components.

Models are organized into logical sections:
1.  Enums: For controlled vocabularies.
2.  Styling & Layout: To replace CSS-in-dict patterns.
3.  Chart & Visualization: For configuring plots and gauges.
4.  Component & Panel Data: For individual UI components like recommendations and the market compass.
5.  System & Integration: For monitoring backend systems like HuiHui experts.
6.  Master State Models: Top-level models to manage the entire dashboard state.
"""

from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, PositiveFloat, conint, confloat

# Import from existing data models to ensure integration
from .core_models import AdvancedOptionsMetricsV2_5
from .ai_ml_models import MOEUnifiedResponseV2_5, MOEExpertResponseV2_5

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Enums for Controlled Vocabularies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PanelType(str, Enum):
    """Defines the types of panels available in the AI Hub."""
    MARKET_COMPASS = "market_compass"
    TRADE_RECOMMENDATIONS = "trade_recommendations"
    MARKET_ANALYSIS = "market_analysis"
    FLOW_INTELLIGENCE = "flow_intelligence"
    VOLATILITY_GAMMA = "volatility_gamma"
    CUSTOM_FORMULAS = "custom_formulas"
    DATA_PIPELINE_MONITOR = "data_pipeline_monitor"
    HUIHUI_EXPERTS_MONITOR = "huihui_experts_monitor"
    PERFORMANCE_MONITOR = "performance_monitor"
    ALERTS_STATUS = "alerts_status"
    PERSISTENT_REGIME = "persistent_regime"

class ComponentStatus(str, Enum):
    """Represents the operational status of a component or data feed."""
    OK = "OK"
    WARNING = "Warning"
    ERROR = "Error"
    UNKNOWN = "Unknown"
    LOADING = "Loading"

class RecommendationStrength(str, Enum):
    """Defines the conviction level of a trade recommendation."""
    HIGH = "High Conviction"
    MEDIUM = "Medium Conviction"
    LOW = "Low Conviction"
    SPECULATIVE = "Speculative"

class CompassTheme(str, Enum):
    """Defines the visual theme for the Market Compass."""
    ELITE = "elite"
    PROFESSIONAL = "professional"
    MINIMALIST = "minimalist"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Component Styling & Layout Models (Replaces dicts for CSS)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class StyleModel(BaseModel):
    """A flexible base model for CSS-like styling properties. Forbids extra fields."""
    model_config = ConfigDict(extra='forbid')

class TypographyStyle(StyleModel):
    """Defines text styling properties."""
    fontSize: Optional[str] = None
    fontWeight: Optional[str] = None
    color: Optional[str] = None
    lineHeight: Optional[str] = None
    textAlign: Optional[str] = None

class CardStyle(StyleModel):
    """Defines styling for card components."""
    backgroundColor: str
    color: str
    padding: str
    borderRadius: str
    border: Optional[str] = None
    boxShadow: Optional[str] = None

class BadgeStyle(StyleModel):
    """Defines styling for badge components."""
    backgroundColor: str
    color: str
    padding: str
    borderRadius: str
    fontSize: str
    fontWeight: str

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Chart & Visualization Configuration Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GaugeConfigModel(BaseModel):
    """Configuration for a single gauge visualization."""
    value: float = 0.0
    title: str
    range_min: float
    range_max: float
    color_scheme: Optional[str] = "Viridis"
    height: int = 150
    model_config = ConfigDict(extra='forbid')

class ChartLayoutConfig(BaseModel):
    """Configuration for a Plotly chart layout."""
    height: Optional[int] = None
    width: Optional[int] = None
    title_text: Optional[str] = None
    showlegend: bool = False
    paper_bgcolor: str = 'rgba(0,0,0,0)'
    plot_bgcolor: str = 'rgba(0,0,0,0)'
    font: Dict[str, Any] = Field(default_factory=lambda: {"color": "#FFFFFF"})
    model_config = ConfigDict(extra='forbid')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Component & Panel Data Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PanelConfigModel(BaseModel):
    """Configuration for a generic dashboard panel."""
    id: str
    title: str
    panel_type: PanelType
    is_visible: bool = True
    status: ComponentStatus = ComponentStatus.LOADING
    last_updated: Optional[datetime] = None
    model_config = ConfigDict(extra='forbid')

class MetricDisplayModel(BaseModel):
    """Data for a single metric display, typically a gauge."""
    name: str
    value: Optional[float] = None
    confidence: Optional[confloat(ge=0.0, le=1.0)] = None
    status: ComponentStatus = ComponentStatus.LOADING
    gauge_config: GaugeConfigModel
    model_config = ConfigDict(extra='forbid')

class MetricsPanelModel(BaseModel):
    """Data for a panel containing multiple metric displays."""
    panel_config: PanelConfigModel
    metrics: List[MetricDisplayModel] = Field(default_factory=list)
    model_config = ConfigDict(extra='forbid')

class AIRecommendationItem(BaseModel):
    """Represents a single, actionable AI-driven trade recommendation."""
    id: str
    strategy_name: str
    direction: Literal["Bullish", "Bearish", "Neutral"]
    instrument_type: Literal["Call", "Put", "Spread", "Underlying"]
    strength: RecommendationStrength
    confidence_score: confloat(ge=0.0, le=1.0)
    rationale: List[str] = Field(..., min_length=1)
    target_price: Optional[PositiveFloat] = None
    stop_loss: Optional[PositiveFloat] = None
    timeframe: str
    generated_at: datetime
    model_config = ConfigDict(extra='forbid')

class AIRecommendationsPanelModel(BaseModel):
    """Data for the entire AI recommendations panel."""
    panel_config: PanelConfigModel
    recommendations: List[AIRecommendationItem] = Field(default_factory=list)
    model_config = ConfigDict(extra='forbid')

class MarketCompassSegment(BaseModel):
    """Defines a single segment or slice of the Market Compass."""
    label: str
    score: confloat(ge=0.0, le=1.0)
    color: str
    description: str
    tactical_advice: str
    model_config = ConfigDict(extra='forbid')

class MarketCompassModel(BaseModel):
    """
    A comprehensive model for the Market Compass visualization.
    This is a user priority and directly feeds the compass component.
    """
    panel_config: PanelConfigModel
    segments: List[MarketCompassSegment] = Field(..., min_length=4)
    overall_directional_bias: confloat(ge=-1.0, le=1.0) # -1 (Bearish) to +1 (Bullish)
    bias_label: Literal["Strong Bearish", "Bearish", "Neutral", "Bullish", "Strong Bullish"]
    active_theme: CompassTheme = CompassTheme.ELITE
    tactical_summary: str
    model_config = ConfigDict(extra='forbid')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. System & Integration Models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ExpertStatusModel(BaseModel):
    """ViewModel for displaying the status of a single HuiHui expert."""
    expert_id: str
    expert_name: str
    status: ComponentStatus
    last_response_time_ms: Optional[float] = None
    last_confidence: Optional[confloat(ge=0.0, le=1.0)] = None
    last_seen: Optional[datetime] = None
    model_config = ConfigDict(extra='forbid')

class HuiHuiExpertsMonitorModel(BaseModel):
    """Data for the panel that monitors the health of all HuiHui experts."""
    panel_config: PanelConfigModel
    experts: List[ExpertStatusModel] = Field(default_factory=list)
    model_config = ConfigDict(extra='forbid')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6. Master State Models for AI Hub
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AIHubLayoutConfig(BaseModel):
    """Defines the layout and visibility of all panels in the AI Hub."""
    row1: List[PanelType] = [PanelType.MARKET_COMPASS, PanelType.TRADE_RECOMMENDATIONS, PanelType.MARKET_ANALYSIS]
    row2: List[PanelType] = [PanelType.FLOW_INTELLIGENCE, PanelType.VOLATILITY_GAMMA, PanelType.CUSTOM_FORMULAS]
    row3: List[PanelType] = [PanelType.DATA_PIPELINE_MONITOR, PanelType.HUIHUI_EXPERTS_MONITOR, PanelType.PERFORMANCE_MONITOR, PanelType.ALERTS_STATUS]
    model_config = ConfigDict(extra='forbid')

class AIHubStateModel(BaseModel):
    """
    The master Pydantic model for managing the entire state of the AI Hub.
    This model is intended to be stored in dcc.Store and used in callbacks,
    enforcing a strict, validated, and dict-free architecture.
    """
    # Top-level state
    hub_status: ComponentStatus = ComponentStatus.LOADING
    last_updated: Optional[datetime] = None
    target_symbol: Optional[str] = None
    error_message: Optional[str] = None

    # Configuration
    layout_config: AIHubLayoutConfig = Field(default_factory=AIHubLayoutConfig)

    # Panel Data Models - These hold the data for each major component
    market_compass: Optional[MarketCompassModel] = None
    ai_recommendations: Optional[AIRecommendationsPanelModel] = None
    
    # Metrics Panels
    flow_intelligence_metrics: Optional[MetricsPanelModel] = None
    volatility_gamma_metrics: Optional[MetricsPanelModel] = None
    advanced_options_metrics: Optional[MetricsPanelModel] = None # For LWPAI, VABAI, etc.

    # System Monitoring Panels
    huihui_monitor: Optional[HuiHuiExpertsMonitorModel] = None
    
    # Raw analysis bundle from the orchestrator for debugging or deep dives
    raw_analysis_bundle: Optional[MOEUnifiedResponseV2_5] = None

    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True  # Allow complex Pydantic models within this model
    )
