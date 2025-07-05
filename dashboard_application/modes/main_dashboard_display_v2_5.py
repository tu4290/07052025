"""
Main Dashboard Display - EOTS v2.5 Refactored
=============================================

Fully Pydantic-compliant main dashboard with unified architecture.
Follows the established refactoring pattern for consistent mode behavior.

Key Features:
- Strict Pydantic v2 validation
- Unified component architecture
- Elite visual design system
- Comprehensive error handling
- Performance optimized
- CONTROL PANEL FILTERING ENFORCED
"""

import logging
from typing import Any, List, Optional, Union
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.development.base_component import Component
from pydantic import BaseModel, Field, ConfigDict

# Core imports
from data_models.core_models import FinalAnalysisBundleV2_5
from data_models.dashboard_mode_models import (
    MainDashboardConfig,
    FilteredDataBundle,
    ModeValidationResult
)
from dashboard_application.utils_dashboard_v2_5 import (
    create_empty_figure,
    add_price_line,
    PLOTLY_TEMPLATE,
    add_bottom_right_timestamp_annotation,
    apply_dark_theme_template
)

logger = logging.getLogger(__name__)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class MainDashboardState(BaseModel):
    """State management for main dashboard mode."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    bundle: Optional[FinalAnalysisBundleV2_5] = None
    config: Optional[MainDashboardConfig] = None
    filtered_data: Optional[FilteredDataBundle] = None
    is_initialized: bool = False
    error_state: Optional[str] = None
    last_update: Optional[str] = None

class EliteMetricsCard(BaseModel):
    """Elite metrics card configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    title: str = Field(..., min_length=1)
    value: Union[str, float, int] = Field(...)
    subtitle: Optional[str] = None
    color_scheme: str = Field(default="primary")
    icon: Optional[str] = None
    trend_indicator: Optional[str] = None

# =============================================================================
# CONTROL PANEL FILTERING FUNCTIONS
# =============================================================================

def apply_control_panel_filters(bundle: FinalAnalysisBundleV2_5, control_params: Any) -> FilteredDataBundle:
    """Apply control panel filters to the data bundle."""
    # CRITICAL: Validate control panel parameters
    if not control_params:
        raise ValueError("Control panel parameters are required - cannot filter data without settings")
    
    # Extract control panel settings
    target_symbol = getattr(control_params, 'target_symbol', None)
    price_range_percent = getattr(control_params, 'price_range_percent', None)
    dte_min = getattr(control_params, 'dte_min', None)
    dte_max = getattr(control_params, 'dte_max', None)
    
    if not target_symbol:
        raise ValueError("Target symbol is required from control panel")
    if price_range_percent is None:
        raise ValueError("Price range percentage is required from control panel")
    if dte_min is None or dte_max is None:
        raise ValueError("DTE range is required from control panel")
    
    # Get current price for the symbol
    und_data = getattr(bundle, 'underlying_aggregates', None)
    if not und_data:
        raise ValueError("Underlying data is required for filtering")
    
    current_price = getattr(und_data, 'current_price', None)
    if current_price is None:
        raise ValueError("Current price is required for price range filtering")
    
    # Calculate price range
    price_range_factor = price_range_percent / 100.0
    price_range_min = current_price * (1 - price_range_factor)
    price_range_max = current_price * (1 + price_range_factor)
    
    # Filter strikes by price range
    strike_data = getattr(bundle, 'strike_level_metrics', None)
    filtered_strikes = []
    total_strikes = 0
    
    if strike_data and hasattr(strike_data, 'strikes'):
        total_strikes = len(strike_data.strikes)
        for strike in strike_data.strikes:
            strike_price = getattr(strike, 'strike_price', None)
            if strike_price and price_range_min <= strike_price <= price_range_max:
                filtered_strikes.append(strike)
    
    # Filter contracts by DTE range (placeholder - would need contract data)
    filtered_contracts = []
    total_contracts = 0
    
    return FilteredDataBundle(
        filtered_strikes=filtered_strikes,
        filtered_contracts=filtered_contracts,
        price_range_min=price_range_min,
        price_range_max=price_range_max,
        current_price=current_price,
        price_range_percent=price_range_percent,
        dte_min=dte_min,
        dte_max=dte_max,
        total_strikes_available=total_strikes,
        total_contracts_available=total_contracts,
        strikes_filtered_count=len(filtered_strikes),
        contracts_filtered_count=len(filtered_contracts)
    )

def create_filter_info_panel(filtered_data: FilteredDataBundle) -> dbc.Card:
    """Create a panel showing applied control panel filters."""
    return dbc.Card([
        dbc.CardHeader([
            html.H6("Active Filters", className="mb-0 text-info")
        ]),
        dbc.CardBody([
            html.P(filtered_data.get_filter_summary(), className="mb-0 small text-muted")
        ])
    ], className="mb-3 filter-info-card")

# =============================================================================
# CORE COMPONENT BUILDERS
# =============================================================================

def create_elite_overview_card(filtered_data: FilteredDataBundle, config: MainDashboardConfig) -> dbc.Card:
    """Create the elite overview metrics card using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without filtered data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within price range - cannot display overview without filtered data")
    
    # Calculate metrics from FILTERED strikes only
    total_oi = 0
    total_volume = 0
    avg_gamma = 0
    
    for strike in filtered_data.filtered_strikes:
        oi = getattr(strike, 'total_open_interest', None)
        volume = getattr(strike, 'total_volume', None)
        gamma = getattr(strike, 'total_gamma', None)
        
        if oi is None or volume is None or gamma is None:
            raise ValueError(f"Missing required data for filtered strike - cannot proceed without complete data")
        
        total_oi += oi
        total_volume += volume
        avg_gamma += gamma
    
    avg_gamma = avg_gamma / len(filtered_data.filtered_strikes) if filtered_data.filtered_strikes else 0
    
    # Build metrics with REAL filtered data only
    metrics = [
        EliteMetricsCard(
            title="Filtered OI",
            value=f"{total_oi:,}",
            color_scheme="success",
            trend_indicator="FILTERED"
        ),
        EliteMetricsCard(
            title="Filtered Volume",
            value=f"{total_volume:,}",
            color_scheme="info",
            trend_indicator="RANGE"
        ),
        EliteMetricsCard(
            title="Avg Gamma",
            value=f"{avg_gamma:.4f}",
            color_scheme="warning",
            trend_indicator="CALC"
        )
    ]
    
    # Build card content
    card_content = []
    for metric in metrics:
        card_content.append(
            dbc.Col([
                html.H4(metric.value, className=f"text-{metric.color_scheme} mb-0"),
                html.Small(f"{metric.trend_indicator} {metric.title}", className="text-muted")
            ], width=4)
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Elite Market Overview (Filtered)", className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            dbc.Row(card_content)
        ])
    ], className="mb-4 elite-card")

def create_flow_analytics_chart(filtered_data: FilteredDataBundle, config: MainDashboardConfig) -> dcc.Graph:
    """Create the flow analytics chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without filtered data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create flow analytics without filtered data")
    
    fig = create_empty_figure()
    
    # Create flow analytics from FILTERED strikes
    strike_prices = []
    flow_values = []
    
    for strike in filtered_data.filtered_strikes:
        strike_price = getattr(strike, 'strike_price', None)
        call_volume = getattr(strike, 'call_volume', None)
        put_volume = getattr(strike, 'put_volume', None)
        
        if strike_price is None or call_volume is None or put_volume is None:
            raise ValueError("Missing required data for filtered strike - cannot create analytics")
        
        strike_prices.append(strike_price)
        flow_values.append(call_volume - put_volume)  # Net flow
    
    # Add REAL filtered data trace
    fig.add_trace(go.Scatter(
        x=strike_prices,
        y=flow_values,
        mode='lines+markers',
        name='Net Flow (Filtered)',
        line=dict(color='#00ff88', width=3)
    ))
    
    fig.update_layout(
        title=f"Elite Flow Analytics (Filtered: {len(filtered_data.filtered_strikes)} strikes)",
        xaxis_title="Strike Price",
        yaxis_title="Net Flow",
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

# =============================================================================
# VALIDATION & STATE MANAGEMENT
# =============================================================================

def validate_dashboard_inputs(bundle: FinalAnalysisBundleV2_5, config: Any) -> ModeValidationResult:
    """Validate inputs for main dashboard mode."""
    # FAIL FAST - Validate bundle
    if not bundle:
        return ModeValidationResult(
            is_valid=False,
            errors=["Bundle is required for trading operations"],
            warnings=[]
        )
    
    # Validate config
    if config is None:
        config = MainDashboardConfig()
    elif not isinstance(config, MainDashboardConfig):
        try:
            config = MainDashboardConfig.model_validate(config)
        except Exception as e:
            return ModeValidationResult(
                is_valid=False,
                errors=[f"Invalid config: {str(e)}"],
                warnings=[]
            )
    
    return ModeValidationResult(
        is_valid=True,
        errors=[],
        warnings=[]
    )

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_layout(bundle: FinalAnalysisBundleV2_5, config: Any, control_params: Any = None) -> html.Div:
    """
    Creates the layout for the main dashboard display, incorporating elite metrics
    and visualizations with STRICT CONTROL PANEL FILTERING.
    
    Args:
        bundle: The analysis bundle containing all metrics
        config: Configuration object for the dashboard
        control_params: Control panel parameters (symbol, DTE, price range, etc.)
        
    Returns:
        html.Div: The complete dashboard layout
    
    Raises:
        ValueError: If bundle, control_params, or required data is missing
        ValidationError: If config validation fails
    """
    # FAIL FAST - Validate inputs immediately
    validation = validate_dashboard_inputs(bundle, config)
    if not validation.is_valid:
        raise ValueError(f"Dashboard validation failed: {', '.join(validation.errors)}")
    
    # CRITICAL: Apply control panel filters
    if not control_params:
        raise ValueError("Control panel parameters are required - cannot display dashboard without filtering settings")
    
    filtered_data = apply_control_panel_filters(bundle, control_params)
    logger.info(f"Applied filters: {filtered_data.get_filter_summary()}")
    
    # Build components using FILTERED data only
    elite_overview = create_elite_overview_card(filtered_data, config)
    flow_chart = create_flow_analytics_chart(filtered_data, config)
    filter_info_panel = create_filter_info_panel(filtered_data)
    
    # Build layout
    layout = html.Div([
        # Header with Filter Info
        dbc.Row([
            dbc.Col([
                html.H1("Elite Options Trading System", className="text-primary mb-0"),
                html.P("Main Dashboard - Real-time Market Intelligence (FILTERED)", className="text-muted")
            ], width=8),
            dbc.Col([
                filter_info_panel
            ], width=4)
        ], className="mb-4"),
        
        # Elite Overview
        dbc.Row([
            dbc.Col(elite_overview, width=12)
        ]),
        
        # Charts Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(flow_chart)
                ], className="elite-card")
            ], width=12)
        ], className="mb-4")
        
    ], className="elite-dashboard-container")
    
    logger.info("Main dashboard layout created successfully with FILTERED market data")
    return layout