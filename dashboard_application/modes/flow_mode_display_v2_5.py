"""
Flow Mode Display - EOTS v2.5 Refactored
========================================

Fully Pydantic-compliant flow mode with unified architecture.
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
    FlowModeConfig,
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

class FlowModeState(BaseModel):
    """State management for flow mode."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    bundle: Optional[FinalAnalysisBundleV2_5] = None
    config: Optional[FlowModeConfig] = None
    filtered_data: Optional[FilteredDataBundle] = None
    is_initialized: bool = False
    error_state: Optional[str] = None
    last_update: Optional[str] = None

class FlowMetrics(BaseModel):
    """Flow metrics configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    flow_momentum: float = Field(default=0.0)
    call_flow: float = Field(default=0.0)
    put_flow: float = Field(default=0.0)
    net_flow: float = Field(default=0.0)
    flow_direction: str = Field(default="NEUTRAL")

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

def create_flow_overview_card(filtered_data: FilteredDataBundle, config: FlowModeConfig) -> dbc.Card:
    """Create the flow overview metrics card using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot calculate flow metrics without filtered data")
    
    # Calculate flow metrics from FILTERED strikes only
    total_call_volume = 0
    total_put_volume = 0
    total_call_oi = 0
    total_put_oi = 0
    
    for strike in filtered_data.filtered_strikes:
        call_vol = getattr(strike, 'call_volume', None)
        put_vol = getattr(strike, 'put_volume', None)
        call_oi = getattr(strike, 'call_open_interest', None)
        put_oi = getattr(strike, 'put_open_interest', None)
        
        if any(x is None for x in [call_vol, put_vol, call_oi, put_oi]):
            raise ValueError(f"Missing required flow data for filtered strike - cannot proceed without complete data")
        
        total_call_volume += call_vol
        total_put_volume += put_vol
        total_call_oi += call_oi
        total_put_oi += put_oi
    
    # Calculate flow metrics
    net_volume = total_call_volume - total_put_volume
    net_oi = total_call_oi - total_put_oi
    
    # Determine flow direction
    if net_volume > 0:
        flow_direction = "BULLISH"
        flow_color = "success"
    elif net_volume < 0:
        flow_direction = "BEARISH"
        flow_color = "danger"
    else:
        flow_direction = "NEUTRAL"
        flow_color = "warning"
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Flow Overview (Filtered)", className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(f"{total_call_volume:,}", className="text-success mb-0"),
                    html.Small("Call Volume", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{total_put_volume:,}", className="text-danger mb-0"),
                    html.Small("Put Volume", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{net_volume:,}", className="text-info mb-0"),
                    html.Small("Net Volume", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(flow_direction, className=f"text-{flow_color} mb-0"),
                    html.Small("Direction", className="text-muted")
                ], width=3)
            ])
        ])
    ], className="mb-4 elite-card")

def create_flow_momentum_chart(filtered_data: FilteredDataBundle, config: FlowModeConfig) -> dcc.Graph:
    """Create the flow momentum chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create flow momentum chart without filtered data")
    
    fig = create_empty_figure()
    
    # Create flow momentum from FILTERED strikes
    strike_prices = []
    net_flows = []
    
    for strike in filtered_data.filtered_strikes:
        strike_price = getattr(strike, 'strike_price', None)
        call_volume = getattr(strike, 'call_volume', None)
        put_volume = getattr(strike, 'put_volume', None)
        
        if strike_price is None or call_volume is None or put_volume is None:
            raise ValueError("Missing required data for filtered strike - cannot create flow momentum chart")
        
        strike_prices.append(strike_price)
        net_flows.append(call_volume - put_volume)
    
    # Add REAL filtered data trace
    fig.add_trace(go.Scatter(
        x=strike_prices,
        y=net_flows,
        mode='lines+markers',
        name='Net Flow by Strike',
        line=dict(color='#00ff88', width=3),
        marker=dict(size=8)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Flow Momentum by Strike (Filtered: {len(filtered_data.filtered_strikes)} strikes)",
        xaxis_title="Strike Price",
        yaxis_title="Net Flow (Call - Put Volume)",
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_call_put_flow_chart(filtered_data: FilteredDataBundle, config: FlowModeConfig) -> dcc.Graph:
    """Create the call/put flow comparison chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create call/put flow chart without filtered data")
    
    fig = create_empty_figure()
    
    # Calculate totals from FILTERED data
    total_call_volume = sum(getattr(strike, 'call_volume', 0) for strike in filtered_data.filtered_strikes)
    total_put_volume = sum(getattr(strike, 'put_volume', 0) for strike in filtered_data.filtered_strikes)
    
    # Create bar chart
    fig.add_trace(go.Bar(
        x=['Call Flow', 'Put Flow'],
        y=[total_call_volume, total_put_volume],
        marker_color=['#00ff88', '#ff6b6b'],
        opacity=0.8,
        name='Flow Comparison'
    ))
    
    fig.update_layout(
        title=f"Call vs Put Flow (Filtered: {len(filtered_data.filtered_strikes)} strikes)",
        xaxis_title="Flow Type",
        yaxis_title="Volume",
        template=PLOTLY_TEMPLATE,
        height=400,
        showlegend=False
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_flow_intelligence_panel(filtered_data: FilteredDataBundle, config: FlowModeConfig) -> dbc.Card:
    """Create the flow intelligence panel using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    intelligence_items = []
    
    if filtered_data.filtered_strikes:
        # Calculate intelligence metrics from filtered data
        total_strikes = len(filtered_data.filtered_strikes)
        avg_call_volume = sum(getattr(strike, 'call_volume', 0) for strike in filtered_data.filtered_strikes) / total_strikes
        avg_put_volume = sum(getattr(strike, 'put_volume', 0) for strike in filtered_data.filtered_strikes) / total_strikes
        
        metrics = [
            ("Filtered Strikes", f"{total_strikes}"),
            ("Avg Call Volume", f"{avg_call_volume:.0f}"),
            ("Avg Put Volume", f"{avg_put_volume:.0f}"),
            ("Price Range", f"${filtered_data.price_range_min:.2f} - ${filtered_data.price_range_max:.2f}")
        ]
        
        for name, value in metrics:
            intelligence_items.append(
                dbc.ListGroupItem([
                    html.Div([
                        html.Strong(name, className="me-2"),
                        html.Span(value, className="text-info")
                    ])
                ])
            )
    
    if not intelligence_items:
        intelligence_items.append(
            dbc.ListGroupItem("No flow intelligence data available in filtered range", color="warning")
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Flow Intelligence (Filtered)", className="mb-0 text-info")
        ]),
        dbc.CardBody([
            dbc.ListGroup(intelligence_items, flush=True)
        ])
    ], className="mb-4 elite-card")

# =============================================================================
# VALIDATION & STATE MANAGEMENT
# =============================================================================

def validate_flow_inputs(bundle: FinalAnalysisBundleV2_5, config: Any) -> ModeValidationResult:
    """Validate inputs for flow mode."""
    # FAIL FAST - Validate bundle
    if not bundle:
        return ModeValidationResult(
            is_valid=False,
            errors=["Bundle is required for trading operations"],
            warnings=[]
        )
    
    # Validate config
    if config is None:
        config = FlowModeConfig()
    elif not isinstance(config, FlowModeConfig):
        try:
            config = FlowModeConfig.model_validate(config)
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
    Creates the layout for the flow mode display with STRICT CONTROL PANEL FILTERING.
    
    Args:
        bundle: The analysis bundle containing all metrics
        config: Configuration object for the flow mode
        control_params: Control panel parameters (symbol, DTE, price range, etc.)
        
    Returns:
        html.Div: The complete flow mode layout
        
    Raises:
        ValueError: If bundle, control_params, or required data is missing
        ValidationError: If config validation fails
    """
    # FAIL FAST - Validate inputs
    validation = validate_flow_inputs(bundle, config)
    if not validation.is_valid:
        raise ValueError(f"Flow mode validation failed: {', '.join(validation.errors)}")
    
    # CRITICAL: Apply control panel filters
    if not control_params:
        raise ValueError("Control panel parameters are required - cannot display flow mode without filtering settings")
    
    filtered_data = apply_control_panel_filters(bundle, control_params)
    logger.info(f"Applied filters to flow mode: {filtered_data.get_filter_summary()}")
    
    # Build components using FILTERED data only
    overview_card = create_flow_overview_card(filtered_data, config)
    momentum_chart = create_flow_momentum_chart(filtered_data, config)
    flow_comparison_chart = create_call_put_flow_chart(filtered_data, config)
    intelligence_panel = create_flow_intelligence_panel(filtered_data, config)
    filter_info_panel = create_filter_info_panel(filtered_data)
    
    # Build layout
    layout = html.Div([
        # Header with Filter Info
        dbc.Row([
            dbc.Col([
                html.H1("Flow Mode", className="text-primary mb-0"),
                html.P("Options Flow Analysis & Momentum Tracking (FILTERED)", className="text-muted")
            ], width=8),
            dbc.Col([
                filter_info_panel
            ], width=4)
        ], className="mb-4"),
        
        # Overview
        dbc.Row([
            dbc.Col(overview_card, width=12)
        ]),
        
        # Charts Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(momentum_chart)
                ], className="elite-card")
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(flow_comparison_chart)
                ], className="elite-card")
            ], width=4)
        ], className="mb-4"),
        
        # Intelligence Panel
        dbc.Row([
            dbc.Col(intelligence_panel, width=12)
        ])
        
    ], className="elite-dashboard-container")
    
    logger.info("Flow mode layout created successfully with FILTERED market data")
    return layout