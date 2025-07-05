"""
Time Decay Mode Display - EOTS v2.5 Refactored
==============================================

Fully Pydantic-compliant time decay mode with unified architecture.
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
    TimeDecayModeConfig,
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

class TimeDecayModeState(BaseModel):
    """State management for time decay mode."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    bundle: Optional[FinalAnalysisBundleV2_5] = None
    config: Optional[TimeDecayModeConfig] = None
    filtered_data: Optional[FilteredDataBundle] = None
    is_initialized: bool = False
    error_state: Optional[str] = None
    last_update: Optional[str] = None

class TimeDecayMetrics(BaseModel):
    """Time decay metrics configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    theta_exposure: float = Field(default=0.0)
    gamma_exposure: float = Field(default=0.0)
    time_to_expiry: int = Field(default=0)
    decay_rate: float = Field(default=0.0)
    pinning_effect: float = Field(default=0.0)

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
    
    # Filter strikes by price range AND DTE range
    strike_data = getattr(bundle, 'strike_level_metrics', None)
    filtered_strikes = []
    total_strikes = 0
    
    if strike_data and hasattr(strike_data, 'strikes'):
        total_strikes = len(strike_data.strikes)
        for strike in strike_data.strikes:
            strike_price = getattr(strike, 'strike_price', None)
            # For time decay, we also need to check DTE if available
            strike_dte = getattr(strike, 'days_to_expiry', None)
            
            price_in_range = strike_price and price_range_min <= strike_price <= price_range_max
            dte_in_range = strike_dte is None or (dte_min <= strike_dte <= dte_max)
            
            if price_in_range and dte_in_range:
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

def create_time_decay_overview_card(filtered_data: FilteredDataBundle, config: TimeDecayModeConfig) -> dbc.Card:
    """Create the time decay overview metrics card using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot calculate time decay metrics without filtered data")
    
    # Calculate time decay metrics from FILTERED strikes only
    total_theta = 0
    total_gamma = 0
    total_contracts = 0
    avg_dte = 0
    dte_count = 0
    
    for strike in filtered_data.filtered_strikes:
        theta = getattr(strike, 'total_theta', None)
        gamma = getattr(strike, 'total_gamma', None)
        dte = getattr(strike, 'days_to_expiry', None)
        
        if theta is not None:
            total_theta += theta
        if gamma is not None:
            total_gamma += gamma
        if dte is not None:
            avg_dte += dte
            dte_count += 1
        
        total_contracts += 1
    
    avg_dte = avg_dte / dte_count if dte_count > 0 else 0
    
    # Determine decay regime
    if abs(total_theta) > 1000:
        decay_regime = "HIGH"
        regime_color = "danger"
    elif abs(total_theta) < 100:
        decay_regime = "LOW"
        regime_color = "success"
    else:
        decay_regime = "NORMAL"
        regime_color = "warning"
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Time Decay Overview (Filtered)", className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(f"{total_theta:.2f}", className="text-danger mb-0"),
                    html.Small("Total Theta", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{total_gamma:.2f}", className="text-success mb-0"),
                    html.Small("Total Gamma", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{avg_dte:.0f}", className="text-info mb-0"),
                    html.Small("Avg DTE", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(decay_regime, className=f"text-{regime_color} mb-0"),
                    html.Small("Decay Regime", className="text-muted")
                ], width=3)
            ])
        ])
    ], className="mb-4 elite-card")

def create_theta_exposure_chart(filtered_data: FilteredDataBundle, config: TimeDecayModeConfig) -> dcc.Graph:
    """Create the theta exposure chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create theta chart without filtered data")
    
    fig = create_empty_figure()
    
    # Create theta exposure from FILTERED strikes
    strike_prices = []
    theta_values = []
    
    for strike in filtered_data.filtered_strikes:
        strike_price = getattr(strike, 'strike_price', None)
        theta = getattr(strike, 'total_theta', None)
        
        if strike_price is None:
            raise ValueError("Missing strike price for filtered strike - cannot create theta chart")
        
        if theta is not None:
            strike_prices.append(strike_price)
            theta_values.append(theta)
    
    if not theta_values:
        raise ValueError("No valid theta data found in filtered strikes")
    
    # Add theta exposure trace
    fig.add_trace(go.Bar(
        x=strike_prices,
        y=theta_values,
        name='Theta Exposure',
        marker_color='#ff6b6b',
        opacity=0.8
    ))
    
    fig.update_layout(
        title=f"Theta Exposure by Strike (Filtered: {len(theta_values)} strikes)",
        xaxis_title="Strike Price",
        yaxis_title="Theta Exposure",
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_time_decay_profile_chart(filtered_data: FilteredDataBundle, config: TimeDecayModeConfig) -> dcc.Graph:
    """Create the time decay profile chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create decay profile without filtered data")
    
    fig = create_empty_figure()
    
    # Create decay profile from FILTERED data
    dte_values = []
    theta_values = []
    
    for strike in filtered_data.filtered_strikes:
        dte = getattr(strike, 'days_to_expiry', None)
        theta = getattr(strike, 'total_theta', None)
        
        if dte is not None and theta is not None:
            dte_values.append(dte)
            theta_values.append(abs(theta))  # Use absolute value for decay rate
    
    if not dte_values:
        raise ValueError("No valid DTE/theta data found in filtered strikes")
    
    # Sort by DTE for proper line chart
    sorted_data = sorted(zip(dte_values, theta_values))
    dte_sorted, theta_sorted = zip(*sorted_data)
    
    fig.add_trace(go.Scatter(
        x=dte_sorted,
        y=theta_sorted,
        mode='lines+markers',
        name='Time Decay Profile',
        line=dict(color='#ffd700', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"Time Decay Profile (Filtered: {len(dte_values)} data points)",
        xaxis_title="Days to Expiration",
        yaxis_title="Theta (Absolute)",
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_pinning_analysis_panel(filtered_data: FilteredDataBundle, config: TimeDecayModeConfig) -> dbc.Card:
    """Create the pinning analysis panel using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    pinning_items = []
    
    if filtered_data.filtered_strikes:
        # Calculate pinning metrics from filtered data
        total_strikes = len(filtered_data.filtered_strikes)
        current_price = filtered_data.current_price
        
        # Find strikes closest to current price (potential pinning levels)
        closest_strikes = []
        for strike in filtered_data.filtered_strikes:
            strike_price = getattr(strike, 'strike_price', None)
            if strike_price is not None:
                distance = abs(strike_price - current_price)
                closest_strikes.append((distance, strike_price, strike))
        
        # Sort by distance and take top 3
        closest_strikes.sort()
        top_strikes = closest_strikes[:3]
        
        # Calculate pinning metrics
        total_gamma_near_money = sum(getattr(strike[2], 'total_gamma', 0) for strike in top_strikes)
        total_oi_near_money = sum(getattr(strike[2], 'total_open_interest', 0) for strike in top_strikes)
        
        metrics = [
            ("Filtered Strikes", f"{total_strikes}"),
            ("Current Price", f"${current_price:.2f}"),
            ("Nearest Strike", f"${top_strikes[0][1]:.2f}" if top_strikes else "N/A"),
            ("Gamma Near Money", f"{total_gamma_near_money:.4f}"),
            ("OI Near Money", f"{total_oi_near_money:,}")
        ]
        
        for name, value in metrics:
            color = "info"
            if "Gamma" in name and total_gamma_near_money > 0.1:
                color = "warning"
            elif "OI" in name and total_oi_near_money > 10000:
                color = "danger"
            
            pinning_items.append(
                dbc.ListGroupItem([
                    html.Div([
                        html.Strong(name, className="me-2"),
                        html.Span(value, className=f"text-{color}")
                    ])
                ])
            )
    
    if not pinning_items:
        pinning_items.append(
            dbc.ListGroupItem("No pinning analysis data available in filtered range", color="warning")
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Pinning Analysis (Filtered)", className="mb-0 text-info")
        ]),
        dbc.CardBody([
            dbc.ListGroup(pinning_items, flush=True)
        ])
    ], className="mb-4 elite-card")

# =============================================================================
# VALIDATION & STATE MANAGEMENT
# =============================================================================

def validate_time_decay_inputs(bundle: FinalAnalysisBundleV2_5, config: Any) -> ModeValidationResult:
    """Validate inputs for time decay mode."""
    # FAIL FAST - Validate bundle
    if not bundle:
        return ModeValidationResult(
            is_valid=False,
            errors=["Bundle is required for trading operations"],
            warnings=[]
        )
    
    # Validate config
    if config is None:
        config = TimeDecayModeConfig()
    elif not isinstance(config, TimeDecayModeConfig):
        try:
            config = TimeDecayModeConfig.model_validate(config)
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
    Creates the layout for the time decay mode display with STRICT CONTROL PANEL FILTERING.
    
    Args:
        bundle: The analysis bundle containing all metrics
        config: Configuration object for the time decay mode
        control_params: Control panel parameters (symbol, DTE, price range, etc.)
        
    Returns:
        html.Div: The complete time decay mode layout
        
    Raises:
        ValueError: If bundle, control_params, or required data is missing
        ValidationError: If config validation fails
    """
    # FAIL FAST - Validate inputs
    validation = validate_time_decay_inputs(bundle, config)
    if not validation.is_valid:
        raise ValueError(f"Time decay mode validation failed: {', '.join(validation.errors)}")
    
    # CRITICAL: Apply control panel filters
    if not control_params:
        raise ValueError("Control panel parameters are required - cannot display time decay mode without filtering settings")
    
    filtered_data = apply_control_panel_filters(bundle, control_params)
    logger.info(f"Applied filters to time decay mode: {filtered_data.get_filter_summary()}")
    
    # Build components using FILTERED data only
    overview_card = create_time_decay_overview_card(filtered_data, config)
    theta_chart = create_theta_exposure_chart(filtered_data, config)
    decay_profile_chart = create_time_decay_profile_chart(filtered_data, config)
    pinning_panel = create_pinning_analysis_panel(filtered_data, config)
    filter_info_panel = create_filter_info_panel(filtered_data)
    
    # Build layout
    layout = html.Div([
        # Header with Filter Info
        dbc.Row([
            dbc.Col([
                html.H1("Time Decay Mode", className="text-primary mb-0"),
                html.P("Time Decay Analysis & Pinning Effects (FILTERED)", className="text-muted")
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
                    dbc.CardBody(theta_chart)
                ], className="elite-card")
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(decay_profile_chart)
                ], className="elite-card")
            ], width=4)
        ], className="mb-4"),
        
        # Pinning Analysis Panel
        dbc.Row([
            dbc.Col(pinning_panel, width=12)
        ])
        
    ], className="elite-dashboard-container")
    
    logger.info("Time decay mode layout created successfully with FILTERED market data")
    return layout