"""
Volatility Mode Display - EOTS v2.5 Refactored
==============================================

Fully Pydantic-compliant volatility mode with unified architecture.
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
    VolatilityModeConfig,
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

class VolatilityModeState(BaseModel):
    """State management for volatility mode."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    bundle: Optional[FinalAnalysisBundleV2_5] = None
    config: Optional[VolatilityModeConfig] = None
    filtered_data: Optional[FilteredDataBundle] = None
    is_initialized: bool = False
    error_state: Optional[str] = None
    last_update: Optional[str] = None

class VolatilityMetrics(BaseModel):
    """Volatility metrics configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    implied_volatility: float = Field(default=0.0)
    historical_volatility: float = Field(default=0.0)
    volatility_surface: float = Field(default=0.0)
    volatility_skew: float = Field(default=0.0)
    volatility_regime: str = Field(default="NORMAL")

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

def create_volatility_overview_card(filtered_data: FilteredDataBundle, config: VolatilityModeConfig) -> dbc.Card:
    """Create the volatility overview metrics card using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot calculate volatility metrics without filtered data")
    
    # Calculate volatility metrics from FILTERED strikes only
    total_implied_vol = 0
    vol_count = 0
    min_vol = float('inf')
    max_vol = 0
    
    for strike in filtered_data.filtered_strikes:
        call_iv = getattr(strike, 'call_implied_volatility', None)
        put_iv = getattr(strike, 'put_implied_volatility', None)
        
        if call_iv is not None:
            total_implied_vol += call_iv
            vol_count += 1
            min_vol = min(min_vol, call_iv)
            max_vol = max(max_vol, call_iv)
        
        if put_iv is not None:
            total_implied_vol += put_iv
            vol_count += 1
            min_vol = min(min_vol, put_iv)
            max_vol = max(max_vol, put_iv)
    
    if vol_count == 0:
        raise ValueError("No valid volatility data found in filtered strikes")
    
    avg_implied_vol = total_implied_vol / vol_count
    vol_spread = max_vol - min_vol
    
    # Determine volatility regime
    if avg_implied_vol > 0.3:
        vol_regime = "HIGH"
        regime_color = "danger"
    elif avg_implied_vol < 0.15:
        vol_regime = "LOW"
        regime_color = "success"
    else:
        vol_regime = "NORMAL"
        regime_color = "warning"
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Volatility Overview (Filtered)", className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(f"{avg_implied_vol:.1%}", className="text-info mb-0"),
                    html.Small("Avg Implied Vol", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{vol_spread:.1%}", className="text-warning mb-0"),
                    html.Small("Vol Spread", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{min_vol:.1%}", className="text-success mb-0"),
                    html.Small("Min Vol", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(vol_regime, className=f"text-{regime_color} mb-0"),
                    html.Small("Vol Regime", className="text-muted")
                ], width=3)
            ])
        ])
    ], className="mb-4 elite-card")

def create_volatility_surface_chart(filtered_data: FilteredDataBundle, config: VolatilityModeConfig) -> dcc.Graph:
    """Create the volatility surface chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create volatility surface without filtered data")
    
    fig = create_empty_figure()
    
    # Create volatility surface from FILTERED strikes
    strike_prices = []
    call_ivs = []
    put_ivs = []
    
    for strike in filtered_data.filtered_strikes:
        strike_price = getattr(strike, 'strike_price', None)
        call_iv = getattr(strike, 'call_implied_volatility', None)
        put_iv = getattr(strike, 'put_implied_volatility', None)
        
        if strike_price is None:
            raise ValueError("Missing strike price for filtered strike - cannot create volatility surface")
        
        strike_prices.append(strike_price)
        call_ivs.append(call_iv * 100 if call_iv is not None else None)  # Convert to percentage
        put_ivs.append(put_iv * 100 if put_iv is not None else None)  # Convert to percentage
    
    # Add traces for calls and puts
    if any(iv is not None for iv in call_ivs):
        fig.add_trace(go.Scatter(
            x=strike_prices,
            y=call_ivs,
            mode='lines+markers',
            name='Call IV',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=8)
        ))
    
    if any(iv is not None for iv in put_ivs):
        fig.add_trace(go.Scatter(
            x=strike_prices,
            y=put_ivs,
            mode='lines+markers',
            name='Put IV',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title=f"Volatility Surface (Filtered: {len(filtered_data.filtered_strikes)} strikes)",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility (%)",
        template=PLOTLY_TEMPLATE,
        height=500
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_volatility_skew_chart(filtered_data: FilteredDataBundle, config: VolatilityModeConfig) -> dcc.Graph:
    """Create the volatility skew chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create volatility skew without filtered data")
    
    fig = create_empty_figure()
    
    # Calculate skew from FILTERED data
    strike_prices = []
    skew_values = []
    
    for strike in filtered_data.filtered_strikes:
        strike_price = getattr(strike, 'strike_price', None)
        call_iv = getattr(strike, 'call_implied_volatility', None)
        put_iv = getattr(strike, 'put_implied_volatility', None)
        
        if strike_price is None:
            raise ValueError("Missing strike price for filtered strike - cannot create skew chart")
        
        # Calculate skew as difference between put and call IV
        if call_iv is not None and put_iv is not None:
            skew = (put_iv - call_iv) * 100  # Convert to percentage points
            strike_prices.append(strike_price)
            skew_values.append(skew)
    
    if not skew_values:
        raise ValueError("No valid skew data found in filtered strikes")
    
    # Add skew trace
    fig.add_trace(go.Scatter(
        x=strike_prices,
        y=skew_values,
        mode='lines+markers',
        name='Volatility Skew',
        line=dict(color='#ffd700', width=3),
        marker=dict(size=10)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"Volatility Skew Profile (Filtered: {len(skew_values)} strikes)",
        xaxis_title="Strike Price",
        yaxis_title="Skew (Put IV - Call IV) %",
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_volatility_intelligence_panel(filtered_data: FilteredDataBundle, config: VolatilityModeConfig) -> dbc.Card:
    """Create the volatility intelligence panel using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    intelligence_items = []
    
    if filtered_data.filtered_strikes:
        # Calculate intelligence metrics from filtered data
        total_strikes = len(filtered_data.filtered_strikes)
        
        # Calculate volatility statistics
        all_ivs = []
        for strike in filtered_data.filtered_strikes:
            call_iv = getattr(strike, 'call_implied_volatility', None)
            put_iv = getattr(strike, 'put_implied_volatility', None)
            if call_iv is not None:
                all_ivs.append(call_iv)
            if put_iv is not None:
                all_ivs.append(put_iv)
        
        if all_ivs:
            avg_iv = sum(all_ivs) / len(all_ivs)
            min_iv = min(all_ivs)
            max_iv = max(all_ivs)
            
            metrics = [
                ("Filtered Strikes", f"{total_strikes}"),
                ("Avg IV", f"{avg_iv:.1%}"),
                ("Min IV", f"{min_iv:.1%}"),
                ("Max IV", f"{max_iv:.1%}"),
                ("IV Range", f"{(max_iv - min_iv):.1%}")
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
            dbc.ListGroupItem("No volatility intelligence data available in filtered range", color="warning")
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Volatility Intelligence (Filtered)", className="mb-0 text-info")
        ]),
        dbc.CardBody([
            dbc.ListGroup(intelligence_items, flush=True)
        ])
    ], className="mb-4 elite-card")

# =============================================================================
# VALIDATION & STATE MANAGEMENT
# =============================================================================

def validate_volatility_inputs(bundle: FinalAnalysisBundleV2_5, config: Any) -> ModeValidationResult:
    """Validate inputs for volatility mode."""
    # FAIL FAST - Validate bundle
    if not bundle:
        return ModeValidationResult(
            is_valid=False,
            errors=["Bundle is required for trading operations"],
            warnings=[]
        )
    
    # Validate config
    if config is None:
        config = VolatilityModeConfig()
    elif not isinstance(config, VolatilityModeConfig):
        try:
            config = VolatilityModeConfig.model_validate(config)
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
    Creates the layout for the volatility mode display with STRICT CONTROL PANEL FILTERING.
    
    Args:
        bundle: The analysis bundle containing all metrics
        config: Configuration object for the volatility mode
        control_params: Control panel parameters (symbol, DTE, price range, etc.)
        
    Returns:
        html.Div: The complete volatility mode layout
        
    Raises:
        ValueError: If bundle, control_params, or required data is missing
        ValidationError: If config validation fails
    """
    # FAIL FAST - Validate inputs
    validation = validate_volatility_inputs(bundle, config)
    if not validation.is_valid:
        raise ValueError(f"Volatility mode validation failed: {', '.join(validation.errors)}")
    
    # CRITICAL: Apply control panel filters
    if not control_params:
        raise ValueError("Control panel parameters are required - cannot display volatility mode without filtering settings")
    
    filtered_data = apply_control_panel_filters(bundle, control_params)
    logger.info(f"Applied filters to volatility mode: {filtered_data.get_filter_summary()}")
    
    # Build components using FILTERED data only
    overview_card = create_volatility_overview_card(filtered_data, config)
    surface_chart = create_volatility_surface_chart(filtered_data, config)
    skew_chart = create_volatility_skew_chart(filtered_data, config)
    intelligence_panel = create_volatility_intelligence_panel(filtered_data, config)
    filter_info_panel = create_filter_info_panel(filtered_data)
    
    # Build layout
    layout = html.Div([
        # Header with Filter Info
        dbc.Row([
            dbc.Col([
                html.H1("Volatility Mode", className="text-primary mb-0"),
                html.P("Volatility Surface Analysis & Regime Detection (FILTERED)", className="text-muted")
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
                    dbc.CardBody(surface_chart)
                ], className="elite-card")
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(skew_chart)
                ], className="elite-card")
            ], width=4)
        ], className="mb-4"),
        
        # Intelligence Panel
        dbc.Row([
            dbc.Col(intelligence_panel, width=12)
        ])
        
    ], className="elite-dashboard-container")
    
    logger.info("Volatility mode layout created successfully with FILTERED market data")
    return layout