"""
Structure Mode Display - EOTS v2.5 Refactored
=============================================

Fully Pydantic-compliant structure mode with unified architecture.
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
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.development.base_component import Component
from pydantic import BaseModel, Field, ConfigDict

# Core imports
from data_models.core_models import FinalAnalysisBundleV2_5
from data_models.dashboard_mode_models import (
    StructureModeConfig,
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

class StructureModeState(BaseModel):
    """State management for structure mode."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    bundle: Optional[FinalAnalysisBundleV2_5] = None
    config: Optional[StructureModeConfig] = None
    filtered_data: Optional[FilteredDataBundle] = None
    is_initialized: bool = False
    error_state: Optional[str] = None
    last_update: Optional[str] = None

class StructureMetrics(BaseModel):
    """Structure metrics configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    strike_price: float = Field(...)
    open_interest: int = Field(default=0)
    volume: int = Field(default=0)
    gamma: float = Field(default=0.0)
    delta: float = Field(default=0.0)
    option_type: str = Field(default="CALL")

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

def create_structure_overview_card(filtered_data: FilteredDataBundle, config: StructureModeConfig) -> dbc.Card:
    """Create the structure overview metrics card using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot calculate structure metrics without filtered data")
    
    strikes = filtered_data.filtered_strikes
    
    # Calculate REAL structure metrics from FILTERED data - no defaults
    total_oi = 0
    total_volume = 0
    max_gamma = 0
    max_gamma_strike = None
    
    for strike in strikes:
        oi = getattr(strike, 'total_open_interest', None)
        volume = getattr(strike, 'total_volume', None)
        gamma = getattr(strike, 'total_gamma', None)
        strike_price = getattr(strike, 'strike_price', None)
        
        if oi is None or volume is None or gamma is None or strike_price is None:
            raise ValueError(f"Missing required data for filtered strike {strike_price} - cannot proceed without complete data")
        
        total_oi += oi
        total_volume += volume
        
        if gamma > max_gamma:
            max_gamma = gamma
            max_gamma_strike = strike_price
    
    if max_gamma_strike is None:
        raise ValueError("No valid gamma data found in filtered strikes - cannot determine max gamma strike")
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Structure Overview (Filtered)", className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(f"{total_oi:,}", className="text-success mb-0"),
                    html.Small("Total Open Interest", className="text-muted")
                ], width=4),
                dbc.Col([
                    html.H4(f"{total_volume:,}", className="text-info mb-0"),
                    html.Small("Total Volume", className="text-muted")
                ], width=4),
                dbc.Col([
                    html.H4(f"${max_gamma_strike:.2f}", className="text-warning mb-0"),
                    html.Small("Max Gamma Strike", className="text-muted")
                ], width=4)
            ])
        ])
    ], className="mb-4 elite-card")

def create_strike_structure_chart(filtered_data: FilteredDataBundle, config: StructureModeConfig) -> dcc.Graph:
    """Create the strike structure chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create structure chart without filtered data")
    
    strikes = filtered_data.filtered_strikes
    
    # Prepare REAL filtered data only - no defaults
    strike_prices = []
    call_oi = []
    put_oi = []
    
    for strike in strikes:
        strike_price = getattr(strike, 'strike_price', None)
        call_open_interest = getattr(strike, 'call_open_interest', None)
        put_open_interest = getattr(strike, 'put_open_interest', None)
        
        if strike_price is None or call_open_interest is None or put_open_interest is None:
            raise ValueError(f"Missing required data for filtered strike {strike_price} - cannot create chart without complete data")
        
        strike_prices.append(strike_price)
        call_oi.append(call_open_interest)
        put_oi.append(-put_open_interest)  # Negative for puts
    
    if not strike_prices:
        raise ValueError("No valid strike data found in filtered strikes - cannot create structure chart")
    
    fig = create_empty_figure()
    
    # Add traces with REAL filtered data
    fig.add_trace(go.Bar(
        x=strike_prices,
        y=call_oi,
        name='Call OI',
        marker_color='#00ff88',
        opacity=0.8
    ))
    
    fig.add_trace(go.Bar(
        x=strike_prices,
        y=put_oi,
        name='Put OI',
        marker_color='#ff6b6b',
        opacity=0.8
    ))
    
    fig.update_layout(
        title=f"Strike Structure - Open Interest (Filtered: {len(strikes)} strikes)",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest",
        template=PLOTLY_TEMPLATE,
        height=500,
        barmode='relative'
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_gamma_exposure_chart(filtered_data: FilteredDataBundle, config: StructureModeConfig) -> dcc.Graph:
    """Create the gamma exposure chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create gamma chart without filtered data")
    
    fig = create_empty_figure()
    strikes = filtered_data.filtered_strikes
    
    # Prepare filtered data
    strike_prices = []
    gamma_exposure = []
    
    for strike in strikes:
        strike_price = getattr(strike, 'strike_price', None)
        total_gamma = getattr(strike, 'total_gamma', None)
        
        if strike_price is None or total_gamma is None:
            raise ValueError(f"Missing required data for filtered strike {strike_price} - cannot create gamma chart")
        
        strike_prices.append(strike_price)
        gamma_exposure.append(total_gamma)
    
    # Add trace with REAL filtered data
    fig.add_trace(go.Scatter(
        x=strike_prices,
        y=gamma_exposure,
        mode='lines+markers',
        name='Gamma Exposure',
        line=dict(color='#ffd700', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"Gamma Exposure Profile (Filtered: {len(strikes)} strikes)",
        xaxis_title="Strike Price",
        yaxis_title="Gamma Exposure",
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_structure_table(filtered_data: FilteredDataBundle, config: StructureModeConfig) -> dash_table.DataTable:
    """Create the structure data table using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create table without filtered data")
    
    # Get filtered strike data
    strikes = filtered_data.filtered_strikes
    table_data = []
    
    for strike in strikes:
        strike_price = getattr(strike, 'strike_price', None)
        call_oi = getattr(strike, 'call_open_interest', None)
        put_oi = getattr(strike, 'put_open_interest', None)
        call_vol = getattr(strike, 'call_volume', None)
        put_vol = getattr(strike, 'put_volume', None)
        gamma = getattr(strike, 'total_gamma', None)
        
        if any(x is None for x in [strike_price, call_oi, put_oi, call_vol, put_vol, gamma]):
            raise ValueError(f"Missing required data for filtered strike {strike_price} - cannot create table")
        
        table_data.append({
            'Strike': f"${strike_price:.2f}",
            'Call OI': f"{call_oi:,}",
            'Put OI': f"{put_oi:,}",
            'Call Vol': f"{call_vol:,}",
            'Put Vol': f"{put_vol:,}",
            'Gamma': f"{gamma:.4f}"
        })
    
    return dash_table.DataTable(
        data=table_data,
        columns=[
            {"name": "Strike", "id": "Strike"},
            {"name": "Call OI", "id": "Call OI"},
            {"name": "Put OI", "id": "Put OI"},
            {"name": "Call Vol", "id": "Call Vol"},
            {"name": "Put Vol", "id": "Put Vol"},
            {"name": "Gamma", "id": "Gamma"}
        ],
        style_cell={'textAlign': 'center', 'backgroundColor': '#1e1e1e', 'color': 'white'},
        style_header={'backgroundColor': '#2d2d2d', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#2a2a2a'
            }
        ],
        page_size=10,
        className="elite-table"
    )

# =============================================================================
# VALIDATION & STATE MANAGEMENT
# =============================================================================

def validate_structure_inputs(bundle: FinalAnalysisBundleV2_5, config: Any) -> ModeValidationResult:
    """Validate inputs for structure mode."""
    # FAIL FAST - Validate bundle
    if not bundle:
        return ModeValidationResult(
            is_valid=False,
            errors=["Bundle is required for trading operations"],
            warnings=[]
        )
    
    # Validate config
    if config is None:
        config = StructureModeConfig()
    elif not isinstance(config, StructureModeConfig):
        try:
            config = StructureModeConfig.model_validate(config)
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
    Creates the layout for the structure mode display with STRICT CONTROL PANEL FILTERING.
    
    Args:
        bundle: The analysis bundle containing all metrics
        config: Configuration object for the structure mode
        control_params: Control panel parameters (symbol, DTE, price range, etc.)
        
    Returns:
        html.Div: The complete structure mode layout
        
    Raises:
        ValueError: If bundle, control_params, or required data is missing
        ValidationError: If config validation fails
    """
    # FAIL FAST - Validate inputs
    validation = validate_structure_inputs(bundle, config)
    if not validation.is_valid:
        raise ValueError(f"Structure mode validation failed: {', '.join(validation.errors)}")
    
    # CRITICAL: Apply control panel filters
    if not control_params:
        raise ValueError("Control panel parameters are required - cannot display structure mode without filtering settings")
    
    filtered_data = apply_control_panel_filters(bundle, control_params)
    logger.info(f"Applied filters to structure mode: {filtered_data.get_filter_summary()}")
    
    # Build components using FILTERED data only
    overview_card = create_structure_overview_card(filtered_data, config)
    structure_chart = create_strike_structure_chart(filtered_data, config)
    gamma_chart = create_gamma_exposure_chart(filtered_data, config)
    structure_table = create_structure_table(filtered_data, config)
    filter_info_panel = create_filter_info_panel(filtered_data)
    
    # Build layout
    layout = html.Div([
        # Header with Filter Info
        dbc.Row([
            dbc.Col([
                html.H1("Structure Mode", className="text-primary mb-0"),
                html.P("Options Structure Analysis & Strike Distribution (FILTERED)", className="text-muted")
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
                    dbc.CardBody(structure_chart)
                ], className="elite-card")
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(gamma_chart)
                ], className="elite-card")
            ], width=4)
        ], className="mb-4"),
        
        # Data Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Strike Level Details (Filtered)", className="mb-0 text-info")
                    ]),
                    dbc.CardBody(structure_table)
                ], className="elite-card")
            ], width=12)
        ])
        
    ], className="elite-dashboard-container")
    
    logger.info("Structure mode layout created successfully with FILTERED market data")
    return layout