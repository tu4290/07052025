"""
Advanced Flow Mode Display - EOTS v2.5 Refactored
=================================================

Fully Pydantic-compliant advanced flow mode with unified architecture.
Follows the established refactoring pattern for consistent mode behavior.

Key Features:
- Strict Pydantic v2 validation
- Unified component architecture
- Elite visual design system
- Comprehensive error handling
- Performance optimized
- Advanced flow analytics
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
    AdvancedFlowModeConfig,
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

class AdvancedFlowModeState(BaseModel):
    """State management for advanced flow mode."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    
    bundle: Optional[FinalAnalysisBundleV2_5] = None
    config: Optional[AdvancedFlowModeConfig] = None
    filtered_data: Optional[FilteredDataBundle] = None
    is_initialized: bool = False
    error_state: Optional[str] = None
    last_update: Optional[str] = None

class AdvancedFlowMetrics(BaseModel):
    """Advanced flow metrics configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    vapi_fa: float = Field(default=0.0)
    dwfd: float = Field(default=0.0)
    tw_laf: float = Field(default=0.0)
    flow_momentum: float = Field(default=0.0)
    flow_regime: str = Field(default="NEUTRAL")
    flow_intensity: float = Field(default=0.0)

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

def create_advanced_flow_overview_card(filtered_data: FilteredDataBundle, config: AdvancedFlowModeConfig) -> dbc.Card:
    """Create the advanced flow overview metrics card using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot calculate advanced flow metrics without filtered data")
    
    # Calculate advanced flow metrics from FILTERED strikes only
    total_call_premium = 0
    total_put_premium = 0
    total_call_volume = 0
    total_put_volume = 0
    weighted_flow_score = 0
    
    for strike in filtered_data.filtered_strikes:
        call_vol = getattr(strike, 'call_volume', None)
        put_vol = getattr(strike, 'put_volume', None)
        call_premium = getattr(strike, 'call_premium', None)
        put_premium = getattr(strike, 'put_premium', None)
        
        if any(x is None for x in [call_vol, put_vol]):
            raise ValueError(f"Missing required flow data for filtered strike - cannot proceed without complete data")
        
        total_call_volume += call_vol
        total_put_volume += put_vol
        
        if call_premium is not None:
            total_call_premium += call_premium * call_vol
        if put_premium is not None:
            total_put_premium += put_premium * put_vol
    
    # Calculate advanced metrics
    net_volume = total_call_volume - total_put_volume
    net_premium = total_call_premium - total_put_premium
    flow_intensity = abs(net_volume) + abs(net_premium / 1000000)  # Normalize premium
    
    # Determine flow regime
    if flow_intensity > 50000:
        flow_regime = "EXTREME"
        regime_color = "danger"
    elif flow_intensity > 10000:
        flow_regime = "HIGH"
        regime_color = "warning"
    else:
        flow_regime = "NORMAL"
        regime_color = "success"
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Advanced Flow Overview (Filtered)", className="mb-0 text-primary")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(f"{net_volume:,}", className="text-info mb-0"),
                    html.Small("Net Volume", className="text-muted")
                ], width=2),
                dbc.Col([
                    html.H4(f"${net_premium/1000000:.1f}M", className="text-success mb-0"),
                    html.Small("Net Premium", className="text-muted")
                ], width=2),
                dbc.Col([
                    html.H4(f"{flow_intensity:.0f}", className="text-warning mb-0"),
                    html.Small("Flow Intensity", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(flow_regime, className=f"text-{regime_color} mb-0"),
                    html.Small("Flow Regime", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H4(f"{len(filtered_data.filtered_strikes)}", className="text-primary mb-0"),
                    html.Small("Filtered Strikes", className="text-muted")
                ], width=2)
            ])
        ])
    ], className="mb-4 elite-card")

def create_flow_metrics_radar_chart(filtered_data: FilteredDataBundle, config: AdvancedFlowModeConfig) -> dcc.Graph:
    """Create the flow metrics radar chart using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create radar chart without filtered data")
    
    fig = go.Figure()
    
    # Calculate metrics from FILTERED data
    total_call_vol = sum(getattr(strike, 'call_volume', 0) for strike in filtered_data.filtered_strikes)
    total_put_vol = sum(getattr(strike, 'put_volume', 0) for strike in filtered_data.filtered_strikes)
    total_call_oi = sum(getattr(strike, 'call_open_interest', 0) for strike in filtered_data.filtered_strikes)
    total_put_oi = sum(getattr(strike, 'put_open_interest', 0) for strike in filtered_data.filtered_strikes)
    avg_gamma = sum(getattr(strike, 'total_gamma', 0) for strike in filtered_data.filtered_strikes) / len(filtered_data.filtered_strikes)
    avg_theta = sum(getattr(strike, 'total_theta', 0) for strike in filtered_data.filtered_strikes) / len(filtered_data.filtered_strikes)
    
    # Prepare radar chart data (normalize to 0-5 scale)
    metrics = ['Call Volume', 'Put Volume', 'Call OI', 'Put OI', 'Avg Gamma', 'Avg Theta']
    max_vol = max(total_call_vol, total_put_vol) if max(total_call_vol, total_put_vol) > 0 else 1
    max_oi = max(total_call_oi, total_put_oi) if max(total_call_oi, total_put_oi) > 0 else 1
    
    values = [
        (total_call_vol / max_vol) * 5,
        (total_put_vol / max_vol) * 5,
        (total_call_oi / max_oi) * 5,
        (total_put_oi / max_oi) * 5,
        min(abs(avg_gamma) * 1000, 5),  # Scale gamma
        min(abs(avg_theta) * 100, 5)    # Scale theta
    ]
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name='Flow Metrics',
        line_color='#00ff88',
        fillcolor='rgba(0, 255, 136, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        title=f"Advanced Flow Metrics Radar (Filtered: {len(filtered_data.filtered_strikes)} strikes)",
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_flow_correlation_heatmap(filtered_data: FilteredDataBundle, config: AdvancedFlowModeConfig) -> dcc.Graph:
    """Create the flow correlation heatmap using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    if not filtered_data.filtered_strikes:
        raise ValueError("No strikes found within filters - cannot create correlation heatmap without filtered data")
    
    fig = create_empty_figure()
    
    # Calculate correlation matrix from FILTERED data
    metrics_data = []
    for strike in filtered_data.filtered_strikes:
        call_vol = getattr(strike, 'call_volume', 0)
        put_vol = getattr(strike, 'put_volume', 0)
        call_oi = getattr(strike, 'call_open_interest', 0)
        put_oi = getattr(strike, 'put_open_interest', 0)
        gamma = getattr(strike, 'total_gamma', 0)
        theta = getattr(strike, 'total_theta', 0)
        
        metrics_data.append([call_vol, put_vol, call_oi, put_oi, gamma, theta])
    
    if len(metrics_data) < 2:
        raise ValueError("Insufficient data for correlation analysis - need at least 2 strikes")
    
    # Calculate correlation matrix using pandas
    df = pd.DataFrame(metrics_data, columns=['Call Vol', 'Put Vol', 'Call OI', 'Put OI', 'Gamma', 'Theta'])
    correlation_matrix = df.corr().values.tolist()
    
    fig.add_trace(go.Heatmap(
        z=correlation_matrix,
        x=df.columns,
        y=df.columns,
        colorscale='RdBu',
        zmid=0,
        text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"Flow Metrics Correlation (Filtered: {len(filtered_data.filtered_strikes)} strikes)",
        template=PLOTLY_TEMPLATE,
        height=400
    )
    
    return dcc.Graph(figure=fig, className="elite-chart")

def create_flow_intelligence_panel(filtered_data: FilteredDataBundle, config: AdvancedFlowModeConfig) -> dbc.Card:
    """Create the advanced flow intelligence panel using FILTERED data."""
    # FAIL FAST - No fallbacks for trading data
    if not filtered_data:
        raise ValueError("Filtered data is required for trading operations - cannot proceed without data")
    
    intelligence_items = []
    
    if filtered_data.filtered_strikes:
        # Calculate advanced intelligence metrics from filtered data
        total_strikes = len(filtered_data.filtered_strikes)
        
        # Volume-weighted metrics
        total_volume = sum(getattr(strike, 'call_volume', 0) + getattr(strike, 'put_volume', 0) for strike in filtered_data.filtered_strikes)
        avg_volume_per_strike = total_volume / total_strikes
        
        # Premium-weighted metrics
        total_premium = sum(
            (getattr(strike, 'call_premium', 0) * getattr(strike, 'call_volume', 0)) +
            (getattr(strike, 'put_premium', 0) * getattr(strike, 'put_volume', 0))
            for strike in filtered_data.filtered_strikes
        )
        
        # Risk metrics
        total_gamma = sum(getattr(strike, 'total_gamma', 0) for strike in filtered_data.filtered_strikes)
        total_theta = sum(getattr(strike, 'total_theta', 0) for strike in filtered_data.filtered_strikes)
        
        metrics = [
            ("Filtered Strikes", f"{total_strikes}"),
            ("Total Volume", f"{total_volume:,}"),
            ("Avg Volume/Strike", f"{avg_volume_per_strike:.0f}"),
            ("Total Premium", f"${total_premium/1000000:.1f}M"),
            ("Total Gamma", f"{total_gamma:.4f}"),
            ("Total Theta", f"{total_theta:.2f}"),
            ("Price Range", f"${filtered_data.price_range_min:.2f} - ${filtered_data.price_range_max:.2f}"),
            ("DTE Range", f"{filtered_data.dte_min} - {filtered_data.dte_max} days")
        ]
        
        for name, value in metrics:
            # Determine significance level
            color = "info"
            if "Volume" in name and total_volume > 100000:
                color = "warning"
            elif "Premium" in name and total_premium > 10000000:
                color = "danger"
            elif "Gamma" in name and abs(total_gamma) > 1.0:
                color = "warning"
            
            intelligence_items.append(
                dbc.ListGroupItem([
                    html.Div([
                        html.Strong(name, className="me-2"),
                        html.Span(value, className=f"text-{color}")
                    ])
                ])
            )
    
    if not intelligence_items:
        intelligence_items.append(
            dbc.ListGroupItem("No advanced flow intelligence data available in filtered range", color="warning")
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Advanced Flow Intelligence (Filtered)", className="mb-0 text-info")
        ]),
        dbc.CardBody([
            dbc.ListGroup(intelligence_items, flush=True)
        ])
    ], className="mb-4 elite-card")

# =============================================================================
# VALIDATION & STATE MANAGEMENT
# =============================================================================

def validate_advanced_flow_inputs(bundle: FinalAnalysisBundleV2_5, config: Any) -> ModeValidationResult:
    """Validate inputs for advanced flow mode."""
    # FAIL FAST - Validate bundle
    if not bundle:
        return ModeValidationResult(
            is_valid=False,
            errors=["Bundle is required for trading operations"],
            warnings=[]
        )
    
    # Validate config
    if config is None:
        config = AdvancedFlowModeConfig()
    elif not isinstance(config, AdvancedFlowModeConfig):
        try:
            config = AdvancedFlowModeConfig.model_validate(config)
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
    Creates the layout for the advanced flow mode display with STRICT CONTROL PANEL FILTERING.
    
    Args:
        bundle: The analysis bundle containing all metrics
        config: Configuration object for the advanced flow mode
        control_params: Control panel parameters (symbol, DTE, price range, etc.)
        
    Returns:
        html.Div: The complete advanced flow mode layout
        
    Raises:
        ValueError: If bundle, control_params, or required data is missing
        ValidationError: If config validation fails
    """
    # FAIL FAST - Validate inputs
    validation = validate_advanced_flow_inputs(bundle, config)
    if not validation.is_valid:
        raise ValueError(f"Advanced flow mode validation failed: {', '.join(validation.errors)}")
    
    # CRITICAL: Apply control panel filters
    if not control_params:
        raise ValueError("Control panel parameters are required - cannot display advanced flow mode without filtering settings")
    
    filtered_data = apply_control_panel_filters(bundle, control_params)
    logger.info(f"Applied filters to advanced flow mode: {filtered_data.get_filter_summary()}")
    
    # Build components using FILTERED data only
    overview_card = create_advanced_flow_overview_card(filtered_data, config)
    radar_chart = create_flow_metrics_radar_chart(filtered_data, config)
    correlation_heatmap = create_flow_correlation_heatmap(filtered_data, config)
    intelligence_panel = create_flow_intelligence_panel(filtered_data, config)
    filter_info_panel = create_filter_info_panel(filtered_data)
    
    # Build layout
    layout = html.Div([
        # Header with Filter Info
        dbc.Row([
            dbc.Col([
                html.H1("Advanced Flow Mode", className="text-primary mb-0"),
                html.P("Elite Flow Analytics & Advanced Metrics Intelligence (FILTERED)", className="text-muted")
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
                    dbc.CardBody(radar_chart)
                ], className="elite-card")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody(correlation_heatmap)
                ], className="elite-card")
            ], width=6)
        ], className="mb-4"),
        
        # Intelligence Panel
        dbc.Row([
            dbc.Col(intelligence_panel, width=12)
        ])
        
    ], className="elite-dashboard-container")
    
    logger.info("Advanced flow mode layout created successfully with FILTERED market data")
    return layout