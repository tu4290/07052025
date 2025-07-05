"""
Main Dashboard Display - EOTS v2.5 Refactored
=============================================

Fully Pydantic-compliant main dashboard mode with control panel integration.
This serves as the template for all other dashboard modes.

Key Features:
- Full Pydantic v2 compliance
- Control panel parameter filtering
- Unified architecture pattern
- Error handling and validation
- Consistent chart styling

Author: EOTS v2.5 Development Team
Version: 2.5.0 (Pydantic-First Refactor)
"""

import logging
from typing import Optional, List
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc

# EOTS Pydantic Models
from data_models import (
    FinalAnalysisBundleV2_5,
    ControlPanelParametersV2_5,
    MainDashboardState,
    MainDashboardConfig,
    FilteredDataBundle,
    ProcessedUnderlyingAggregatesV2_5,
    ProcessedStrikeLevelMetricsV2_5
)

# Shared utilities
from .shared_utilities import (
    apply_control_panel_filtering,
    validate_mode_inputs,
    extract_control_panel_params,
    create_error_layout,
    create_warning_layout,
    create_empty_chart,
    add_control_panel_annotations,
    apply_standard_chart_styling,
    safe_get_metric,
    create_filter_info_card
)

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_layout(bundle: FinalAnalysisBundleV2_5, config: any) -> html.Div:
    """
    Main entry point for main dashboard layout.
    
    This function validates inputs, extracts control panel parameters,
    and delegates to the state-based renderer.
    
    Args:
        bundle: Analysis data bundle (must be FinalAnalysisBundleV2_5)
        config: Configuration object (must be Pydantic model)
        
    Returns:
        html.Div: Complete dashboard layout
    """
    try:
        # Validate inputs
        validation_result = validate_mode_inputs(bundle, config)
        if not validation_result.is_valid:
            error_msg = f"Invalid inputs: {'; '.join(validation_result.errors)}"
            logger.error(error_msg)
            return create_error_layout(error_msg, "Main Dashboard")
        
        # Extract control panel parameters
        control_panel_params = extract_control_panel_params(config)
        if not control_panel_params:
            error_msg = "Control panel parameters not found in config"
            logger.error(error_msg)
            return create_error_layout(error_msg, "Main Dashboard")
        
        # Transform to mode state
        mode_state = transform_bundle_to_main_dashboard_state(bundle, control_panel_params)
        
        # Render layout
        return render_main_dashboard_layout(mode_state)
        
    except Exception as e:
        error_msg = f"Failed to create main dashboard layout: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return create_error_layout(error_msg, "Main Dashboard")


# =============================================================================
# DATA TRANSFORMATION
# =============================================================================

def transform_bundle_to_main_dashboard_state(
    bundle: FinalAnalysisBundleV2_5,
    control_panel_params: ControlPanelParametersV2_5
) -> MainDashboardState:
    """
    Transform analysis bundle to main dashboard state with filtering applied.
    
    Args:
        bundle: Analysis data bundle
        control_panel_params: Control panel parameters for filtering
        
    Returns:
        MainDashboardState: Transformed and filtered state
    """
    try:
        # Apply control panel filtering
        filtered_data = apply_control_panel_filtering(bundle, control_panel_params)
        
        # Extract regime indicator data
        regime_indicator_data = extract_regime_indicator_data(bundle, filtered_data)
        
        # Extract flow gauges data
        flow_gauges_data = extract_flow_gauges_data(bundle, filtered_data)
        
        # Extract key metrics summary
        key_metrics_summary = extract_key_metrics_summary(bundle, filtered_data)
        
        # Create state
        return MainDashboardState(
            target_symbol=bundle.target_symbol,
            control_panel_params=control_panel_params,
            analysis_bundle=bundle,
            filtered_data=filtered_data,
            regime_indicator_data=regime_indicator_data,
            flow_gauges_data=flow_gauges_data,
            key_metrics_summary=key_metrics_summary,
            warnings=[]
        )
        
    except Exception as e:
        logger.error(f"Error transforming bundle to main dashboard state: {e}", exc_info=True)
        
        # Return error state
        return MainDashboardState(
            target_symbol=bundle.target_symbol,
            control_panel_params=control_panel_params,
            analysis_bundle=bundle,
            error_message=f"Data transformation failed: {str(e)}",
            warnings=["Some data may not be available due to processing errors"]
        )


def extract_regime_indicator_data(
    bundle: FinalAnalysisBundleV2_5,
    filtered_data: FilteredDataBundle
) -> dict:
    """Extract market regime indicator data from bundle."""
    try:
        underlying_data = bundle.processed_data_bundle.underlying_aggregates
        
        return {
            "current_regime": safe_get_metric(underlying_data, "market_regime", "Unknown"),
            "regime_strength": safe_get_metric(underlying_data, "regime_strength", 0.0),
            "regime_change_probability": safe_get_metric(underlying_data, "regime_change_prob", 0.0),
            "last_regime_change": safe_get_metric(underlying_data, "last_regime_change", "N/A"),
            "volatility_regime": safe_get_metric(underlying_data, "volatility_regime", "Normal"),
            "trend_direction": safe_get_metric(underlying_data, "trend_direction", "Neutral")
        }
    except Exception as e:
        logger.error(f"Error extracting regime indicator data: {e}")
        return {
            "current_regime": "Unknown",
            "regime_strength": 0.0,
            "regime_change_probability": 0.0,
            "last_regime_change": "N/A",
            "volatility_regime": "Unknown",
            "trend_direction": "Unknown"
        }


def extract_flow_gauges_data(
    bundle: FinalAnalysisBundleV2_5,
    filtered_data: FilteredDataBundle
) -> dict:
    """Extract flow gauges data from bundle."""
    try:
        underlying_data = bundle.processed_data_bundle.underlying_aggregates
        
        return {
            "net_flow": safe_get_metric(underlying_data, "net_flow_0dte", 0.0),
            "call_flow": safe_get_metric(underlying_data, "call_flow_0dte", 0.0),
            "put_flow": safe_get_metric(underlying_data, "put_flow_0dte", 0.0),
            "flow_ratio": safe_get_metric(underlying_data, "flow_ratio_0dte", 1.0),
            "volume_weighted_flow": safe_get_metric(underlying_data, "vw_flow_0dte", 0.0),
            "flow_momentum": safe_get_metric(underlying_data, "flow_momentum", 0.0)
        }
    except Exception as e:
        logger.error(f"Error extracting flow gauges data: {e}")
        return {
            "net_flow": 0.0,
            "call_flow": 0.0,
            "put_flow": 0.0,
            "flow_ratio": 1.0,
            "volume_weighted_flow": 0.0,
            "flow_momentum": 0.0
        }


def extract_key_metrics_summary(
    bundle: FinalAnalysisBundleV2_5,
    filtered_data: FilteredDataBundle
) -> dict:
    """Extract key metrics summary from bundle."""
    try:
        underlying_data = bundle.processed_data_bundle.underlying_aggregates
        
        return {
            "current_price": filtered_data.current_price,
            "price_change": safe_get_metric(underlying_data, "price_change", 0.0),
            "price_change_percent": safe_get_metric(underlying_data, "price_change_percent", 0.0),
            "volume": safe_get_metric(underlying_data, "volume", 0),
            "avg_volume": safe_get_metric(underlying_data, "avg_volume", 0),
            "implied_volatility": safe_get_metric(underlying_data, "implied_volatility", 0.0),
            "realized_volatility": safe_get_metric(underlying_data, "realized_volatility", 0.0),
            "vix_level": safe_get_metric(underlying_data, "vix_level", 0.0),
            "open_interest": safe_get_metric(underlying_data, "total_open_interest", 0),
            "put_call_ratio": safe_get_metric(underlying_data, "put_call_ratio", 1.0)
        }
    except Exception as e:
        logger.error(f"Error extracting key metrics summary: {e}")
        return {
            "current_price": filtered_data.current_price,
            "price_change": 0.0,
            "price_change_percent": 0.0,
            "volume": 0,
            "avg_volume": 0,
            "implied_volatility": 0.0,
            "realized_volatility": 0.0,
            "vix_level": 0.0,
            "open_interest": 0,
            "put_call_ratio": 1.0
        }


# =============================================================================
# LAYOUT RENDERING
# =============================================================================

def render_main_dashboard_layout(state: MainDashboardState) -> html.Div:
    """
    Pure renderer for main dashboard layout.
    
    Takes validated state and returns complete layout without side effects.
    
    Args:
        state: Main dashboard state
        
    Returns:
        html.Div: Complete dashboard layout
    """
    try:
        # Handle error state
        if state.error_message:
            return create_error_layout(state.error_message, "Main Dashboard")
        
        # Create warning alerts if any
        warning_layout = create_warning_layout(state.warnings, "Main Dashboard")
        
        # Create filter info card
        filter_info = create_filter_info_card(state.filtered_data) if state.filtered_data else html.Div()
        
        # Create main components
        components = [
            create_header_section(state),
            filter_info,
            warning_layout,
            create_regime_indicator_section(state),
            create_flow_gauges_section(state),
            create_key_metrics_section(state),
            create_charts_section(state)
        ]
        
        return html.Div([
            dbc.Container([
                dbc.Row([
                    dbc.Col(component, width=12, className="mb-4")
                    for component in components if component
                ])
            ], fluid=True)
        ])
        
    except Exception as e:
        logger.error(f"Error rendering main dashboard layout: {e}", exc_info=True)
        return create_error_layout(f"Layout rendering failed: {str(e)}", "Main Dashboard")


def create_header_section(state: MainDashboardState) -> html.Div:
    """Create the header section with symbol and timestamp."""
    return html.Div([
        html.H2([
            f"Main Dashboard - {state.target_symbol}",
            html.Small(
                f" | Updated: {state.last_updated.strftime('%Y-%m-%d %H:%M:%S')}",
                className="text-muted ms-3"
            )
        ], className="mb-0")
    ])


def create_regime_indicator_section(state: MainDashboardState) -> dbc.Card:
    """Create the market regime indicator section."""
    if not state.regime_indicator_data:
        return create_empty_chart("Regime data not available")
    
    regime_data = state.regime_indicator_data
    
    # Determine regime color
    regime_colors = {
        "Bullish": "success",
        "Bearish": "danger", 
        "Neutral": "warning",
        "Unknown": "secondary"
    }
    regime_color = regime_colors.get(regime_data["current_regime"], "secondary")
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-chart-line me-2"),
            "Market Regime Indicator"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(regime_data["current_regime"], className=f"text-{regime_color}"),
                    html.P("Current Regime", className="text-muted mb-0")
                ], width=3),
                dbc.Col([
                    html.H4(f"{regime_data['regime_strength']:.2f}"),
                    html.P("Strength", className="text-muted mb-0")
                ], width=3),
                dbc.Col([
                    html.H4(f"{regime_data['regime_change_probability']:.1%}"),
                    html.P("Change Probability", className="text-muted mb-0")
                ], width=3),
                dbc.Col([
                    html.H4(regime_data["volatility_regime"]),
                    html.P("Volatility Regime", className="text-muted mb-0")
                ], width=3)
            ])
        ])
    ], className="mb-4")


def create_flow_gauges_section(state: MainDashboardState) -> dbc.Card:
    """Create the flow gauges section."""
    if not state.flow_gauges_data:
        return create_empty_chart("Flow data not available")
    
    flow_data = state.flow_gauges_data
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-tachometer-alt me-2"),
            "Options Flow Gauges"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(f"${flow_data['net_flow']:,.0f}"),
                    html.P("Net Flow", className="text-muted mb-0")
                ], width=2),
                dbc.Col([
                    html.H4(f"${flow_data['call_flow']:,.0f}"),
                    html.P("Call Flow", className="text-muted mb-0")
                ], width=2),
                dbc.Col([
                    html.H4(f"${flow_data['put_flow']:,.0f}"),
                    html.P("Put Flow", className="text-muted mb-0")
                ], width=2),
                dbc.Col([
                    html.H4(f"{flow_data['flow_ratio']:.2f}"),
                    html.P("Flow Ratio", className="text-muted mb-0")
                ], width=2),
                dbc.Col([
                    html.H4(f"${flow_data['volume_weighted_flow']:,.0f}"),
                    html.P("VW Flow", className="text-muted mb-0")
                ], width=2),
                dbc.Col([
                    html.H4(f"{flow_data['flow_momentum']:,.0f}"),
                    html.P("Momentum", className="text-muted mb-0")
                ], width=2)
            ])
        ])
    ], className="mb-4")


def create_key_metrics_section(state: MainDashboardState) -> dbc.Card:
    """Create the key metrics summary section."""
    if not state.key_metrics_summary:
        return create_empty_chart("Metrics data not available")
    
    metrics = state.key_metrics_summary
    
    # Determine price change color
    price_change = metrics["price_change_percent"]
    price_color = "success" if price_change > 0 else "danger" if price_change < 0 else "secondary"
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-chart-bar me-2"),
            "Key Metrics Summary"
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4(f"${metrics['current_price']:.2f}"),
                    html.P([
                        f"{metrics['price_change']:+.2f} ",
                        html.Span(f"({price_change:+.2%})", className=f"text-{price_color}")
                    ], className="text-muted mb-0")
                ], width=3),
                dbc.Col([
                    html.H4(f"{metrics['volume']:,}"),
                    html.P(f"Avg: {metrics['avg_volume']:,}", className="text-muted mb-0")
                ], width=3),
                dbc.Col([
                    html.H4(f"{metrics['implied_volatility']:.1%}"),
                    html.P("Implied Vol", className="text-muted mb-0")
                ], width=3),
                dbc.Col([
                    html.H4(f"{metrics['put_call_ratio']:.2f}"),
                    html.P("Put/Call Ratio", className="text-muted mb-0")
                ], width=3)
            ])
        ])
    ], className="mb-4")


def create_charts_section(state: MainDashboardState) -> html.Div:
    """Create the charts section with filtered data visualizations."""
    if not state.filtered_data:
        return create_empty_chart("No data available for charts")
    
    charts = [
        create_strike_distribution_chart(state),
        create_volume_profile_chart(state),
        create_time_series_chart(state),
        create_volatility_surface_chart(state)
    ]
    
    return html.Div([
        dbc.Row([
            dbc.Col(chart, width=6, className="mb-4")
            for chart in charts
        ])
    ])


def create_strike_distribution_chart(state: MainDashboardState) -> dcc.Graph:
    """Create strike distribution chart using filtered data."""
    try:
        if not state.filtered_data.filtered_strikes:
            return create_empty_chart("No strike data available")
        
        # Extract strike data
        strikes = [strike.strike for strike in state.filtered_data.filtered_strikes]
        volumes = [safe_get_metric(strike, "total_volume", 0) for strike in state.filtered_data.filtered_strikes]
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=strikes,
            y=volumes,
            name="Volume by Strike",
            marker_color="lightblue"
        ))
        
        # Add control panel annotations
        fig = add_control_panel_annotations(fig, state.filtered_data)
        
        # Apply standard styling
        fig = apply_standard_chart_styling(
            fig,
            f"{state.target_symbol} - Strike Distribution",
            height=400
        )
        
        fig.update_xaxis(title="Strike Price")
        fig.update_yaxis(title="Volume")
        
        return dcc.Graph(figure=fig)
        
    except Exception as e:
        logger.error(f"Error creating strike distribution chart: {e}")
        return create_empty_chart("Error creating chart")


def create_volume_profile_chart(state: MainDashboardState) -> dcc.Graph:
    """Create volume profile chart using filtered data."""
    try:
        if not state.filtered_data.filtered_contracts:
            return create_empty_chart("No contract data available")
        
        # Extract contract data
        strikes = [contract.strike for contract in state.filtered_data.filtered_contracts]
        volumes = [safe_get_metric(contract, "volume", 0) for contract in state.filtered_data.filtered_contracts]
        
        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=strikes,
            y=volumes,
            mode="markers+lines",
            name="Volume Profile",
            marker=dict(size=8, color="orange")
        ))
        
        # Add control panel annotations
        fig = add_control_panel_annotations(fig, state.filtered_data)
        
        # Apply standard styling
        fig = apply_standard_chart_styling(
            fig,
            f"{state.target_symbol} - Volume Profile",
            height=400
        )
        
        fig.update_xaxis(title="Strike Price")
        fig.update_yaxis(title="Volume")
        
        return dcc.Graph(figure=fig)
        
    except Exception as e:
        logger.error(f"Error creating volume profile chart: {e}")
        return create_empty_chart("Error creating chart")


def create_time_series_chart(state: MainDashboardState) -> dcc.Graph:
    """Create time series chart placeholder."""
    # This would be implemented with actual time series data
    return create_empty_chart("Time series chart - Coming soon")


def create_volatility_surface_chart(state: MainDashboardState) -> dcc.Graph:
    """Create volatility surface chart placeholder."""
    # This would be implemented with actual volatility surface data
    return create_empty_chart("Volatility surface - Coming soon")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'create_layout',
    'transform_bundle_to_main_dashboard_state',
    'render_main_dashboard_layout'
]