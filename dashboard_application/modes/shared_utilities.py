"""
Shared Utilities for Dashboard Modes - EOTS v2.5
=================================================

This module provides common functionality for all dashboard modes,
implementing the unified architecture pattern with Pydantic compliance.

Key Features:
- Control panel filtering logic
- Input validation utilities
- Error handling patterns
- Chart creation helpers
- Data transformation utilities

Author: EOTS v2.5 Development Team
Version: 2.5.0 (Pydantic-First Architecture)
"""

import logging
from typing import List, Optional, Any, Dict, Tuple
from datetime import datetime
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

from data_models import (
    FinalAnalysisBundleV2_5,
    ControlPanelParametersV2_5,
    FilteredDataBundle,
    ModeValidationResult,
    ChartDataModel,
    BaseModeState
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONTROL PANEL FILTERING
# =============================================================================

def apply_control_panel_filtering(
    bundle: FinalAnalysisBundleV2_5,
    control_panel_params: ControlPanelParametersV2_5
) -> FilteredDataBundle:
    """
    Apply control panel filters to analysis bundle.
    
    This is the centralized filtering logic that all modes should use
    to ensure consistent behavior across the dashboard.
    
    Args:
        bundle: The analysis data bundle
        control_panel_params: Control panel parameters for filtering
        
    Returns:
        FilteredDataBundle: Filtered data with metadata
    """
    try:
        # Validate symbol match
        if bundle.target_symbol != control_panel_params.symbol:
            logger.warning(
                f"Symbol mismatch: bundle={bundle.target_symbol}, "
                f"control_panel={control_panel_params.symbol}"
            )
        
        # Get current price from underlying aggregates
        underlying_aggregates = bundle.processed_data_bundle.underlying_aggregates
        current_price = underlying_aggregates.current_price
        
        # Calculate price range
        price_range_pct = control_panel_params.price_range_percent / 100.0
        price_min = current_price * (1 - price_range_pct)
        price_max = current_price * (1 + price_range_pct)
        
        # Get original data counts
        strike_data = bundle.processed_data_bundle.strike_level_data_with_metrics or []
        contract_data = bundle.processed_data_bundle.contract_level_data_with_metrics or []
        
        total_strikes = len(strike_data)
        total_contracts = len(contract_data)
        
        # Filter strikes by price range
        filtered_strikes = []
        if strike_data:
            filtered_strikes = [
                strike for strike in strike_data
                if price_min <= strike.strike <= price_max
            ]
            logger.debug(
                f"Filtered strikes: {len(filtered_strikes)}/{total_strikes} "
                f"within range {price_min:.2f}-{price_max:.2f}"
            )
        
        # Filter contracts by DTE range
        filtered_contracts = []
        if contract_data:
            filtered_contracts = [
                contract for contract in contract_data
                if control_panel_params.dte_min <= contract.dte <= control_panel_params.dte_max
            ]
            logger.debug(
                f"Filtered contracts: {len(filtered_contracts)}/{total_contracts} "
                f"within DTE range {control_panel_params.dte_min}-{control_panel_params.dte_max}"
            )
        
        # Create filtered data bundle
        filtered_bundle = FilteredDataBundle(
            filtered_strikes=filtered_strikes,
            filtered_contracts=filtered_contracts,
            price_range_min=price_min,
            price_range_max=price_max,
            current_price=current_price,
            price_range_percent=control_panel_params.price_range_percent,
            dte_min=control_panel_params.dte_min,
            dte_max=control_panel_params.dte_max,
            total_strikes_available=total_strikes,
            total_contracts_available=total_contracts,
            strikes_filtered_count=len(filtered_strikes),
            contracts_filtered_count=len(filtered_contracts),
            filter_applied_at=datetime.now()
        )
        
        logger.info(f"Applied control panel filtering: {filtered_bundle.get_filter_summary()}")
        return filtered_bundle
        
    except Exception as e:
        logger.error(f"Error applying control panel filtering: {e}", exc_info=True)
        
        # Return safe fallback
        return FilteredDataBundle(
            filtered_strikes=[],
            filtered_contracts=[],
            price_range_min=current_price * 0.8 if 'current_price' in locals() else 100.0,
            price_range_max=current_price * 1.2 if 'current_price' in locals() else 120.0,
            current_price=current_price if 'current_price' in locals() else 110.0,
            price_range_percent=control_panel_params.price_range_percent,
            dte_min=control_panel_params.dte_min,
            dte_max=control_panel_params.dte_max,
            total_strikes_available=0,
            total_contracts_available=0,
            strikes_filtered_count=0,
            contracts_filtered_count=0,
            filter_applied_at=datetime.now()
        )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_mode_inputs(bundle: Any, config: Any) -> ModeValidationResult:
    """
    Validate inputs for mode functions.
    
    This ensures all modes receive properly validated inputs before processing.
    
    Args:
        bundle: Analysis bundle (should be FinalAnalysisBundleV2_5)
        config: Configuration object (should be Pydantic model)
        
    Returns:
        ModeValidationResult: Validation result with errors/warnings
    """
    errors = []
    warnings = []
    
    # Validate bundle type
    if not isinstance(bundle, FinalAnalysisBundleV2_5):
        errors.append(f"bundle must be FinalAnalysisBundleV2_5, got {type(bundle)}")
    else:
        # Validate bundle contents
        if not bundle.target_symbol:
            errors.append("bundle.target_symbol is required")
        
        if not bundle.processed_data_bundle:
            errors.append("bundle.processed_data_bundle is required")
        
        if bundle.processed_data_bundle and not bundle.processed_data_bundle.underlying_aggregates:
            warnings.append("bundle.processed_data_bundle.underlying_aggregates is missing")
    
    # Validate config type
    if not hasattr(config, 'model_dump'):
        errors.append(f"config must be a Pydantic model, got {type(config)}")
    
    return ModeValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


def extract_control_panel_params(config: Any) -> Optional[ControlPanelParametersV2_5]:
    """
    Extract control panel parameters from config.
    
    This handles various config structures to find control panel parameters.
    
    Args:
        config: Configuration object
        
    Returns:
        ControlPanelParametersV2_5 or None if not found
    """
    try:
        # Direct access
        if hasattr(config, 'control_panel_params'):
            return config.control_panel_params
        
        # Check for dashboard defaults
        if hasattr(config, 'dashboard_defaults'):
            defaults = config.dashboard_defaults
            return ControlPanelParametersV2_5(
                symbol=defaults.symbol,
                dte_min=defaults.dte_min,
                dte_max=defaults.dte_max,
                price_range_percent=defaults.price_range_percent
            )
        
        # Check for visualization settings
        if hasattr(config, 'visualization_settings') and hasattr(config.visualization_settings, 'dashboard_defaults'):
            defaults = config.visualization_settings.dashboard_defaults
            return ControlPanelParametersV2_5(
                symbol=defaults.symbol,
                dte_min=defaults.dte_min,
                dte_max=defaults.dte_max,
                price_range_percent=defaults.price_range_percent
            )
        
        # Fallback: try to construct from individual fields
        if all(hasattr(config, field) for field in ['symbol', 'dte_min', 'dte_max', 'price_range_percent']):
            return ControlPanelParametersV2_5(
                symbol=config.symbol,
                dte_min=config.dte_min,
                dte_max=config.dte_max,
                price_range_percent=config.price_range_percent
            )
        
        logger.warning("No control panel parameters found in config")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting control panel params: {e}")
        return None


# =============================================================================
# ERROR HANDLING
# =============================================================================

def create_error_layout(error_message: str, mode_name: str = "Dashboard Mode") -> html.Div:
    """
    Create a standardized error layout for modes.
    
    Args:
        error_message: The error message to display
        mode_name: Name of the mode for context
        
    Returns:
        html.Div: Error layout component
    """
    return html.Div([
        dbc.Container([
            dbc.Alert([
                html.H4([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"{mode_name} Error"
                ], className="alert-heading"),
                html.P(error_message),
                html.Hr(),
                html.P([
                    "Please check the system logs for more details. ",
                    "If this error persists, contact the system administrator."
                ], className="mb-0 small")
            ], color="danger", className="mt-4")
        ], fluid=True)
    ])


def create_warning_layout(warnings: List[str], mode_name: str = "Dashboard Mode") -> html.Div:
    """
    Create a standardized warning layout for modes.
    
    Args:
        warnings: List of warning messages
        mode_name: Name of the mode for context
        
    Returns:
        html.Div: Warning layout component
    """
    if not warnings:
        return html.Div()
    
    warning_items = [html.Li(warning) for warning in warnings]
    
    return dbc.Alert([
        html.H6([
            html.I(className="fas fa-exclamation-circle me-2"),
            f"{mode_name} Warnings"
        ]),
        html.Ul(warning_items, className="mb-0")
    ], color="warning", className="mb-3")


def create_empty_chart(message: str = "No data available", height: int = 400) -> dcc.Graph:
    """
    Create an empty chart with a message.
    
    Args:
        message: Message to display
        height: Chart height
        
    Returns:
        dcc.Graph: Empty chart component
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        height=height,
        template="plotly_dark",
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )
    
    return dcc.Graph(figure=fig)


# =============================================================================
# CHART UTILITIES
# =============================================================================

def add_control_panel_annotations(
    fig: go.Figure,
    filtered_data: FilteredDataBundle,
    show_price_lines: bool = True,
    show_filter_info: bool = True
) -> go.Figure:
    """
    Add control panel filter annotations to a chart.
    
    Args:
        fig: Plotly figure to annotate
        filtered_data: Filtered data bundle with range information
        show_price_lines: Whether to show price range lines
        show_filter_info: Whether to show filter information
        
    Returns:
        go.Figure: Annotated figure
    """
    try:
        if show_price_lines:
            # Add current price line
            fig.add_vline(
                x=filtered_data.current_price,
                line_dash="solid",
                line_color="yellow",
                line_width=2,
                annotation_text="Current Price",
                annotation_position="top"
            )
            
            # Add price range lines
            fig.add_vline(
                x=filtered_data.price_range_min,
                line_dash="dash",
                line_color="orange",
                line_width=1,
                annotation_text=f"Min Range ({filtered_data.price_range_percent}%)",
                annotation_position="top"
            )
            
            fig.add_vline(
                x=filtered_data.price_range_max,
                line_dash="dash",
                line_color="orange", 
                line_width=1,
                annotation_text=f"Max Range ({filtered_data.price_range_percent}%)",
                annotation_position="top"
            )
        
        if show_filter_info:
            # Add filter summary annotation
            filter_text = (
                f"Filters Applied: DTE {filtered_data.dte_min}-{filtered_data.dte_max} | "
                f"Price ±{filtered_data.price_range_percent}% | "
                f"Strikes: {filtered_data.strikes_filtered_count}/{filtered_data.total_strikes_available} | "
                f"Contracts: {filtered_data.contracts_filtered_count}/{filtered_data.total_contracts_available}"
            )
            
            fig.add_annotation(
                text=filter_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="lightgray"),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="gray",
                borderwidth=1
            )
    
    except Exception as e:
        logger.error(f"Error adding control panel annotations: {e}")
    
    return fig


def apply_standard_chart_styling(
    fig: go.Figure,
    title: str,
    height: int = 400,
    show_timestamp: bool = True
) -> go.Figure:
    """
    Apply standard styling to charts across all modes.
    
    Args:
        fig: Plotly figure to style
        title: Chart title
        height: Chart height
        show_timestamp: Whether to show timestamp
        
    Returns:
        go.Figure: Styled figure
    """
    try:
        # Update layout with standard styling
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color="white"),
                x=0.02
            ),
            height=height,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)",
            font=dict(color="white"),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add timestamp if requested
        if show_timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.add_annotation(
                text=f"Updated: {timestamp}",
                xref="paper", yref="paper",
                x=0.98, y=0.02,
                showarrow=False,
                font=dict(size=8, color="lightgray"),
                xanchor="right"
            )
    
    except Exception as e:
        logger.error(f"Error applying chart styling: {e}")
    
    return fig


# =============================================================================
# DATA TRANSFORMATION UTILITIES
# =============================================================================

def safe_get_metric(data: Any, metric_path: str, default: Any = 0.0) -> Any:
    """
    Safely get a metric from nested data structures.
    
    Args:
        data: Data object to extract from
        metric_path: Dot-separated path to metric (e.g., "underlying.vri_0dte")
        default: Default value if metric not found
        
    Returns:
        Any: Metric value or default
    """
    try:
        if not data:
            return default
        
        # Handle Pydantic models
        if hasattr(data, 'model_dump'):
            data_dict = data.model_dump()
        elif isinstance(data, dict):
            data_dict = data
        else:
            # Try to access as attributes
            current = data
            for part in metric_path.split('.'):
                current = getattr(current, part, None)
                if current is None:
                    return default
            return current
        
        # Navigate through dictionary
        current = data_dict
        for part in metric_path.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current if current is not None else default
        
    except Exception as e:
        logger.debug(f"Error getting metric {metric_path}: {e}")
        return default


def create_filter_info_card(filtered_data: FilteredDataBundle) -> dbc.Card:
    """
    Create an information card showing current filter settings.
    
    Args:
        filtered_data: Filtered data bundle
        
    Returns:
        dbc.Card: Information card component
    """
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-filter me-2"),
            "Active Filters"
        ]),
        dbc.CardBody([
            html.P([
                html.Strong("Symbol: "), 
                f"Current Price ${filtered_data.current_price:.2f}"
            ], className="mb-1"),
            html.P([
                html.Strong("Price Range: "), 
                f"${filtered_data.price_range_min:.2f} - ${filtered_data.price_range_max:.2f} "
                f"(±{filtered_data.price_range_percent}%)"
            ], className="mb-1"),
            html.P([
                html.Strong("DTE Range: "), 
                f"{filtered_data.dte_min} - {filtered_data.dte_max} days"
            ], className="mb-1"),
            html.P([
                html.Strong("Data Filtered: "), 
                f"{filtered_data.strikes_filtered_count}/{filtered_data.total_strikes_available} strikes, "
                f"{filtered_data.contracts_filtered_count}/{filtered_data.total_contracts_available} contracts"
            ], className="mb-0")
        ])
    ], className="mb-3", color="info", outline=True)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Filtering functions
    'apply_control_panel_filtering',
    
    # Validation functions
    'validate_mode_inputs',
    'extract_control_panel_params',
    
    # Error handling
    'create_error_layout',
    'create_warning_layout',
    'create_empty_chart',
    
    # Chart utilities
    'add_control_panel_annotations',
    'apply_standard_chart_styling',
    
    # Data utilities
    'safe_get_metric',
    'create_filter_info_card',
]