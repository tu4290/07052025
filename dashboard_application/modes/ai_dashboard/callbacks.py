"""
AI Dashboard Callbacks v2.5 - Pydantic-First Architecture
==========================================================

This module contains critical callbacks for the AI Hub, refactored to enforce a
strict Pydantic-first, ZERO DICT ACCEPTANCE policy. All data flowing from Dash
stores is immediately validated into Pydantic models.

Key Changes:
- All `Dict` type hints for data structures replaced with Pydantic models.
- Dictionary access patterns (`.get()`, `.keys()`) completely removed.
- Helper functions now accept and return Pydantic models.
- All operations are type-safe and validated end-to-end.

Author: EOTS v2.5 Development Team
Version: 2.5.1
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from dash import Input, Output, html
import plotly.graph_objects as go
from pydantic import BaseModel, ValidationError

# EOTS Pydantic Models - The single source of truth for data structures
from data_models import (
    FinalAnalysisBundleV2_5,
    AdvancedOptionsMetricsV2_5,
    GaugeConfigModel
)
import compliance_decorators_v2_5

logger = logging.getLogger(__name__)

# Define a simple Pydantic model for learning status to avoid using dicts
class AILearningStatusModel(BaseModel):
    """Pydantic model for AI learning status display."""
    status: str = "Unknown"
    last_update: str = "Never"


def register_ai_dashboard_callbacks(app, orchestrator):
    """
    Register AI dashboard callbacks for advanced options metrics and AI hub functionality.
    
    Args:
        app: Dash application instance
        orchestrator: ITS Orchestrator instance for data processing
    """
    
    @app.callback(
        [
            Output('ai-hub-advanced-metrics-display', 'children'),
            Output('ai-hub-lwpai-gauge', 'figure'),
            Output('ai-hub-vabai-gauge', 'figure'),
            Output('ai-hub-aofm-gauge', 'figure'),
            Output('ai-hub-lidb-gauge', 'figure'),
        ],
        [
            Input('main-data-store-id', 'data'),
            Input('symbol-input-id', 'value'),
        ],
        prevent_initial_call=False
    )
    @compliance_decorators_v2_5.track_callback(callback_name='update_ai_hub_advanced_metrics')
    def update_ai_hub_advanced_metrics(main_data: Dict, symbol: str) -> Tuple:
        """
        🚀 PYDANTIC-FIRST: Update AI hub advanced metrics display with real data.
        Receives a dict from dcc.Store and immediately validates it into a Pydantic model.
        """
        logger.info(f"🔥 AI HUB CALLBACK TRIGGERED! symbol={symbol}, main_data_exists={bool(main_data)}")

        if not main_data or not symbol:
            logger.warning(f"❌ AI HUB: Missing data or symbol. main_data={bool(main_data)}, symbol={symbol}")
            return _create_empty_metrics_display()

        try:
            # PYDANTIC-FIRST VALIDATION: Immediately convert dict from dcc.Store into a Pydantic model.
            # This is the gatekeeper for data integrity.
            bundle = FinalAnalysisBundleV2_5.model_validate(main_data)
            logger.info(f"✅ AI HUB: Successfully validated main_data into FinalAnalysisBundleV2_5 for {symbol}")

            # Extract advanced metrics using the validated Pydantic model.
            advanced_metrics = _get_advanced_metrics_from_bundle(bundle)

            if not advanced_metrics:
                logger.error(f"❌ AI HUB: No advanced metrics found in the validated bundle for {symbol}")
                return _create_empty_metrics_display()

            logger.info(f"✅ AI HUB: Found advanced metrics for {symbol}")
            
            # Create metrics display and gauges using Pydantic models
            metrics_display = _create_metrics_display(advanced_metrics, symbol)
            logger.info(f"✅ AI HUB: Created metrics display for {symbol}")

            lwpai_gauge = _create_gauge_figure(GaugeConfigModel(value=advanced_metrics.lwpai or 0.0, title="LWPAI", range_min=-1.0, range_max=1.0))
            vabai_gauge = _create_gauge_figure(GaugeConfigModel(value=advanced_metrics.vabai or 0.0, title="VABAI", range_min=-1.0, range_max=1.0))
            aofm_gauge = _create_gauge_figure(GaugeConfigModel(value=advanced_metrics.aofm or 0.0, title="AOFM", range_min=-1.0, range_max=1.0))
            lidb_gauge = _create_gauge_figure(GaugeConfigModel(value=advanced_metrics.lidb or 0.0, title="LIDB", range_min=-1.0, range_max=1.0))

            logger.info(f"✅ AI HUB: Created all gauges for {symbol}")
            logger.info(f"🎉 AI HUB: Returning complete metrics display for {symbol}")

            return metrics_display, lwpai_gauge, vabai_gauge, aofm_gauge, lidb_gauge

        except ValidationError as e:
            logger.error(f"💥 AI HUB Pydantic Validation ERROR for {symbol}: {e}")
            return _create_error_display(f"Data structure mismatch. See logs.")
        except Exception as e:
            logger.error(f"💥 AI HUB CALLBACK ERROR for {symbol}: {e}", exc_info=True)
            return _create_error_display(str(e))

    @app.callback(
        Output('ai-hub-learning-status', 'children'),
        [
            Input('main-data-store-id', 'data'),
            Input('interval-live-update-id', 'n_intervals'),
        ],
        prevent_initial_call=False
    )
    def update_ai_learning_status(main_data: Dict, n_intervals: int):
        """Update AI learning status display using Pydantic models."""
        try:
            if hasattr(orchestrator, 'adaptive_learning_integration'):
                # Assuming get_learning_status returns a dict, we validate it.
                status_dict = orchestrator.adaptive_learning_integration.get_learning_status()
                learning_status = AILearningStatusModel.model_validate(status_dict or {})
                return _create_learning_status_display(learning_status)
            else:
                return "AI Learning: Not Available"
                
        except Exception as e:
            logger.error(f"Error updating AI learning status: {e}")
            return f"AI Learning: Error - {str(e)}"

def _get_advanced_metrics_from_bundle(bundle: FinalAnalysisBundleV2_5) -> Optional[AdvancedOptionsMetricsV2_5]:
    """
    🚀 PYDANTIC-FIRST: Safely extract advanced options metrics from a validated Pydantic bundle.
    This function uses direct attribute access, trusting the Pydantic model's structure.
    No dictionary checks are needed.
    """
    try:
        if (
            bundle.processed_data_bundle and
            bundle.processed_data_bundle.underlying_data_enriched and
            bundle.processed_data_bundle.underlying_data_enriched.advanced_options_metrics
        ):
            metrics = bundle.processed_data_bundle.underlying_data_enriched.advanced_options_metrics
            if isinstance(metrics, AdvancedOptionsMetricsV2_5):
                logger.info(f"✅ BUNDLE EXTRACT: Found AdvancedOptionsMetricsV2_5 for {bundle.target_symbol}")
                return metrics
            else:
                # This case should ideally not happen in a strict Pydantic environment
                # but provides a safe fallback if data is somehow malformed.
                logger.warning(f"⚠️ BUNDLE EXTRACT: 'advanced_options_metrics' is not a Pydantic model, attempting validation for {bundle.target_symbol}.")
                return AdvancedOptionsMetricsV2_5.model_validate(metrics)

        logger.warning(f"❌ BUNDLE EXTRACT: No advanced options metrics found in bundle for {bundle.target_symbol}")
        return None

    except (AttributeError, ValidationError) as e:
        logger.error(f"💥 BUNDLE EXTRACT: Error accessing or validating advanced metrics from bundle for {bundle.target_symbol}: {e}", exc_info=True)
        return None

def _create_gauge_figure(config: GaugeConfigModel) -> go.Figure:
    """Create a gauge figure for advanced options metrics using a Pydantic config model."""
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=config.value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': config.title, 'font': {'size': 16, 'color': 'white'}},
            gauge={
                'axis': {'range': [config.range_min, config.range_max], 'tickcolor': 'white'},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [config.range_min, 0], 'color': "rgba(255, 0, 0, 0.7)"},
                    {'range': [0, config.range_max], 'color': "rgba(0, 255, 0, 0.7)"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': config.value
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white", 'family': "Arial"},
            height=config.height,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating gauge figure for {config.title}: {e}", exc_info=True)
        return go.Figure()

def _create_metrics_display(metrics: AdvancedOptionsMetricsV2_5, symbol: str) -> List[html.H4 | html.Div]:
    """Create HTML display for advanced options metrics from a Pydantic model."""
    try:
        return [
            html.H4(f"Advanced Options Metrics - {symbol}", style={'color': 'white', 'textAlign': 'center'}),
            html.Div([
                html.P(f"LWPAI: {metrics.lwpai:.6f}" if metrics.lwpai is not None else "LWPAI: N/A", style={'color': 'cyan'}),
                html.P(f"VABAI: {metrics.vabai:.6f}" if metrics.vabai is not None else "VABAI: N/A", style={'color': 'lime'}),
                html.P(f"AOFM: {metrics.aofm:.6f}" if metrics.aofm is not None else "AOFM: N/A", style={'color': 'orange'}),
                html.P(f"LIDB: {metrics.lidb:.6f}" if metrics.lidb is not None else "LIDB: N/A", style={'color': 'magenta'}),
                html.Hr(style={'borderColor': 'gray'}),
                html.P(f"Confidence: {metrics.confidence_score:.3f}" if metrics.confidence_score is not None else "Confidence: N/A", style={'color': 'yellow'}),
                html.P(f"Contracts Analyzed: {metrics.contracts_analyzed}" if metrics.contracts_analyzed is not None else "Contracts: N/A", style={'color': 'lightblue'}),
                html.P(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}", style={'color': 'gray', 'fontSize': '12px'})
            ], style={'padding': '10px'})
        ]
        
    except Exception as e:
        logger.error(f"Error creating metrics display: {e}", exc_info=True)
        return [html.P(f"Error displaying metrics: {str(e)}", style={'color': 'red'})]

def _create_empty_metrics_display() -> Tuple:
    """Create empty display when no metrics are available."""
    empty_display = [
        html.H4("Advanced Options Metrics", style={'color': 'white', 'textAlign': 'center'}),
        html.P("No data available", style={'color': 'gray', 'textAlign': 'center'})
    ]
    
    empty_gauge = go.Figure().update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200
    )
    
    return empty_display, empty_gauge, empty_gauge, empty_gauge, empty_gauge

def _create_error_display(error_msg: str) -> Tuple:
    """Create error display when metrics update fails."""
    error_display = [
        html.H4("Advanced Options Metrics", style={'color': 'white', 'textAlign': 'center'}),
        html.P(f"Error: {error_msg}", style={'color': 'red', 'textAlign': 'center', 'whiteSpace': 'pre-wrap'})
    ]
    
    error_gauge = go.Figure().update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200
    )
    
    return error_display, error_gauge, error_gauge, error_gauge, error_gauge

def _create_learning_status_display(learning_status: Optional[AILearningStatusModel]) -> str:
    """Create learning status display from a Pydantic model."""
    if not learning_status:
        return "AI Learning: Initializing..."
    
    return f"AI Learning: {learning_status.status} | Last Update: {learning_status.last_update}"

def register_collapsible_callbacks(app):
    """
    Register collapsible callbacks for AI dashboard info sections.
    This function is called by the callback manager.
    """
    # This function does not handle complex data models and can remain as is.
    # If it were to handle state, it should use dcc.Store with Pydantic models.
    pass

# Export the registration functions
__all__ = ['register_ai_dashboard_callbacks', 'register_collapsible_callbacks']
