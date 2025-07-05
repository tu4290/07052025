# dashboard_application/modes/ai_dashboard/volatility_gamma_panel.py
"""
Volatility & Gamma Panel for EOTS AI Hub v2.5
===============================================

PYDANTIC-FIRST & ZERO DICT ACCEPTANCE.

This module provides a dedicated, Pydantic-first panel for visualizing real-time
volatility and market regime intelligence, sourced directly from the HuiHui
`market_regime_expert`. It replaces the static metrics container from the
legacy `layouts_metrics.py`.

Key Features:
1.  **Direct HuiHui Integration**: Consumes and validates data from the
    `market_regime_expert` MoE.
2.  **Strict Pydantic Models**: Uses `MetricsPanelModel`, `MetricDisplayModel`, and
    `GaugeConfigModel` for end-to-end type safety.
3.  **Rich Visualization**: Displays key metrics like VRI 2.0, Regime Confidence,
    and Transition Probability using sophisticated gauges.
4.  **State-Driven UI**: The panel's appearance and data are driven entirely by
    the `AIHubStateModel`.

The core flow is:
- `transform_regime_expert_to_panel_model()`: Transforms expert data into a `MetricsPanelModel`.
- `create_volatility_gamma_panel()`: Renders the Dash component from the model.

Author: EOTS v2.5 Development Team
Version: 2.5.2
"""

import logging
from typing import Optional

from dash import html, dcc
import plotly.graph_objects as go  # Needed for creating Figure objects

# EOTS Pydantic Models: The single source of truth for all data structures
from data_models import (
    MOEUnifiedResponseV2_5,
    MarketRegimeAnalysisDetails,
    MetricsPanelModel,
    MetricDisplayModel,
    GaugeConfigModel,
    PanelConfigModel,
    ComponentStatus,
    PanelType
)

# Import shared components and visualization functions
from .components import (
    create_placeholder_card,
    get_unified_text_style,
    get_unified_badge_style,
    AI_COLORS
)
from .visualizations import create_metric_gauge, create_regime_transition_gauge, RegimeTransitionGaugeConfig

logger = logging.getLogger(__name__)


# --- 1. Core Data Transformation Logic ---

def transform_regime_expert_to_panel_model(
    analysis_bundle: Optional[MOEUnifiedResponseV2_5]
) -> MetricsPanelModel:
    """
    Transforms data from the `market_regime_expert` within a MOE bundle into a
    strictly validated `MetricsPanelModel`.

    This function isolates data processing from UI rendering.

    Args:
        analysis_bundle: The unified response from the HuiHui experts (MOE).

    Returns:
        A fully populated and validated MetricsPanelModel for the UI.
    """
    panel_config = PanelConfigModel(
        id="volatility-gamma-panel",
        title="📈 Volatility & Regime",
        panel_type=PanelType.VOLATILITY_GAMMA
    )

    if not analysis_bundle:
        panel_config.status = ComponentStatus.LOADING
        return MetricsPanelModel(panel_config=panel_config, metrics=[])

    try:
        # Safely find and validate the market regime expert's response
        regime_details, expert_confidence = _get_expert_details(analysis_bundle, "market_regime", MarketRegimeAnalysisDetails)

        if not regime_details:
            logger.warning("Volatility Panel: Market Regime Expert details not found or failed.")
            panel_config.status = ComponentStatus.WARNING
            return MetricsPanelModel(panel_config=panel_config, metrics=[])

        # Create a list of metric displays from the validated expert details
        metrics_list = [
            MetricDisplayModel(
                name="VRI 2.0 Score",
                value=regime_details.vri_score,
                gauge_config=GaugeConfigModel(
                    value=regime_details.vri_score,
                    title="VRI 2.0 (Trend Strength)",
                    range_min=0, range_max=100, height=180
                )
            ),
            MetricDisplayModel(
                name="Expert Confidence",
                value=expert_confidence * 100,
                gauge_config=GaugeConfigModel(
                    value=expert_confidence * 100,
                    title="Regime Expert Confidence",
                    range_min=0, range_max=100, height=180
                )
            ),
            # Placeholder for Gamma Exposure as requested.
            # This can be fully implemented when the expert provides gamma data.
            MetricDisplayModel(
                name="Gamma Exposure",
                value=0, # Placeholder value
                gauge_config=GaugeConfigModel(
                    value=0,
                    title="Gamma Exposure (GEX)",
                    range_min=-10, range_max=10, height=180
                )
            )
        ]
        
        panel_config.status = ComponentStatus.OK
        panel_config.last_updated = analysis_bundle.timestamp
        
        return MetricsPanelModel(
            panel_config=panel_config,
            metrics=metrics_list
        )

    except Exception as e:
        logger.error(f"Volatility Panel: Failed to transform expert data. Error: {e}", exc_info=True)
        panel_config.status = ComponentStatus.ERROR
        return MetricsPanelModel(panel_config=panel_config, metrics=[])


# --- 2. UI Rendering Logic ---

def create_volatility_gamma_panel(
    volatility_panel_model: Optional[MetricsPanelModel],
    regime_details: Optional[MarketRegimeAnalysisDetails]
) -> html.Div:
    """
    Assembles the Volatility & Gamma panel from a validated `MetricsPanelModel`.
    This is a pure rendering function.

    Args:
        volatility_panel_model: The Pydantic model containing data for the panel's gauges.
        regime_details: The validated details from the expert for textual and transition gauge display.

    Returns:
        A Dash html.Div representing the complete Volatility & Gamma panel.
    """
    if not volatility_panel_model or volatility_panel_model.panel_config.status != ComponentStatus.OK:
        message = "Awaiting volatility analysis data..."
        if volatility_panel_model and volatility_panel_model.panel_config.status == ComponentStatus.ERROR:
            message = "An error occurred processing volatility data."
        elif volatility_panel_model and volatility_panel_model.panel_config.status == ComponentStatus.WARNING:
            message = "Volatility expert data is unavailable or failed."
        return create_placeholder_card("Volatility & Regime", message)

    try:
        # Create the grid of standard gauges
        gauge_grid = [
            html.Div(
                className="col-lg-4 col-md-6 mb-3",
                children=dcc.Graph(
                    figure=create_metric_gauge(metric.gauge_config),
                    config={'displayModeBar': False}
                )
            ) for metric in volatility_panel_model.metrics
        ]

        # Create the special regime transition gauge
        transition_gauge_fig = go.Figure()
        if regime_details:
            transition_config = RegimeTransitionGaugeConfig(
                transition_prob=regime_details.transition_probability,
                regime_confidence=regime_details.vri_score / 100.0 # Example mapping
            )
            transition_gauge_fig = create_regime_transition_gauge(transition_config)

        # Assemble the component
        return html.Div([
            html.H5(
                volatility_panel_model.panel_config.title,
                className="card-title",
                style=get_unified_text_style("subtitle").model_dump(exclude_none=True)
            ),
            _create_textual_info_section(regime_details),
            html.Div(className="row mt-3", children=[
                *gauge_grid,
                # Add the transition gauge to its own column for emphasis
                html.Div(
                    className="col-lg-12 col-md-12 mb-3",
                    children=dcc.Graph(
                        figure=transition_gauge_fig,
                        config={'displayModeBar': False}
                    )
                )
            ])
        ])
    except Exception as e:
        logger.error(f"Volatility Panel: Failed to render UI. Error: {e}", exc_info=True)
        return create_placeholder_card("Volatility & Regime", "UI rendering failed.")


# --- 3. Helper Functions ---

def _get_expert_details(bundle, expert_id, model_class):
    """
    Safely extracts and validates expert details and confidence from the bundle.
    Returns a tuple of (details_model, confidence_score).
    """
    for resp in bundle.expert_responses:
        if resp.expert_id == expert_id and resp.success and resp.response_data.details:
            try:
                details = model_class.model_validate(resp.response_data.details)
                confidence = resp.response_data.confidence
                return details, confidence
            except Exception as e:
                logger.warning(f"Could not validate details for expert {expert_id}: {e}")
    return None, 0.0

def _create_textual_info_section(details: Optional[MarketRegimeAnalysisDetails]) -> html.Div:
    """Creates a small section for displaying non-gauge textual information."""
    if not details:
        return html.Div()

    regime_style = 'success' if 'BULL' in details.regime_name.upper() else 'danger' if 'BEAR' in details.regime_name.upper() else 'warning'
    vol_style = 'danger' if details.volatility_level in ['HIGH', 'EXTREME'] else 'success'

    return html.Div(
        className="d-flex justify-content-around align-items-center p-2",
        style={
            "backgroundColor": "rgba(0,0,0,0.2)",
            "borderRadius": "8px",
            "border": f"1px solid {AI_COLORS['card_border']}"
        },
        children=[
            html.Div([
                html.Span("Regime: ", style={"color": AI_COLORS['muted']}),
                html.Span(
                    details.regime_name.replace('_', ' ').title(),
                    style=get_unified_badge_style(regime_style).model_dump(exclude_none=True)
                )
            ]),
            html.Div([
                html.Span("Volatility: ", style={"color": AI_COLORS['muted']}),
                html.Span(
                    details.volatility_level,
                    style=get_unified_badge_style(vol_style).model_dump(exclude_none=True)
                )
            ]),
            html.Div([
                html.Span("Trend: ", style={"color": AI_COLORS['muted']}),
                html.Span(details.trend_direction, style={"color": AI_COLORS['dark'], "fontWeight": "bold"})
            ])
        ]
    )
