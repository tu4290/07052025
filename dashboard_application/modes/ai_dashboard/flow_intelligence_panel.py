# dashboard_application/modes/ai_dashboard/flow_intelligence_panel.py
"""
Flow Intelligence Panel for EOTS AI Hub v2.5
==============================================

PYDANTIC-FIRST & ZERO DICT ACCEPTANCE.

This module provides a dedicated, Pydantic-first panel for visualizing real-time
options flow intelligence sourced directly from the HuiHui `options_flow_expert`.
It replaces the static metrics container from the legacy `layouts_metrics.py`.

Key Features:
1.  **Direct HuiHui Integration**: Consumes and validates data from the
    `options_flow_expert` MoE.
2.  **Strict Pydantic Models**: Uses `MetricsPanelModel`, `MetricDisplayModel`, and
    `GaugeConfigModel` for end-to-end type safety.
3.  **Rich Visualization**: Displays key flow metrics like VAPI-FA, DWFD, and
    Institutional Probability using sophisticated gauges.
4.  **State-Driven UI**: The panel's appearance and data are driven entirely by
    the `AIHubStateModel`.

The core flow is:
- `transform_flow_expert_to_panel_model()`: Transforms expert data into a `MetricsPanelModel`.
- `create_flow_intelligence_panel()`: Renders the Dash component from the model.

Author: EOTS v2.5 Development Team
Version: 2.5.2
"""

import logging
from typing import Optional

from dash import html, dcc

# EOTS Pydantic Models: The single source of truth for all data structures
from data_models import (
    MOEUnifiedResponseV2_5,
    OptionsFlowAnalysisDetails,
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
    get_card_style,
    get_unified_text_style,
    get_unified_badge_style,
    AI_COLORS
)
from .visualizations import create_metric_gauge

logger = logging.getLogger(__name__)


# --- 1. Core Data Transformation Logic ---

def transform_flow_expert_to_panel_model(
    analysis_bundle: Optional[MOEUnifiedResponseV2_5]
) -> MetricsPanelModel:
    """
    Transforms data from the `options_flow_expert` within a MOE bundle into a
    strictly validated `MetricsPanelModel`.

    This function isolates data processing from UI rendering, adhering to a clean
    separation of concerns.

    Args:
        analysis_bundle: The unified response from the HuiHui experts (MOE).

    Returns:
        A fully populated and validated MetricsPanelModel for the UI.
    """
    panel_config = PanelConfigModel(
        id="flow-intelligence-panel",
        title="📊 Flow Intelligence",
        panel_type=PanelType.FLOW_INTELLIGENCE
    )

    if not analysis_bundle:
        panel_config.status = ComponentStatus.LOADING
        return MetricsPanelModel(panel_config=panel_config, metrics=[])

    try:
        # Safely find and validate the options flow expert's response
        flow_details = _get_expert_details(analysis_bundle, "options_flow", OptionsFlowAnalysisDetails)

        if not flow_details:
            logger.warning("Flow Panel: Options Flow Expert details not found or failed.")
            panel_config.status = ComponentStatus.WARNING
            return MetricsPanelModel(panel_config=panel_config, metrics=[])

        # Create a list of metric displays from the validated expert details
        metrics_list = [
            MetricDisplayModel(
                name="VAPI-FA Score",
                value=flow_details.vapi_fa_score,
                gauge_config=GaugeConfigModel(
                    value=flow_details.vapi_fa_score,
                    title="VAPI-FA (Vol-Adj Price Impact)",
                    range_min=-100, range_max=100, height=180
                )
            ),
            MetricDisplayModel(
                name="DWFD Score",
                value=flow_details.dwfd_score,
                gauge_config=GaugeConfigModel(
                    value=flow_details.dwfd_score,
                    title="DWFD (Dealer vs Whale Flow)",
                    range_min=-100, range_max=100, height=180
                )
            ),
            MetricDisplayModel(
                name="Institutional Probability",
                value=flow_details.institutional_probability * 100, # Display as percentage
                gauge_config=GaugeConfigModel(
                    value=flow_details.institutional_probability * 100,
                    title="Institutional Probability",
                    range_min=0, range_max=100, height=180
                )
            )
        ]
        
        # Add non-gauge textual data to the panel model for rendering
        panel_config.status = ComponentStatus.OK
        panel_config.last_updated = analysis_bundle.timestamp
        
        # Store extra details in the panel_config model itself if needed, or use a new model
        # For now, we pass it directly to the renderer. A better approach might be
        # to add `extra_details: Optional[BaseModel]` to the MetricsPanelModel.
        
        return MetricsPanelModel(
            panel_config=panel_config,
            metrics=metrics_list
        )

    except Exception as e:
        logger.error(f"Flow Panel: Failed to transform expert data. Error: {e}", exc_info=True)
        panel_config.status = ComponentStatus.ERROR
        return MetricsPanelModel(panel_config=panel_config, metrics=[])


# --- 2. UI Rendering Logic ---

def create_flow_intelligence_panel(
    flow_panel_model: Optional[MetricsPanelModel],
    flow_details: Optional[OptionsFlowAnalysisDetails] # Passed separately for textual info
) -> html.Div:
    """
    Assembles the Flow Intelligence panel from a validated `MetricsPanelModel`.
    This is a pure rendering function with no data processing logic.

    Args:
        flow_panel_model: The Pydantic model containing data and config for the panel.
        flow_details: The validated details from the expert for textual display.

    Returns:
        A Dash html.Div representing the complete Flow Intelligence panel.
    """
    if not flow_panel_model or flow_panel_model.panel_config.status != ComponentStatus.OK:
        message = "Awaiting flow analysis data..."
        if flow_panel_model and flow_panel_model.panel_config.status == ComponentStatus.ERROR:
            message = "An error occurred processing flow data."
        elif flow_panel_model and flow_panel_model.panel_config.status == ComponentStatus.WARNING:
            message = "Flow expert data is unavailable or failed."
        return create_placeholder_card("Flow Intelligence", message)

    try:
        # Create the grid of gauges from the metrics list in the model
        gauge_grid = [
            html.Div(
                className="col-lg-4 col-md-6 mb-3",
                children=dcc.Graph(
                    figure=create_metric_gauge(metric.gauge_config),
                    config={'displayModeBar': False}
                )
            ) for metric in flow_panel_model.metrics
        ]

        # Assemble the component with title, textual info, and gauge grid
        return html.Div([
            html.H5(
                flow_panel_model.panel_config.title,
                className="card-title",
                style=get_unified_text_style("subtitle").model_dump(exclude_none=True)
            ),
            _create_textual_info_section(flow_details),
            html.Div(className="row mt-3", children=gauge_grid)
        ])
    except Exception as e:
        logger.error(f"Flow Panel: Failed to render UI. Error: {e}", exc_info=True)
        return create_placeholder_card("Flow Intelligence", "UI rendering failed.")


# --- 3. Helper Functions ---

def _get_expert_details(bundle, expert_id, model_class):
    """Safely extracts and validates expert details from the bundle."""
    for resp in bundle.expert_responses:
        if resp.expert_id == expert_id and resp.success and resp.response_data.details:
            try:
                # The details can be a dict from JSON, so we must validate it
                return model_class.model_validate(resp.response_data.details)
            except Exception as e:
                logger.warning(f"Could not validate details for expert {expert_id}: {e}")
    return None

def _create_textual_info_section(details: Optional[OptionsFlowAnalysisDetails]) -> html.Div:
    """Creates a small section for displaying non-gauge textual information."""
    if not details:
        return html.Div()

    flow_type_style = 'success' if details.flow_type == 'BULLISH' else 'danger' if details.flow_type == 'BEARISH' else 'warning'
    
    return html.Div(
        className="d-flex justify-content-around align-items-center p-2",
        style={
            "backgroundColor": "rgba(0,0,0,0.2)",
            "borderRadius": "8px",
            "border": f"1px solid {AI_COLORS['card_border']}"
        },
        children=[
            html.Div([
                html.Span("Flow Type: ", style={"color": AI_COLORS['muted']}),
                html.Span(
                    details.flow_type,
                    style=get_unified_badge_style(flow_type_style).model_dump(exclude_none=True)
                )
            ]),
            html.Div([
                html.Span("Intensity: ", style={"color": AI_COLORS['muted']}),
                html.Span(details.flow_intensity, style={"color": AI_COLORS['dark'], "fontWeight": "bold"})
            ])
        ]
    )
