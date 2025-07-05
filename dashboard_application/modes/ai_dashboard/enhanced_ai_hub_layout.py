"""
Enhanced AI Hub Layout - Consolidated & Pydantic-First v2.5
==========================================================

This file defines the main layout for the AI Hub, refactored to be fully
compliant with the Pydantic-first, ZERO DICT ACCEPTANCE architecture.

Key Architectural Changes:
- The main layout function `create_enhanced_ai_hub_layout` is now a pure
  renderer that accepts a single, validated `AIHubStateModel`.
- A new data transformation function `transform_analysis_to_hub_state` is
  introduced to separate data processing from rendering logic.
- High-priority components (Market Compass, AI Recommendations) are integrated
  using their own dedicated modules and Pydantic models.
- All `Dict` and `Any` type hints for data structures have been eliminated in
  favor of strict Pydantic models.

Author: EOTS v2.5 Development Team
Version: 2.5.2 (Pydantic-First Refactor)
"""

import logging
from typing import Optional

from dash import html, dcc
from pydantic import ValidationError

# EOTS Pydantic Models - The single source of truth for data structures
from data_models import (
    AIHubStateModel,
    MOEUnifiedResponseV2_5,
    FinalAnalysisBundleV2_5,
    ProcessedUnderlyingAggregatesV2_5,
    MarketCompassModel,
    AIRecommendationsPanelModel
)

# Import new, Pydantic-compliant component modules
from .market_compass_component import (
    create_legendary_market_compass_panel,
    create_market_compass_model_from_analysis
)
from .ai_recommendations_component import (
    create_ai_recommendations_panel,
    create_recommendations_model_from_analysis
)
# --- NEW PANELS ---------------------------------------------------------------
from .flow_intelligence_panel import (
    transform_flow_expert_to_panel_model,
    create_flow_intelligence_panel
)
from .volatility_gamma_panel import (
    transform_regime_expert_to_panel_model,
    create_volatility_gamma_panel
)

# Import shared components and styling constants
from .components import create_placeholder_card, get_card_style, AI_COLORS, AI_SPACING

logger = logging.getLogger(__name__)


# --- 1. Data Transformation (To be called from a callback) ---

def transform_analysis_to_hub_state(
    analysis_bundle: Optional[MOEUnifiedResponseV2_5],
    raw_bundle: Optional[FinalAnalysisBundleV2_5],
    symbol: str
) -> AIHubStateModel:
    """
    Transforms raw analysis data from the orchestrator into a validated AIHubStateModel.
    This function separates data processing from the layout rendering.

    Args:
        analysis_bundle: The unified response from the HuiHui experts (MOE).
        raw_bundle: The original FinalAnalysisBundleV2_5 for underlying data.
        symbol: The target trading symbol.

    Returns:
        A fully populated and validated AIHubStateModel for the UI.
    """
    if not analysis_bundle or not raw_bundle or not symbol:
        logger.warning("Insufficient data to transform into AI Hub state. Returning empty state.")
        return AIHubStateModel(target_symbol=symbol, error_message="Insufficient analysis data.")

    try:
        underlying_price = raw_bundle.processed_data_bundle.underlying_data_enriched.price or 0.0

        # Create the data models for the high-priority components
        compass_model = create_market_compass_model_from_analysis(analysis_bundle, symbol)
        recommendations_model = create_recommendations_model_from_analysis(analysis_bundle, symbol, underlying_price)

        # --- NEW: Flow & Volatility panel models --------------------------------
        flow_panel_model = transform_flow_expert_to_panel_model(analysis_bundle)
        volatility_panel_model = transform_regime_expert_to_panel_model(analysis_bundle)

        # Assemble the master state model
        hub_state = AIHubStateModel(
            target_symbol=symbol,
            last_updated=analysis_bundle.timestamp,
            market_compass=compass_model,
            ai_recommendations=recommendations_model,
            flow_intelligence_metrics=flow_panel_model,
            volatility_gamma_metrics=volatility_panel_model,
            raw_analysis_bundle=analysis_bundle
            # NOTE: Other panels (metrics, health) would be populated here in a full implementation
        )
        return hub_state

    except (ValidationError, AttributeError, Exception) as e:
        logger.error(f"Failed to transform analysis into AI Hub state for {symbol}: {e}", exc_info=True)
        return AIHubStateModel(
            target_symbol=symbol,
            error_message=f"Data transformation failed: {e}"
        )


# --- 2. Main Layout Rendering (Purely UI-focused) ---

def create_enhanced_ai_hub_layout(hub_state: AIHubStateModel) -> html.Div:
    """
    Create the enhanced 3-row AI Hub layout from a validated AIHubStateModel.
    This is a pure rendering function with no data processing logic.

    Args:
        hub_state: A validated AIHubStateModel containing all data for the UI.

    Returns:
        A Dash html.Div representing the complete AI Hub layout.
    """
    if hub_state.error_message:
        return create_error_layout(hub_state.error_message)

    return html.Div([
        # Row 1: Command Center (Market Compass & AI Recommendations)
        create_row_1_command_center(hub_state),

        # Row 2: Core Metrics (Placeholder)
        create_row_2_core_metrics(hub_state),

        # Row 3: System Health Monitor (Placeholder)
        create_row_3_system_health(hub_state)

    ], className="enhanced-ai-hub-container", style={
        "padding": AI_SPACING['lg']
    })


# --- 3. Row-Specific Layout Functions ---

def create_row_1_command_center(hub_state: AIHubStateModel) -> html.Div:
    """
    Create Row 1: Command Center, featuring the high-priority Market Compass
    and AI Recommendations panels.

    Args:
        hub_state: The master state model for the AI Hub.

    Returns:
        A Dash html.Div for the command center row.
    """
    try:
        return html.Div(
            className="row",
            children=[
                # Left Side: AI-Driven Trade Recommendations
                html.Div(
                    className="col-lg-6 mb-4",
                    children=html.Div(
                        className="card h-100",
                        style=get_card_style('default').model_dump(exclude_none=True),
                        children=create_ai_recommendations_panel(hub_state.ai_recommendations)
                    )
                ),
                # Right Side: Legendary Market Compass
                html.Div(
                    className="col-lg-6 mb-4",
                    children=html.Div(
                        className="card h-100",
                        style=get_card_style('default').model_dump(exclude_none=True),
                        children=create_legendary_market_compass_panel(hub_state.market_compass)
                    )
                )
            ]
        )
    except Exception as e:
        logger.error(f"Error creating command center row: {e}", exc_info=True)
        return create_error_layout("Command center UI render failed.")


def create_row_2_core_metrics(hub_state: AIHubStateModel) -> html.Div:
    """
    Create Row 2: Core Metrics – Flow Intelligence, Volatility & Gamma,
    and a placeholder for Custom-Formulas.
    """
    # Helper to fetch expert details from the unified bundle
    def _get_details(bundle, expert_id, model_cls):
        if not bundle:
            return None
        for resp in bundle.expert_responses:
            if resp.expert_id == expert_id and resp.success:
                try:
                    return model_cls.model_validate(resp.response_data.details)
                except Exception:  # pragma: no cover
                    return None
        return None

    from data_models import OptionsFlowAnalysisDetails, MarketRegimeAnalysisDetails
    flow_details = _get_details(hub_state.raw_analysis_bundle, "options_flow", OptionsFlowAnalysisDetails)
    regime_details = _get_details(hub_state.raw_analysis_bundle, "market_regime", MarketRegimeAnalysisDetails)

    flow_panel = create_flow_intelligence_panel(hub_state.flow_intelligence_metrics, flow_details)
    vol_panel = create_volatility_gamma_panel(hub_state.volatility_gamma_metrics, regime_details)
    custom_placeholder = create_placeholder_card("Custom Formulas", "Component not yet implemented.")

    return html.Div(className="row", children=[
        html.Div(className="col-md-4 mb-4", children=flow_panel),
        html.Div(className="col-md-4 mb-4", children=vol_panel),
        html.Div(className="col-md-4 mb-4", children=custom_placeholder),
    ])


def create_row_3_system_health(hub_state: AIHubStateModel) -> html.Div:
    """
    Create Row 3: System Health Monitor. (Currently placeholders).
    A full implementation would render panels for Data Pipeline, HuiHui Experts, etc.
    """
    return html.Div(
        className="row",
        children=[
            html.Div(className="col-md-3 mb-4", children=create_placeholder_card("Data Pipeline", "Component not yet implemented.")),
            html.Div(className="col-md-3 mb-4", children=create_placeholder_card("HuiHui Experts", "Component not yet implemented.")),
            html.Div(className="col-md-3 mb-4", children=create_placeholder_card("Performance", "Component not yet implemented.")),
            html.Div(className="col-md-3 mb-4", children=create_placeholder_card("Alerts & Status", "Component not yet implemented.")),
        ]
    )


# --- 4. Utility Functions ---

def create_error_layout(error_message: str) -> html.Div:
    """Create a styled error layout to display critical UI failures."""
    return html.Div(
        className="card",
        style=get_card_style('danger').model_dump(exclude_none=True),
        children=html.Div(
            className="card-body text-center",
            children=[
                html.H4("⚠️ AI Hub Layout Error", className="text-danger"),
                html.P(error_message),
            ]
        )
    )

