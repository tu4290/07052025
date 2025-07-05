"""
Enhanced AI Hub Layout - Main Assembly File v2.5.1
===================================================
This is the main layout entry point. It assembles the dashboard by calling
the modular panel functions from the `panels` directory.

Author: EOTS v2.5 Development Team (Refactored)
Version: 2.5.1
"""
import logging
from typing import Dict, Any
from dash import html

from data_models import FinalAnalysisBundleV2_5
from .components import AI_EFFECTS, AI_SPACING, create_placeholder_card
from compliance_decorators_v2_5 import track_compliance

# Import the refactored panel creation functions from their new, organized locations
from .enhanced_ai_hub_layout import (
    create_row_1_command_center,
    create_row_2_core_metrics,
    create_row_3_system_health,
    create_control_panel_with_regime
)
from .layouts_regime import PersistentMarketRegimeMOE

logger = logging.getLogger(__name__)

# Use a function to manage the singleton instance of the MOE
def get_regime_moe() -> PersistentMarketRegimeMOE:
    """Initializes and/or returns the singleton MOE instance."""
    if not hasattr(get_regime_moe, "instance"):
        get_regime_moe.instance = PersistentMarketRegimeMOE()
    return get_regime_moe.instance

@track_compliance("enhanced_ai_hub_layout", "Enhanced AI Hub Layout")
def create_enhanced_ai_hub_layout(
    bundle_data: FinalAnalysisBundleV2_5,
    symbol: str,
    ai_settings: Dict[str, Any],
    db_manager=None
) -> html.Div:
    """
    Creates the final, enhanced 3-row AI Hub layout by assembling modular panels.
    This function acts as the master assembler for the dashboard UI.
    """
    try:
        # Pydantic-first validation
        if not isinstance(bundle_data, FinalAnalysisBundleV2_5):
            logger.error(f"Invalid bundle_data type: {type(bundle_data)}")
            return create_placeholder_card("Layout Error", "Invalid data bundle provided.")

        # Get the persistent regime MOE and update it with the latest data
        regime_moe = get_regime_moe()
        regime_moe.update_with_bundle_data(bundle_data)

        # --- Layout Assembly ---
        # Each function call here renders a major, self-contained part of the dashboard.
        return html.Div([
            # The system-wide control panel with the persistent regime display
            create_control_panel_with_regime(regime_moe, symbol),
            
            # Row 1: The main command center with the compass and analysis
            create_row_1_command_center(bundle_data, ai_settings, symbol, db_manager),
            
            # Row 2: The three core EOTS metric containers
            create_row_2_core_metrics(bundle_data, ai_settings, symbol),
            
            # Row 3: The four system health and status monitors
            create_row_3_system_health(bundle_data, ai_settings, symbol, db_manager)
            
        ], className="enhanced-ai-hub-container", style={
            "background": AI_EFFECTS.get('gradient_bg', '#0a0e1a'),
            "minHeight": "100vh",
            "padding": AI_SPACING.get('lg', '1.5rem')
        })

    except Exception as e:
        logger.error(f"Fatal error creating enhanced AI Hub layout: {e}", exc_info=True)
        return create_placeholder_card("Fatal Layout Error", f"Could not render dashboard: {e}")