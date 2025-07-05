"""
Enhanced AI Hub Layout - Consolidated & Pydantic-First v2.5
==========================================================

This is the main consolidated layout for the enhanced AI Hub with:
- Pydantic-first validation against EOTS schemas
- 3-row optimized layout (Command Center, Core Metrics, System Health)
- Persistent Market Regime MOE integration
- Modular component architecture

Author: EOTS v2.5 Development Team
Version: 2.5.1 (Consolidated)
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from dash import html

# EOTS Schema imports - Pydantic-first validation
from data_models import FinalAnalysisBundleV2_5
from data_models import ProcessedDataBundleV2_5

# Import existing components - preserve all dependencies
from .components import (
    AI_COLORS, AI_SPACING, AI_EFFECTS,
    get_card_style
)


# Import modular layout components
from .layouts_panels import (
    create_trade_recommendations_panel,
    create_market_analysis_panel,
    create_legendary_market_compass_panel
)

from .layouts_metrics import (
    create_flow_intelligence_container,
    create_volatility_gamma_container,
    create_custom_formulas_container
)

from .layouts_health import (
    create_data_pipeline_monitor,
    create_huihui_experts_monitor,
    create_performance_monitor,
    create_alerts_status_monitor
)

from .layouts_regime import (
    create_persistent_regime_display,
    PersistentMarketRegimeMOE
)

# Preserve existing imports for compatibility


from .component_compliance_tracker_v2_5 import (
    track_data_access, DataSourceType
)
from .compliance_decorators_v2_5 import track_compliance

logger = logging.getLogger(__name__)

# Initialize persistent Market Regime MOE
_regime_moe_instance = None

def get_regime_moe() -> PersistentMarketRegimeMOE:
    """Get or create the persistent Market Regime MOE instance."""
    global _regime_moe_instance
    if _regime_moe_instance is None:
        _regime_moe_instance = PersistentMarketRegimeMOE()
    return _regime_moe_instance

@track_compliance("enhanced_ai_hub_layout", "Enhanced AI Hub Layout")
def create_enhanced_ai_hub_layout(
    bundle_data: FinalAnalysisBundleV2_5, 
    ai_settings: Dict[str, Any], 
    symbol: str, 
    db_manager=None
) -> html.Div:
    """
    Create the enhanced 3-row AI Hub layout with Pydantic-first validation.
    
    Args:
        bundle_data: Validated FinalAnalysisBundleV2_5 from EOTS
        ai_settings: AI configuration settings
        symbol: Trading symbol (e.g., 'SPY')
        db_manager: Database manager instance
        
    Returns:
        html.Div: Complete AI Hub layout
    """
    try:
        # Pydantic-first validation - ensure we have proper schema
        if not isinstance(bundle_data, FinalAnalysisBundleV2_5):
            logger.error(f"Invalid bundle_data type: {type(bundle_data)}")
            return create_error_layout("Invalid data bundle - Pydantic validation failed")
        
        # Track data access for compliance
        track_data_access("enhanced_ai_hub_layout", DataSourceType.FINAL_ANALYSIS, 
                         bundle_data, metadata={"symbol": symbol})
        
        # Extract validated data using Pydantic model access
        processed_data = bundle_data.processed_data_bundle
        if not processed_data or not isinstance(processed_data, ProcessedDataBundleV2_5):
            logger.warning("No valid processed data bundle available")
            return create_error_layout("No processed data available")
        
        # Get regime MOE instance and update with current data
        regime_moe = get_regime_moe()
        regime_moe.update_with_bundle_data(bundle_data)
        
        # Create the enhanced 3-row layout
        return html.Div([
            # Control Panel with Persistent Regime Display
            create_control_panel_with_regime(regime_moe, symbol),
            
            # Row 1: Command Center
            create_row_1_command_center(bundle_data, ai_settings, symbol, db_manager),
            
            # Row 2: Core Metrics (3 containers)
            create_row_2_core_metrics(bundle_data, ai_settings, symbol),
            
            # Row 3: System Health Monitor (4 containers)
            create_row_3_system_health(bundle_data, ai_settings, symbol, db_manager)
            
        ], className="enhanced-ai-hub-container", style={
            "background": AI_EFFECTS['gradient_bg'],
            "minHeight": "100vh",
            "padding": AI_SPACING['md']
        })
        
    except Exception as e:
        logger.error(f"Error creating enhanced AI Hub layout: {str(e)}")
        return create_error_layout(f"Layout creation failed: {str(e)}")

def create_control_panel_with_regime(regime_moe: PersistentMarketRegimeMOE, symbol: str) -> html.Div:
    """Create control panel with persistent regime display."""
    try:
        regime_data = regime_moe.get_regime_display_data()
        
        return html.Div([
            html.Div([
                # Mode selector buttons
                html.Div([
                    html.Button("AI Hub", className="btn btn-primary active", id="mode-ai-hub"),
                    html.Button("Trading", className="btn btn-outline-primary", id="mode-trading"),
                    html.Button("Analysis", className="btn btn-outline-primary", id="mode-analysis"),
                    html.Button("Settings", className="btn btn-outline-primary", id="mode-settings")
                ], className="btn-group mode-selector"),
                
                # Persistent regime display
                create_persistent_regime_display(regime_data, symbol)
                
            ], className="d-flex justify-content-between align-items-center")
        ], className="control-panel", style={
            "background": AI_COLORS['card_bg'],
            "border": AI_EFFECTS['glass_border'],
            "borderRadius": AI_EFFECTS['border_radius_sm'],
            "padding": AI_SPACING['md'],
            "marginBottom": AI_SPACING['lg'],
            "backdropFilter": AI_EFFECTS['backdrop_blur']
        })
        
    except Exception as e:
        logger.error(f"Error creating control panel: {str(e)}")
        return html.Div("Control panel unavailable")

def create_row_1_command_center(
    bundle_data: FinalAnalysisBundleV2_5, 
    ai_settings: Dict[str, Any], 
    symbol: str, 
    db_manager=None
) -> html.Div:
    """Create Row 1: Command Center with 50/50 split layout."""
    try:
        return html.Div([
            html.Div([
                # Left Half: Analysis & Recommendations (50%)
                html.Div([
                    # Top Quadrant: Trade Recommendations
                    html.Div([
                        create_trade_recommendations_panel(bundle_data, ai_settings, symbol)
                    ], className="mb-3", style={"height": "48%"}),
                    
                    # Bottom Quadrant: Market Analysis
                    html.Div([
                        create_market_analysis_panel(bundle_data, ai_settings, symbol)
                    ], style={"height": "48%"})
                    
                ], className="col-md-6 pr-2"),
                
                # Right Half: Legendary Market Compass (50%)
                html.Div([
                    create_legendary_market_compass_panel(bundle_data, ai_settings, symbol)
                ], className="col-md-6 pl-2")
                
            ], className="row h-100")
        ], className="command-center-row", style={
            "height": "400px",
            "marginBottom": AI_SPACING['lg']
        })
        
    except Exception as e:
        logger.error(f"Error creating command center: {str(e)}")
        return create_error_layout("Command center unavailable")

def create_row_2_core_metrics(
    bundle_data: FinalAnalysisBundleV2_5, 
    ai_settings: Dict[str, Any], 
    symbol: str
) -> html.Div:
    """Create Row 2: Core Metrics with 3 containers (33% each)."""
    try:
        return html.Div([
            html.Div([
                # Container 1: Flow Intelligence (33%)
                html.Div([
                    create_flow_intelligence_container(bundle_data, symbol)
                ], className="col-md-4 pr-2"),
                
                # Container 2: Volatility & Gamma (33%)
                html.Div([
                    create_volatility_gamma_container(bundle_data, symbol)
                ], className="col-md-4 px-1"),
                
                # Container 3: Custom Formulas (33%)
                html.Div([
                    create_custom_formulas_container(bundle_data, symbol)
                ], className="col-md-4 pl-2")
                
            ], className="row h-100")
        ], className="core-metrics-row", style={
            "height": "300px",
            "marginBottom": AI_SPACING['lg']
        })
        
    except Exception as e:
        logger.error(f"Error creating core metrics: {str(e)}")
        return create_error_layout("Core metrics unavailable")

def create_row_3_system_health(
    bundle_data: FinalAnalysisBundleV2_5, 
    ai_settings: Dict[str, Any], 
    symbol: str, 
    db_manager=None
) -> html.Div:
    """Create Row 3: System Health Monitor with 4 containers (25% each)."""
    try:
        return html.Div([
            html.Div([
                # Container 1: Data Pipeline (25%)
                html.Div([
                    create_data_pipeline_monitor(bundle_data, symbol)
                ], className="col-md-3 pr-1"),
                
                # Container 2: HuiHui Experts (25%)
                html.Div([
                    create_huihui_experts_monitor(bundle_data, symbol, db_manager)
                ], className="col-md-3 px-1"),
                
                # Container 3: Performance (25%)
                html.Div([
                    create_performance_monitor(bundle_data, symbol)
                ], className="col-md-3 px-1"),
                
                # Container 4: Alerts & Status (25%)
                html.Div([
                    create_alerts_status_monitor(bundle_data, symbol, db_manager)
                ], className="col-md-3 pl-1")
                
            ], className="row h-100")
        ], className="system-health-row", style={
            "height": "250px"
        })
        
    except Exception as e:
        logger.error(f"Error creating system health monitor: {str(e)}")
        return create_error_layout("System health monitor unavailable")

def create_error_layout(error_message: str) -> html.Div:
    """Create error layout with proper styling."""
    return html.Div([
        html.Div([
            html.H4("⚠️ AI Hub Error", className="text-warning"),
            html.P(error_message, className="text-muted"),
            html.Small(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}", 
                      className="text-muted")
        ], className="text-center p-4")
    ], style=get_card_style('danger'))

# Utility function for Pydantic validation
def validate_bundle_data(bundle_data: Any) -> Optional[FinalAnalysisBundleV2_5]:
    """Validate and return properly typed bundle data."""
    try:
        if isinstance(bundle_data, FinalAnalysisBundleV2_5):
            return bundle_data
        elif isinstance(bundle_data, dict):
            return FinalAnalysisBundleV2_5(**bundle_data)
        else:
            logger.error(f"Cannot validate bundle_data of type: {type(bundle_data)}")
            return None
    except Exception as e:
        logger.error(f"Bundle data validation failed: {str(e)}")
        return None

# Export main function for backward compatibility
def create_ai_dashboard_layout(bundle_data, ai_settings, symbol, db_manager=None):
    """Backward compatibility wrapper for existing code."""
    validated_bundle = validate_bundle_data(bundle_data)
    if validated_bundle is None:
        return create_error_layout("Data validation failed")
    
    return create_enhanced_ai_hub_layout(validated_bundle, ai_settings, symbol, db_manager)

