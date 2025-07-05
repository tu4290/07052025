# dashboard_application/app_main.py
# EOTS v2.5 - S-GRADE, AUTHORITATIVE APPLICATION ENTRY POINT (REFACTORED)

import logging
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dash.dash import Dash
import dash_bootstrap_components as dbc

# --- [START] EOTS V2.5 CORE IMPORTS (CORRECTED) ---
# All imports are now absolute from the project root, which is added to sys.path
# by the runner script. This resolves all ModuleNotFoundError issues.
from utils.config_manager_v2_5 import ConfigManagerV2_5
from core_analytics_engine.its_orchestrator_v2_5 import ITSOrchestratorV2_5
from data_management.enhanced_cache_manager_v2_5 import EnhancedCacheManagerV2_5
from data_management.database_manager_v2_5 import DatabaseManagerV2_5

# EOTS V2.5 Dashboard Imports (also now absolute)
import dashboard_application.layout_manager_v2_5 as layout_manager_v2_5
import dashboard_application.callback_manager_v2_5 as callback_manager_v2_5
import dashboard_application.utils_dashboard_v2_5 as utils_dashboard_v2_5

# PYDANTIC-FIRST: Import data models for universal filtering
# --- [END] EOTS V2.5 CORE IMPORTS (CORRECTED) ---

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# 🔒🔒🔒 GLOBAL REFERENCE - ABSOLUTELY MANDATORY - DO NOT MODIFY 🔒🔒🔒
# ═══════════════════════════════════════════════════════════════════════════════════════════════
#
# ⚠️⚠️⚠️ THIS GLOBAL REFERENCE IS USED BY ALL DASHBOARD MODES ⚠️⚠️⚠️
#
# DO NOT CHANGE THE VARIABLE NAME
# DO NOT REASSIGN TO A DIFFERENT FUNCTION
# DO NOT CREATE ALTERNATIVE FILTERING REFERENCES
# DO NOT BYPASS THIS REFERENCE
#
# THIS ENSURES ALL DASHBOARD MODES USE THE SAME HARDWIRED FILTERING FUNCTION
#
# ═══════════════════════════════════════════════════════════════════════════════════════════════

UNIVERSAL_FILTERING_FUNCTION = utils_dashboard_v2_5.apply_universal_filtering_hardwired

def create_dash_app(config_manager: ConfigManagerV2_5, orchestrator) -> Dash:
    """
    🚀 PYDANTIC-FIRST: Create the Dash application with hardwired universal filtering.
    """
    logger.info("🚀 Creating EOTS v2.5 Dashboard Application...")
    
    # Initialize the Dash app with dark theme
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.DARKLY,
            'https://use.fontawesome.com/releases/v5.15.4/css/all.css'
        ],
        suppress_callback_exceptions=True,
        title="EOTS v2.5 - Elite Options Trading System"
    )
    
    # Create layout with hardwired filtering
    app.layout = layout_manager_v2_5.create_master_layout(config_manager)
    
    # Register callbacks with hardwired filtering
    callback_manager_v2_5.register_v2_5_callbacks(app, orchestrator, config_manager)
    
    return app

def run_dashboard(config_manager: ConfigManagerV2_5, orchestrator) -> None:
    """
    🚀 PYDANTIC-FIRST: Run the Dash application with hardwired universal filtering.
    """
    logger.info("🚀 Starting EOTS v2.5 Dashboard...")
    
    # Create and run the Dash app
    app = create_dash_app(config_manager, orchestrator)
    
    logger.info("🌐 Dashboard server starting on http://0.0.0.0:8050")
    logger.info("🔗 Access your dashboard at: http://localhost:8050")
    
    # Run the server - this should block and keep the application running
    try:
        app.run(
            debug=False,  # Set to False for production
            host='0.0.0.0', 
            port='8050',  # Convert to string for type compatibility
            dev_tools_hot_reload=False,  # Disable hot reload for stability
            threaded=True  # Enable threading for better performance
        )
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard server: {e}")
        raise

def main():
    """
    🚀 PYDANTIC-FIRST: Main entry point with hardwired universal filtering.
    """
    try:
        # Initialize configuration
        config_manager = ConfigManagerV2_5()

        # Initialize database manager
        db_manager = DatabaseManagerV2_5(config_manager=config_manager)
        # Removed the unnecessary connect() call since connection is automatic on initialization

        # Initialize enhanced cache manager
        enhanced_cache = EnhancedCacheManagerV2_5()
        logger.info("✅ Initialized EnhancedCacheManagerV2_5")

        # Initialize orchestrator with db_manager and enhanced cache
        orchestrator = ITSOrchestratorV2_5(
            config_manager=config_manager,
            db_manager=db_manager,  # Pass the initialized db_manager
            enhanced_cache=enhanced_cache
        )
        logger.info("✅ Initialized ITSOrchestratorV2_5 with database manager")

        # Run the dashboard
        run_dashboard(config_manager, orchestrator)

    except Exception as e:
        logger.error(f"❌ Failed to start EOTS v2.5: {e}")
        raise

if __name__ == "__main__":
    main()